"""Build a REAL-photo circuit-fault YOLO set.

Target classes (what the user asked for):
  0 complete         — intact circuit region / defect-free crop
  1 incomplete       — open circuit (broken trace / gap)
  2 wires_touching   — short (two conductors touching)
  3 damage           — mouse-bite, spur, missing hole, spurious copper, broken cable

Sources (photographs, not drawings):
  - PKU-Market-PCB / HRIPCB (RobotHuman/PCB_defect) — color PCB photos
  - DeepPCB YOLO export (thangkt/PCB-Prune-YOLO-DeepPCB) — real CCD traces
  - RF100 cable-damage — real damaged cables
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public" / "circuit-faults"
PKU_RAW = ROOT / "data" / "public" / "pku-pcb-raw"
DEEPPCB_RAW = ROOT / "data" / "public" / "deeppcb-raw"
CABLE = ROOT / "data" / "public" / "cable-damage"

CLASSES = ["complete", "incomplete", "wires_touching", "damage"]

NAME_TO_ID: dict[str, int] = {
    "complete": 0,
    "normal": 0,
    "good": 0,
    "temp": 0,
    "template": 0,
    "open": 1,
    "open_circuit": 1,
    "open-circuit": 1,
    "incomplete": 1,
    "short": 2,
    "short_circuit": 2,
    "short-circuit": 2,
    "wires_touching": 2,
    "mouse_bite": 3,
    "mousebite": 3,
    "mouse-bite": 3,
    "spur": 3,
    "missing_hole": 3,
    "missing-hole": 3,
    "pin-hole": 3,
    "pin_hole": 3,
    "pinhole": 3,
    "spurious_copper": 3,
    "spurious-copper": 3,
    "copper": 3,
    "break": 3,
    "thunderbolt": 3,
    "damage": 3,
}

# DeepPCB original type ids (1-based): 1 open, 2 short, rest damage
DEEPPCB_TYPE = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}
# YOLO export is 0-based: 0 open, 1 short, 2 mousebite, 3 spur, 4 copper, 5 pin-hole
DEEPPCB_YOLO = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3, 5: 3}

_SKIP_DIR_NAMES = {".cache", "_complete_crops", ".git"}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}"


def _skip_path(path: Path) -> bool:
    return any(part in _SKIP_DIR_NAMES for part in path.parts)


def _parse_voc(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    root = ET.parse(xml_path).getroot()
    lines: list[str] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        box = obj.find("bndbox")
        if name_el is None or box is None:
            continue
        key = _norm(name_el.text or "")
        if key not in NAME_TO_ID:
            continue
        x1 = float(box.findtext("xmin") or 0)
        y1 = float(box.findtext("ymin") or 0)
        x2 = float(box.findtext("xmax") or 0)
        y2 = float(box.findtext("ymax") or 0)
        lines.append(_yolo_line(NAME_TO_ID[key], x1, y1, x2, y2, img_w, img_h))
    return lines


def _voc_xyxy(xml_path: Path) -> list[tuple[float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        boxes.append(
            (
                float(box.findtext("xmin") or 0),
                float(box.findtext("ymin") or 0),
                float(box.findtext("xmax") or 0),
                float(box.findtext("ymax") or 0),
            )
        )
    return boxes


def _find_pku_xml(img: Path) -> Path | None:
    xml = img.with_name(img.stem + "_annotation.xml")
    if xml.exists():
        return xml
    xml = img.with_suffix(".xml")
    if xml.exists():
        return xml
    return None


def _collect_pku(src: Path) -> list[tuple[Path, list[str], str]]:
    items: list[tuple[Path, list[str], str]] = []
    if not src.exists():
        return items
    for img in src.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if _skip_path(img):
            continue
        xml = _find_pku_xml(img)
        if xml is None:
            continue
        with Image.open(img) as im:
            w, h = im.size
        labels = _parse_voc(xml, w, h)
        if labels:
            items.append((img, labels, "pku"))
    return items


def _window_hits(win: tuple[float, float, float, float], boxes: list[tuple[float, float, float, float]], pad: float) -> bool:
    wx1, wy1, wx2, wy2 = win
    for x1, y1, x2, y2 in boxes:
        ix1 = max(wx1, x1 - pad)
        iy1 = max(wy1, y1 - pad)
        ix2 = min(wx2, x2 + pad)
        iy2 = min(wy2, y2 + pad)
        if ix2 > ix1 and iy2 > iy1:
            return True
    return False


def _collect_pku_complete(src: Path) -> list[tuple[Path, list[str], str]]:
    """Crop defect-free regions of real PKU boards → class `complete`."""
    items: list[tuple[Path, list[str], str]] = []
    if not src.exists():
        return items
    crop_dir = src / "_complete_crops"
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in src.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if _skip_path(img):
            continue
        xml = _find_pku_xml(img)
        if xml is None:
            continue
        with Image.open(img) as im:
            w, h = im.size
            rgb = im.convert("RGB")
        boxes = _voc_xyxy(xml)
        pad = 12.0
        min_side = max(256, int(min(w, h) * 0.35))
        candidates = [
            (0, 0, w / 2, h),
            (w / 2, 0, w, h),
            (0, 0, w, h / 2),
            (0, h / 2, w, h),
            (0, 0, w / 2, h / 2),
            (w / 2, 0, w, h / 2),
            (0, h / 2, w / 2, h),
            (w / 2, h / 2, w, h),
            (w * 0.15, h * 0.15, w * 0.55, h * 0.55),
            (w * 0.45, h * 0.45, w * 0.95, h * 0.95),
        ]
        picked = None
        for win in candidates:
            x1, y1, x2, y2 = win
            if (x2 - x1) < min_side or (y2 - y1) < min_side:
                continue
            if _window_hits(win, boxes, pad):
                continue
            picked = (int(x1), int(y1), int(x2), int(y2))
            break
        if picked is None:
            continue
        crop = rgb.crop(picked)
        cw, ch = crop.size
        if cw < 200 or ch < 200:
            continue
        out_path = crop_dir / f"complete_{n:04d}_{img.stem}.jpg"
        crop.save(out_path, quality=90)
        labels = [_yolo_line(0, cw * 0.02, ch * 0.02, cw * 0.98, ch * 0.98, cw, ch)]
        items.append((out_path, labels, "pku_complete"))
        n += 1
    return items


def _collect_cable(src: Path) -> list[tuple[Path, list[str], str]]:
    items: list[tuple[Path, list[str], str]] = []
    yaml_path = src / "data.yaml"
    if not yaml_path.exists():
        return items
    for split in ("train", "valid", "val", "test"):
        img_dir = src / split / "images"
        lab_dir = src / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                continue
            out_lines: list[str] = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                out_lines.append(" ".join(["3", *parts[1:5]]))
            if out_lines:
                items.append((img, out_lines, "cable"))
    return items


def _label_beside_image(img: Path) -> Path | None:
    replaced = str(img).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
    lab = Path(replaced).with_suffix(".txt")
    if lab.exists():
        return lab
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def _collect_deeppcb_yolo(src: Path) -> list[tuple[Path, list[str], str]]:
    """thangkt YOLO export: 0=open, 1=short, 2–5=damage variants."""
    items: list[tuple[Path, list[str], str]] = []
    if not src.exists():
        return items
    for img in src.rglob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "_test" not in img.name.lower() and "images" not in img.parts:
            continue
        lab = _label_beside_image(img)
        if lab is None:
            continue
        lines: list[str] = []
        for row in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
            bits = row.split()
            if len(bits) < 5:
                continue
            try:
                src_cls = int(float(bits[0]))
            except ValueError:
                continue
            # Skip original x1,y1,x2,y2,type rows (pixel coords, type 1-6).
            if src_cls in DEEPPCB_TYPE and float(bits[1]) > 1.5:
                continue
            dst = DEEPPCB_YOLO.get(src_cls)
            if dst is None:
                continue
            lines.append(" ".join([str(dst), *bits[1:5]]))
        origin = "deeppcb_complete" if "temp" in img.name.lower() else "deeppcb"
        if "temp" in img.name.lower():
            with Image.open(img) as im:
                w, h = im.size
            lines = [_yolo_line(0, w * 0.02, h * 0.02, w * 0.98, h * 0.98, w, h)]
        if lines:
            items.append((img, lines, origin))
    return items


def _collect_deeppcb_legacy(src: Path) -> list[tuple[Path, list[str], str]]:
    """Original DeepPCB: *_test.jpg + txt boxes; *_temp.jpg is the complete template."""
    items: list[tuple[Path, list[str], str]] = []
    if not src.exists():
        return items
    for txt in src.rglob("*.txt"):
        if txt.name.lower() in {"readme.txt", "trainval.txt", "test.txt"}:
            continue
        if "labels" in txt.parts:
            continue
        stem = txt.stem.replace("_test", "").replace("_temp", "")
        parent = txt.parent
        test = None
        temp = None
        for cand in parent.glob(f"{stem}*"):
            n = cand.name.lower()
            if cand.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if "temp" in n:
                temp = cand
            elif "test" in n:
                test = cand
        if test and txt.exists():
            with Image.open(test) as im:
                w, h = im.size
            lines: list[str] = []
            for row in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                bits = [b.strip() for b in row.replace(" ", ",").split(",") if b.strip()]
                if len(bits) < 5:
                    continue
                try:
                    x1, y1, x2, y2 = map(float, bits[:4])
                    typ = int(float(bits[4]))
                except ValueError:
                    continue
                cls = DEEPPCB_TYPE.get(typ)
                if cls is None:
                    continue
                lines.append(_yolo_line(cls, x1, y1, x2, y2, w, h))
            if lines:
                items.append((test, lines, "deeppcb"))
        if temp:
            with Image.open(temp) as im:
                w, h = im.size
            lines = [_yolo_line(0, w * 0.02, h * 0.02, w * 0.98, h * 0.98, w, h)]
            items.append((temp, lines, "deeppcb_complete"))
    return items


def _collect_deeppcb(src: Path) -> list[tuple[Path, list[str], str]]:
    yolo_items = _collect_deeppcb_yolo(src)
    if yolo_items:
        return yolo_items
    return _collect_deeppcb_legacy(src)


def _split(items: list, seed: int) -> dict[str, list]:
    rng = random.Random(seed)
    by_src: dict[str, list] = {}
    for it in items:
        by_src.setdefault(it[2], []).append(it)
    out = {"train": [], "valid": [], "test": []}
    for group in by_src.values():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * 0.12)) if n >= 3 else 0
        n_val = max(1, int(n * 0.13)) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test = max(0, n // 8)
            n_val = max(0, n // 8)
        out["test"].extend(group[:n_test])
        out["valid"].extend(group[n_test : n_test + n_val])
        out["train"].extend(group[n_test + n_val :])
    return out


def _write(dest: Path, splits: dict[str, list]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    counts = {k: 0 for k in CLASSES}
    src_counts: dict[str, int] = {}
    for split, rows in splits.items():
        img_dir = dest / split / "images"
        lab_dir = dest / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i, (img, labels, origin) in enumerate(rows):
            stem = f"{origin}_{i:04d}_{img.stem}"[:80]
            ext = img.suffix.lower() if img.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
            shutil.copy2(img, img_dir / f"{stem}{ext}")
            (lab_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            src_counts[origin] = src_counts.get(origin, 0) + 1
            for line in labels:
                counts[CLASSES[int(line.split()[0])]] += 1
    yaml_text = (
        "names:\n"
        + "".join(f"- {n}\n" for n in CLASSES)
        + f"nc: {len(CLASSES)}\n"
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "note: Real photos. PKU-Market-PCB + DeepPCB + cable-damage. PCB traces ≠ home T&E.\n"
    )
    (dest / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("Wrote", dest)
    print("images by source", src_counts)
    print("boxes by class", counts)
    for split, rows in splits.items():
        print(f"  {split}: {len(rows)}")


def build(seed: int = 7) -> Path:
    items = (
        _collect_pku(PKU_RAW)
        + _collect_pku_complete(PKU_RAW)
        + _collect_cable(CABLE)
        + _collect_deeppcb(DEEPPCB_RAW)
    )
    if not items:
        raise SystemExit(
            "No source images. Run: python -m src.download_circuit_faults"
        )
    splits = _split(items, seed)
    _write(DEST, splits)
    return DEST


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge real PCB + cable photos into circuit-fault YOLO labels.")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    build(parse_args().seed)


if __name__ == "__main__":
    main()

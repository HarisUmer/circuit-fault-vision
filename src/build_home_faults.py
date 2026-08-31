"""Build defect-only YOLO set: open / short / damage. No complete class.

Sources:
  - circuit-faults (drop complete boxes/images)
  - PCB-IND real AOI (open/short/damage map)
  - stripped-wire close-ups (cut/pulled -> damage)
"""
from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public" / "home-faults"
CIRCUIT = ROOT / "data" / "public" / "circuit-faults"
PCB = ROOT / "data" / "public" / "pcb-ind" / "YOLO"
STRIPPED = ROOT / "data" / "public" / "stripped-wire"
SOCKETS = ROOT / "data" / "public" / "indoor-sockets"

CLASSES = ["open", "short", "damage"]

# circuit-faults ids: 0 complete, 1 incomplete, 2 wires_touching, 3 damage
CIRCUIT_MAP = {1: 0, 2: 1, 3: 2}

# PCB-IND: 0-5 damage-like, 6 short, 7 open
PCB_MAP = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 0}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _copy_pair(img: Path, lines: list[str], split: str, prefix: str, allow_empty: bool = False) -> None:
    if not lines and not allow_empty:
        return
    img_dir = DEST / split / "images"
    lab_dir = DEST / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{img.stem}"
    shutil.copy2(img, img_dir / f"{stem}{img.suffix.lower()}")
    payload = ("\n".join(lines) + "\n") if lines else ""
    (lab_dir / f"{stem}.txt").write_text(payload, encoding="utf-8")


def _remap_label(txt: Path, mapping: dict[int, int]) -> list[str]:
    if not txt.exists():
        return []
    out = []
    for raw in txt.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        old = int(float(parts[0]))
        if old not in mapping:
            continue
        parts[0] = str(mapping[old])
        out.append(" ".join(parts))
    return out


def add_circuit() -> Counter:
    counts: Counter = Counter()
    if not CIRCUIT.exists():
        print("skip circuit-faults (missing)")
        return counts
    for split_src, split_dst in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        img_dir = CIRCUIT / split_src / "images"
        lab_dir = CIRCUIT / split_src / "labels"
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            lines = _remap_label(lab_dir / f"{img.stem}.txt", CIRCUIT_MAP)
            if not lines:
                continue
            _copy_pair(img, lines, split_dst, "cf")
            counts[split_dst] += 1
    print("circuit-faults (no complete):", dict(counts))
    return counts


def add_pcb_ind() -> Counter:
    counts: Counter = Counter()
    img_root = PCB / "images"
    lab_root = PCB / "labels"
    if not img_root.is_dir():
        # some exports keep images next to split
        for split in ("train", "val", "test"):
            img_dir = PCB / split / "images"
            lab_dir = PCB / split / "labels"
            if not img_dir.is_dir():
                img_dir = PCB / "images" / split
                lab_dir = PCB / "labels" / split
            if not img_dir.is_dir():
                continue
            for img in img_dir.iterdir():
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lines = _remap_label(lab_dir / f"{img.stem}.txt", PCB_MAP)
                if not lines:
                    continue
                _copy_pair(img, lines, split, "ind")
                counts[split] += 1
        print("pcb-ind:", dict(counts))
        return counts
    for split in ("train", "val", "test"):
        img_dir = img_root / split
        lab_dir = lab_root / split
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            lines = _remap_label(lab_dir / f"{img.stem}.txt", PCB_MAP)
            if not lines:
                continue
            _copy_pair(img, lines, split, "ind")
            counts[split] += 1
    print("pcb-ind:", dict(counts))
    return counts


def _is_defect_wire(path: Path) -> bool:
    blob = str(path).lower().replace("\\", "/")
    good = any(k in blob for k in ("good", "ok", "normal", "patchcore"))
    bad = any(k in blob for k in ("cut", "pull", "pulled", "defect", "damage", "strand"))
    if bad and not good:
        return True
    if good and not bad:
        return False
    return bad


def add_stripped() -> Counter:
    counts: Counter = Counter()
    if not STRIPPED.exists():
        print("skip stripped-wire (missing)")
        return counts
    imgs = [p for p in STRIPPED.rglob("*") if p.suffix.lower() in IMG_EXT and _is_defect_wire(p)]
    rng = random.Random(0)
    rng.shuffle(imgs)
    n = len(imgs)
    splits = (
        [("train", imgs[: int(n * 0.8)])]
        + [("val", imgs[int(n * 0.8) : int(n * 0.9)])]
        + [("test", imgs[int(n * 0.9) :])]
    )
    for split, batch in splits:
        for img in batch:
            # close-up of one damaged wire: box the central 90% as damage
            lines = ["2 0.500000 0.500000 0.900000 0.900000"]
            _copy_pair(img, lines, split, "sw")
            counts[split] += 1
    print("stripped-wire defect close-ups:", dict(counts), "of", n)
    return counts


def add_socket_backgrounds(max_train: int = 600, max_val: int = 80, max_test: int = 80) -> Counter:
    """Home interiors with empty labels so YOLO learns normal sockets are not defects."""
    counts: Counter = Counter()
    if not SOCKETS.exists():
        print("skip indoor-sockets backgrounds (missing)")
        return counts
    imgs = [p for p in SOCKETS.rglob("*") if p.suffix.lower() in IMG_EXT]
    rng = random.Random(1)
    rng.shuffle(imgs)
    caps = {"train": max_train, "val": max_val, "test": max_test}
    # Prefer native split folders if the zip used them.
    by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    leftover: list[Path] = []
    for img in imgs:
        blob = str(img).lower().replace("\\", "/")
        if "/train/" in blob or "\\train\\" in blob or "/train\\" in blob:
            by_split["train"].append(img)
        elif "/val/" in blob or "/valid/" in blob or "/validation/" in blob:
            by_split["val"].append(img)
        elif "/test/" in blob:
            by_split["test"].append(img)
        else:
            leftover.append(img)
    i = 0
    for split in ("train", "val", "test"):
        need = caps[split] - len(by_split[split][: caps[split]])
        if need > 0:
            by_split[split].extend(leftover[i : i + need])
            i += need
    for split, cap in caps.items():
        for img in by_split[split][:cap]:
            _copy_pair(img, [], split, "sock", allow_empty=True)
            counts[split] += 1
    print("indoor-socket backgrounds (empty labels):", dict(counts))
    return counts


def write_yaml(n_images: dict) -> None:
    yaml = f"""path: {DEST.as_posix()}
train: train/images
val: val/images
test: test/images
names:
  0: open
  1: short
  2: damage
nc: 3
# home-faults: defect boxes only. Public mix is still not a Pakistani consumer unit.
"""
    (DEST / "data.yaml").write_text(yaml, encoding="utf-8")
    print("images", n_images)
    print("Wrote", DEST / "data.yaml")


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    add_circuit()
    add_pcb_ind()
    add_stripped()
    add_socket_backgrounds()
    n = {}
    for split in ("train", "val", "test"):
        d = DEST / split / "images"
        n[split] = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else 0
    write_yaml(n)
    boxes: Counter = Counter()
    for lab in DEST.rglob("*.txt"):
        if lab.name == "data.yaml":
            continue
        for line in lab.read_text(encoding="utf-8").splitlines():
            if line.strip():
                boxes[int(line.split()[0])] += 1
    print("boxes", {CLASSES[k]: v for k, v in sorted(boxes.items())})


if __name__ == "__main__":
    main()

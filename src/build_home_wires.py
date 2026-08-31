"""Build a YOLO dataset of home-circuit conductors (red / blue / IEC / legacy / earth).

Public labeled photos of Pakistani/UK residential boards almost do not exist.
This generator makes a *bootstrap* set so YOLO can learn insulation colors.
It is not a substitute for photos of real consumer units in data/own/home_circuits.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public" / "home-wires"

# 0-1: the colors the user asked to test. 2-4: real home stock (IEC + legacy). 5-6: visible faults.
CLASSES = [
    "red_live",
    "blue_neutral",
    "brown_live",
    "black_wire",
    "earth",
    "exposed_copper",
    "burnt",
]

# RGB fills that read as PVC insulation under mixed indoor light.
COLORS = {
    "red_live": (196, 28, 28),
    "blue_neutral": (28, 78, 186),
    "brown_live": (118, 62, 28),
    "black_wire": (22, 22, 24),
    "earth_green": (34, 140, 52),
    "earth_yellow": (230, 196, 36),
    "exposed_copper": (198, 122, 48),
    "burnt": (36, 24, 16),
}


def _jitter(rgb: tuple[int, int, int], rng: random.Random, amp: int = 18) -> tuple[int, int, int]:
    return tuple(max(0, min(255, c + rng.randint(-amp, amp))) for c in rgb)


def _panel_background(size: int, rng: random.Random) -> Image.Image:
    """Consumer-unit / backboard look, not a chroma-key lab."""
    kind = rng.choice(["enclosure", "plywood", "concrete", "white_wall"])
    if kind == "enclosure":
        img = Image.new("RGB", (size, size), (18 + rng.randint(0, 12),) * 3)
        d = ImageDraw.Draw(img)
        m = rng.randint(24, 48)
        d.rounded_rectangle(
            [m, m, size - m, size - m],
            radius=12,
            fill=(52 + rng.randint(0, 20), 56 + rng.randint(0, 16), 58 + rng.randint(0, 16)),
            outline=(30, 30, 32),
            width=3,
        )
        # Fake DIN-rail MCBs as unlabeled clutter (model must find wires, not gray blocks).
        rail_y = rng.randint(size // 3, size // 2)
        x = m + 20
        while x < size - m - 40:
            w, h = rng.randint(22, 34), rng.randint(70, 110)
            d.rectangle([x, rail_y, x + w, rail_y + h], fill=(78, 82, 86), outline=(40, 40, 42))
            d.rectangle([x + 4, rail_y + 8, x + w - 4, rail_y + 22], fill=(200, 40, 40) if rng.random() < 0.3 else (240, 240, 242))
            x += w + rng.randint(2, 8)
        return img
    if kind == "plywood":
        base = (150 + rng.randint(-20, 20), 108 + rng.randint(-16, 16), 62 + rng.randint(-12, 12))
        img = Image.new("RGB", (size, size), base)
        d = ImageDraw.Draw(img)
        for y in range(0, size, rng.randint(10, 18)):
            d.line([(0, y), (size, y + rng.randint(-4, 4))], fill=_jitter(base, rng, 12), width=1)
        return img
    if kind == "concrete":
        img = Image.new("RGB", (size, size), (130 + rng.randint(-15, 15),) * 3)
        pix = img.load()
        for _ in range(800):
            x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
            g = 110 + rng.randint(0, 50)
            pix[x, y] = (g, g, g)
        return img.filter(ImageFilter.GaussianBlur(0.6))
    img = Image.new("RGB", (size, size), (228 + rng.randint(-10, 10), 226, 220))
    d = ImageDraw.Draw(img)
    if rng.random() < 0.6:
        d.rectangle([size // 5, 0, size // 5 + 18, size], fill=(90, 90, 92))
    return img


def _polyline(rng: random.Random, size: int, n: int = 5) -> list[tuple[int, int]]:
    x = rng.randint(40, size - 40)
    y = rng.randint(40, size - 40)
    pts = [(x, y)]
    heading = rng.uniform(0, 2 * math.pi)
    for _ in range(n - 1):
        heading += rng.uniform(-0.7, 0.7)
        step = rng.randint(70, 140)
        x = max(20, min(size - 20, int(x + step * math.cos(heading))))
        y = max(20, min(size - 20, int(y + step * math.sin(heading))))
        pts.append((x, y))
    return pts


def _box_from_line(pts: list[tuple[int, int]], width: int, size: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = width // 2 + 2
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(size - 1, max(xs) + pad), min(size - 1, max(ys) + pad)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    xc, yc = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size
    return xc, yc, bw / size, bh / size


def _draw_earth(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], width: int, rng: random.Random) -> None:
    """Green/yellow striped CPC — the earth conductor on home circuits."""
    for i in range(len(pts) - 1):
        color = COLORS["earth_green"] if i % 2 == 0 else COLORS["earth_yellow"]
        draw.line([pts[i], pts[i + 1]], fill=_jitter(color, rng, 10), width=width, joint="curve")
    r = width // 2
    draw.ellipse([pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r], fill=COLORS["earth_green"])
    draw.ellipse([pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r], fill=COLORS["earth_yellow"])


def _draw_solid(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: tuple[int, int, int], width: int, rng: random.Random) -> None:
    c = _jitter(color, rng, 14)
    draw.line(pts, fill=c, width=width, joint="curve")
    r = width // 2
    draw.ellipse([pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r], fill=c)
    draw.ellipse([pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r], fill=c)


def _make_one(size: int, rng: random.Random) -> tuple[Image.Image, list[str]]:
    img = _panel_background(size, rng)
    draw = ImageDraw.Draw(img)
    labels: list[str] = []

    n_wires = rng.randint(3, 6)
    # Bias toward red + blue so the requested home-circuit pair is always present.
    must = ["red_live", "blue_neutral"]
    pool = ["red_live", "blue_neutral", "brown_live", "black_wire", "earth"]
    names = must + [rng.choice(pool) for _ in range(n_wires - 2)]
    rng.shuffle(names)

    for name in names:
        pts = _polyline(rng, size, n=rng.randint(4, 6))
        width = rng.randint(10, 18)
        if name == "earth":
            _draw_earth(draw, pts, width, rng)
        else:
            _draw_solid(draw, pts, COLORS[name], width, rng)
        xc, yc, bw, bh = _box_from_line(pts, width, size)
        labels.append(f"{CLASSES.index(name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if rng.random() < 0.22:
            end = pts[-1]
            r = width
            copper = _jitter(COLORS["exposed_copper"], rng, 12)
            draw.ellipse([end[0] - r, end[1] - r, end[0] + r + 8, end[1] + r], fill=copper)
            xc, yc, bw, bh = end[0] / size, end[1] / size, (2 * r + 8) / size, (2 * r) / size
            labels.append(f"{CLASSES.index('exposed_copper')} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if rng.random() < 0.16:
            mid = pts[len(pts) // 2]
            br = rng.randint(12, 22)
            draw.ellipse([mid[0] - br, mid[1] - br, mid[0] + br, mid[1] + br], fill=_jitter(COLORS["burnt"], rng, 8))
            labels.append(
                f"{CLASSES.index('burnt')} {mid[0] / size:.6f} {mid[1] / size:.6f} {(2 * br) / size:.6f} {(2 * br) / size:.6f}"
            )

    if rng.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.2))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
    return img, labels


def _write_yaml(dest: Path) -> None:
    text = (
        "names:\n"
        + "".join(f"- {n}\n" for n in CLASSES)
        + f"nc: {len(CLASSES)}\n"
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "license: generated in-repo for bootstrap (not a public photo set)\n"
        "note: Synthetic home-circuit conductors. Replace/mix with data/own/home_circuits for real boards.\n"
    )
    (dest / "data.yaml").write_text(text, encoding="utf-8")


def build(dest: Path, n_train: int, n_val: int, n_test: int, imgsz: int, seed: int) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits = {"train": n_train, "valid": n_val, "test": n_test}
    idx = 0
    for split, n in splits.items():
        img_dir = dest / split / "images"
        lab_dir = dest / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            idx += 1
            img, labels = _make_one(imgsz, rng)
            stem = f"home_{idx:04d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            (lab_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    _write_yaml(dest)
    print(f"Wrote {n_train}+{n_val}+{n_test} images -> {dest}")
    print("Classes:", ", ".join(CLASSES))
    return dest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic home-circuit wire YOLO set.")
    p.add_argument("--dest", type=Path, default=DEST)
    p.add_argument("--n-train", type=int, default=480)
    p.add_argument("--n-val", type=int, default=96)
    p.add_argument("--n-test", type=int, default=96)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build(args.dest, args.n_train, args.n_val, args.n_test, args.imgsz, args.seed)


if __name__ == "__main__":
    main()

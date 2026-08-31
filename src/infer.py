"""Run a trained detector and save annotated shots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.device import resolve_device
from src.more_detect import annotate, predict_bgr, predict_more
from src.preprocess import enhance_bgr

ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "cable_damage": {
        "weights": ROOT / "models" / "investor_proof.pt",
        "source": ROOT / "data" / "public" / "cable-damage" / "test" / "images",
        "out": ROOT / "results" / "investor_proof",
        "need": "python -m src.train --preset cable_damage",
    },
    "home_wires": {
        "weights": ROOT / "models" / "home_wires.pt",
        "source": ROOT / "data" / "public" / "home-wires" / "test" / "images",
        "out": ROOT / "results" / "home_wires",
        "need": "python -m src.train --preset home_wires",
    },
    "home_faults": {
        "weights": ROOT / "models" / "home_faults.pt",
        "source": ROOT / "data" / "public" / "home-faults" / "test" / "images",
        "out": ROOT / "results" / "home_faults",
        "need": "python -m src.train --preset home_faults",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), default="cable_damage")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--more", action="store_true", help="CLAHE/sharpen + merge extra boxes (keeps original detections)")
    p.add_argument("--enhance", action="store_true", help="Run YOLO on CLAHE/sharpen image only")
    p.add_argument("--dual", action="store_true", help="Merge original + enhanced boxes")
    p.add_argument("--tiles", action="store_true", help="SAHI-style tiled inference (higher FP; use with --extra-conf)")
    p.add_argument("--tta-flip", action="store_true", help="Also merge horizontal-flip TTA")
    p.add_argument("--extra-conf", type=float, default=0.4, help="Min conf for extra branches")
    p.add_argument("--hide-complete", action="store_true", help="Do not draw the complete class")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    weights = args.weights or preset["weights"]
    source = args.source or preset["source"]
    out = args.out or preset["out"]
    device = resolve_device(args.device)
    imgsz = args.imgsz if args.imgsz is not None else (320 if device == "cpu" else 640)

    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}. Train first: {preset['need']}")
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    images = sorted(
        [p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[: args.max_images]
    if not images:
        raise SystemExit(f"No images in {source}")

    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    use_more = args.more or args.dual or args.tiles or args.tta_flip
    skip_complete = set()
    if args.hide_complete:
        skip_complete = {i for i, n in model.names.items() if str(n) == "complete"}
    summary = []
    for img in images:
        bgr = cv2.imread(str(img))
        if bgr is None:
            continue
        names = {int(k): str(v) for k, v in model.names.items()}
        if use_more:
            variants = predict_more(
                model,
                bgr,
                imgsz=imgsz,
                conf=args.conf,
                device=device,
                recipe="auto",
                dual=args.more or args.dual,
                tiles=args.tiles,
                tta_flip=args.tta_flip,
                tile_size=imgsz,
                extra_conf=args.extra_conf,
            )
            dets = variants["merged"]
            vis = annotate(bgr, dets, skip_cls=skip_complete)
        elif args.enhance:
            enh = enhance_bgr(bgr, recipe="auto")
            dets = predict_bgr(model, enh, imgsz, args.conf, device)
            dets.names = names
            vis = annotate(enh, dets, skip_cls=skip_complete)
        else:
            results = model.predict(
                source=str(img),
                imgsz=imgsz,
                conf=args.conf,
                device=device,
                save=False,
                verbose=False,
            )
            r = results[0]
            out_path = out / f"pred_{img.stem}.jpg"
            r.save(filename=str(out_path))
            dets_list = []
            if r.boxes is not None:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    dets_list.append({"class": names.get(cls_id, str(cls_id)), "conf": round(float(b.conf[0]), 3)})
            summary.append({"image": img.name, "n": len(dets_list), "detections": dets_list, "saved": out_path.name})
            print(f"{img.name}: {len(dets_list)} dets -> {out_path.name}")
            continue

        dets.names = names
        out_path = out / f"pred_{img.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        dets_list = [
            {"class": names.get(int(c), str(int(c))), "conf": round(float(s), 3)}
            for c, s in zip(dets.cls, dets.conf)
            if int(c) not in skip_complete
        ]
        summary.append({"image": img.name, "n": len(dets_list), "detections": dets_list, "saved": out_path.name})
        print(f"{img.name}: {len(dets_list)} dets -> {out_path.name}")

    (out / "predictions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out / "predictions.json")


if __name__ == "__main__":
    main()

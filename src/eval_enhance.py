"""Measure original vs CLAHE/sharpen vs dual-pass vs tiles on the circuit-faults test split."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.boxes import load_yolo_labels, match_counts, merge_keep_original, pr
from src.device import resolve_device
from src.more_detect import annotate, hstack_labeled, predict_bgr, predict_tiles
from src.preprocess import enhance_bgr, stages_bgr

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval image-processing + extra YOLO passes.")
    p.add_argument("--weights", type=Path, default=ROOT / "models" / "circuit_faults.pt")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "public" / "circuit-faults" / "test")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "enhance_proof")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=0, help="0 = all test images")
    p.add_argument("--tile-limit", type=int, default=0, help="How many images also get tiled inference (0 = skip; tiles add many FPs at conf 0.25)")
    p.add_argument("--extra-conf", type=float, default=0.4, help="Min conf for enhanced/tile extras")
    p.add_argument("--device", default="auto")
    p.add_argument("--grids", type=int, default=8)
    return p.parse_args()


def add_counts(acc: dict, part: dict) -> None:
    for k, v in part.items():
        acc[k] = acc.get(k, 0) + v


def summarize(acc: dict) -> dict:
    if not acc or "tp" not in acc:
        return {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    p, r = pr(acc)
    f1 = round(2 * p * r / max(1e-9, p + r), 4)
    return {**acc, "precision": p, "recall": r, "f1": f1}


def label_path_for(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "grids").mkdir(exist_ok=True)
    (out / "stages").mkdir(exist_ok=True)

    images = sorted(
        [p for p in (args.data / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images in {args.data / 'images'}")

    model = YOLO(str(args.weights))
    names = {int(k): str(v) for k, v in model.names.items()}
    complete_id = next((i for i, n in names.items() if n == "complete"), 0)

    modes = ("original", "enhanced", "dual")
    totals = {m: defaultdict(int) for m in modes}
    totals_defect = {m: defaultdict(int) for m in modes}
    tile_totals = {m: defaultdict(int) for m in ("original", "tiles", "dual_tiles")}
    extra_found = []  # dual recovered a GT the original missed

    rng = random.Random(0)
    grid_candidates: list[Path] = []

    for i, img_path in enumerate(images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = load_yolo_labels(label_path_for(img_path), w, h)
        gt.names = names

        orig = predict_bgr(model, bgr, args.imgsz, args.conf, device)
        orig.names = names
        enh_img = enhance_bgr(bgr, recipe="auto")
        enh = predict_bgr(model, enh_img, args.imgsz, args.extra_conf, device)
        enh.names = names
        dual = merge_keep_original(orig, enh, iou_thr=0.5)

        variants = {"original": orig, "enhanced": enh, "dual": dual}
        for m, d in variants.items():
            add_counts(totals[m], match_counts(d, gt, iou_thr=0.5))
            add_counts(totals_defect[m], match_counts(d, gt, iou_thr=0.5, ignore_cls={complete_id}))

        orig_m = match_counts(orig, gt, 0.5, ignore_cls={complete_id})
        dual_m = match_counts(dual, gt, 0.5, ignore_cls={complete_id})
        if dual_m["tp"] > orig_m["tp"]:
            extra_found.append(img_path.name)
            grid_candidates.append(img_path)

        if i < args.tile_limit:
            tiled = predict_tiles(model, bgr, args.imgsz, args.extra_conf, device, tile=args.imgsz)
            tiled.names = names
            dual_t = merge_keep_original(dual, tiled, iou_thr=0.5)
            add_counts(tile_totals["original"], match_counts(orig, gt, 0.5, ignore_cls={complete_id}))
            add_counts(tile_totals["tiles"], match_counts(tiled, gt, 0.5, ignore_cls={complete_id}))
            add_counts(tile_totals["dual_tiles"], match_counts(dual_t, gt, 0.5, ignore_cls={complete_id}))

        if (i + 1) % 25 == 0 or i + 1 == len(images):
            print(f"{i + 1}/{len(images)}  dual defect TP {totals_defect['dual']['tp']} / orig {totals_defect['original']['tp']}")

    # proof grids: prefer images where dual recovered extra GTs
    if len(grid_candidates) < args.grids:
        rest = [p for p in images if p not in grid_candidates]
        rng.shuffle(rest)
        grid_candidates.extend(rest)
    grid_paths = grid_candidates[: args.grids]

    for img_path in grid_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        orig = predict_bgr(model, bgr, args.imgsz, args.conf, device)
        orig.names = names
        enh_img = enhance_bgr(bgr, recipe="auto")
        enh = predict_bgr(model, enh_img, args.imgsz, args.extra_conf, device)
        enh.names = names
        dual = merge_keep_original(orig, enh, iou_thr=0.5)
        skip = {complete_id}
        det_row = hstack_labeled(
            [
                ("original + YOLO", annotate(bgr, orig, skip_cls=skip)),
                ("enhanced + YOLO", annotate(enh_img, enh, skip_cls=skip)),
                ("dual merge", annotate(bgr, dual, skip_cls=skip)),
            ],
            height=360,
        )
        cv2.imwrite(str(out / "grids" / f"dets_{img_path.stem}.jpg"), det_row)
        st = stages_bgr(bgr)
        stage_row = hstack_labeled(
            [(k, st[k]) for k in ("original", "clahe", "sharpen", "tophat", "enhanced")],
            height=280,
        )
        cv2.imwrite(str(out / "stages" / f"stages_{img_path.stem}.jpg"), stage_row)

    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weights": str(args.weights).replace("\\", "/"),
        "n_images": len(images),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "extra_conf": args.extra_conf,
        "iou": 0.5,
        "merge": "keep original boxes, bump conf if enhanced agrees, add non-overlapping extras",
        "all_classes": {m: summarize(dict(totals[m])) for m in modes},
        "defect_only_ignore_complete": {m: summarize(dict(totals_defect[m])) for m in modes},
        "tiles_subset": None if args.tile_limit <= 0 else {
            "n_images": min(args.tile_limit, len(images)),
            "metrics_defect_only": {m: summarize(dict(tile_totals[m])) for m in tile_totals},
        },
        "images_where_dual_recovered_extra_gt": extra_found[:40],
        "n_dual_helped": len(extra_found),
        "not_claimed": [
            "This is not a new trained mAP from Ultralytics val()",
            "PCB/cable test photos are not Pakistani consumer units",
            "No boxes still does not mean electrically complete or safe",
        ],
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", metrics_path)
    print(json.dumps({k: payload[k] for k in ("all_classes", "defect_only_ignore_complete", "n_dual_helped")}, indent=2))


if __name__ == "__main__":
    main()

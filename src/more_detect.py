"""Run YOLO on original + enhanced + tiles + hflip, then NMS-merge.

The trained weights stay the same. Extra passes recover small defects that a
single 320 px letterbox pass misses. Dual-pass is safe: original boxes are kept.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from src.boxes import Dets, hflip_xyxy, merge_keep_original, tile_windows
from src.preprocess import Recipe, enhance_bgr


def dets_from_result(r, names: dict | None = None) -> Dets:
    names = names or {int(k): str(v) for k, v in r.names.items()}
    if r.boxes is None or len(r.boxes) == 0:
        return Dets.empty(names)
    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = r.boxes.conf.cpu().numpy().astype(np.float32)
    cls = r.boxes.cls.cpu().numpy().astype(np.int32)
    return Dets(xyxy, conf, cls, names)


def predict_bgr(model: YOLO, bgr: np.ndarray, imgsz: int, conf: float, device: str) -> Dets:
    results = model.predict(
        source=bgr,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False,
        verbose=False,
    )
    return dets_from_result(results[0])


def predict_tiles(
    model: YOLO,
    bgr: np.ndarray,
    imgsz: int,
    conf: float,
    device: str,
    tile: int = 320,
    overlap: float = 0.25,
) -> Dets:
    h, w = bgr.shape[:2]
    names = None
    acc = Dets.empty()
    for x1, y1, x2, y2 in tile_windows(h, w, tile=tile, overlap=overlap):
        crop = bgr[y1:y2, x1:x2]
        d = predict_bgr(model, crop, imgsz=min(imgsz, max(crop.shape[0], crop.shape[1])), conf=conf, device=device)
        if names is None:
            names = d.names
            acc.names = names
        if len(d) == 0:
            continue
        shifted = d.xyxy.copy()
        shifted[:, [0, 2]] += x1
        shifted[:, [1, 3]] += y1
        acc = acc.concat(Dets(shifted, d.conf, d.cls, d.names))
    return acc


def predict_hflip(model: YOLO, bgr: np.ndarray, imgsz: int, conf: float, device: str) -> Dets:
    flipped = cv2.flip(bgr, 1)
    d = predict_bgr(model, flipped, imgsz, conf, device)
    if len(d) == 0:
        return d
    w = bgr.shape[1]
    return Dets(hflip_xyxy(d.xyxy, w), d.conf, d.cls, d.names)


def predict_more(
    model: YOLO,
    bgr: np.ndarray,
    imgsz: int = 320,
    conf: float = 0.25,
    device: str = "cpu",
    recipe: Recipe = "auto",
    dual: bool = True,
    tiles: bool = False,
    tta_flip: bool = False,
    tile_size: int = 320,
    extra_conf: float | None = 0.4,
    iou_thr: float = 0.5,
) -> dict[str, Dets]:
    """Return named variants plus `merged`. Extra branches default to conf 0.4."""
    extra_c = conf if extra_conf is None else extra_conf
    original = predict_bgr(model, bgr, imgsz, conf, device)
    out: dict[str, Dets] = {"original": original}
    acc = original
    do_enhance = dual

    if do_enhance:
        enhanced_img = enhance_bgr(bgr, recipe=recipe if recipe != "off" else "auto")
        enhanced = predict_bgr(model, enhanced_img, imgsz, extra_c, device)
        out["enhanced"] = enhanced
        acc = merge_keep_original(acc, enhanced, iou_thr)

    if tiles:
        tiled = predict_tiles(model, bgr, imgsz, extra_c, device, tile=tile_size)
        out["tiles"] = tiled
        acc = merge_keep_original(acc, tiled, iou_thr)
        if dual:
            tiled_e = predict_tiles(
                model,
                enhance_bgr(bgr, recipe=recipe if recipe != "off" else "auto"),
                imgsz,
                extra_c,
                device,
                tile=tile_size,
            )
            out["tiles_enhanced"] = tiled_e
            acc = merge_keep_original(acc, tiled_e, iou_thr)

    if tta_flip:
        flipped = predict_hflip(model, bgr, imgsz, extra_c, device)
        out["hflip"] = flipped
        acc = merge_keep_original(acc, flipped, iou_thr)

    out["merged"] = acc
    return out


def annotate(bgr: np.ndarray, dets: Dets, skip_cls: Iterable[int] = ()) -> np.ndarray:
    vis = bgr.copy()
    skip = set(skip_cls)
    ann = Annotator(vis, line_width=2, font_size=12)
    for i in range(len(dets)):
        cid = int(dets.cls[i])
        if cid in skip:
            continue
        name = dets.names.get(cid, str(cid))
        conf = float(dets.conf[i])
        box = dets.xyxy[i].tolist()
        ann.box_label(box, f"{name} {conf:.2f}", color=colors(cid, True))
    return ann.result()


def hstack_labeled(images: list[tuple[str, np.ndarray]], height: int = 320) -> np.ndarray:
    resized = []
    for title, im in images:
        h, w = im.shape[:2]
        scale = height / h
        im2 = cv2.resize(im, (max(1, int(w * scale)), height), interpolation=cv2.INTER_AREA)
        bar = np.zeros((28, im2.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        resized.append(np.vstack([bar, im2]))
    return np.hstack(resized)

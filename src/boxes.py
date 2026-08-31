"""xyxy helpers: NMS, IoU match, tile windows, hflip."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dets:
    xyxy: np.ndarray  # (N, 4) float
    conf: np.ndarray  # (N,)
    cls: np.ndarray  # (N,) int
    names: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls, names: dict[int, str] | None = None) -> "Dets":
        return cls(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int32),
            names=names or {},
        )

    def concat(self, other: "Dets") -> "Dets":
        names = dict(self.names)
        names.update(other.names)
        if len(self) == 0:
            return Dets(other.xyxy.copy(), other.conf.copy(), other.cls.copy(), names)
        if len(other) == 0:
            return Dets(self.xyxy.copy(), self.conf.copy(), self.cls.copy(), names)
        return Dets(
            xyxy=np.concatenate([self.xyxy, other.xyxy], axis=0),
            conf=np.concatenate([self.conf, other.conf], axis=0),
            cls=np.concatenate([self.cls, other.cls], axis=0),
            names=names,
        )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


def merge_keep_original(orig: Dets, extra: Dets, iou_thr: float = 0.5) -> Dets:
    """Keep original box geometry. Raise conf if an extra box agrees. Add non-overlapping extras."""
    if len(extra) == 0:
        return orig
    if len(orig) == 0:
        return nms(extra, iou_thr)
    conf = orig.conf.copy()
    extras_only: list[int] = []
    for i in range(len(extra)):
        same = orig.cls == extra.cls[i]
        if not np.any(same):
            extras_only.append(i)
            continue
        ious = iou_matrix(extra.xyxy[i : i + 1], orig.xyxy[same])[0]
        if float(ious.max()) < iou_thr:
            extras_only.append(i)
            continue
        local = np.where(same)[0]
        j = int(local[int(np.argmax(ious))])
        conf[j] = max(float(conf[j]), float(extra.conf[i]))
    kept = Dets(orig.xyxy.copy(), conf, orig.cls.copy(), orig.names)
    if not extras_only:
        return kept
    idx = np.array(extras_only, dtype=np.int64)
    added = Dets(extra.xyxy[idx], extra.conf[idx], extra.cls[idx], extra.names)
    return kept.concat(nms(added, iou_thr))


def nms(dets: Dets, iou_thr: float = 0.5) -> Dets:
    if len(dets) == 0:
        return dets
    keep_idx: list[int] = []
    for c in np.unique(dets.cls):
        sel = np.where(dets.cls == c)[0]
        boxes = dets.xyxy[sel]
        scores = dets.conf[sel]
        order = scores.argsort()[::-1]
        while order.size:
            i = int(order[0])
            keep_idx.append(int(sel[i]))
            if order.size == 1:
                break
            ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
            order = order[1:][ious < iou_thr]
    keep_idx.sort()
    k = np.array(keep_idx, dtype=np.int64)
    return Dets(dets.xyxy[k], dets.conf[k], dets.cls[k], dets.names)


def hflip_xyxy(xyxy: np.ndarray, width: int) -> np.ndarray:
    out = xyxy.copy()
    x1 = xyxy[:, 0].copy()
    x2 = xyxy[:, 2].copy()
    out[:, 0] = width - x2
    out[:, 2] = width - x1
    return out


def tile_windows(h: int, w: int, tile: int = 320, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Inclusive-exclusive xyxy windows covering the image."""
    tile = min(tile, max(h, w), max(h, w) if min(h, w) < tile else tile)
    if h <= tile and w <= tile:
        return [(0, 0, w, h)]
    stride = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    wins = []
    for y in ys:
        for x in xs:
            x2 = min(w, x + tile)
            y2 = min(h, y + tile)
            x1 = max(0, x2 - tile) if x2 - x < tile and w >= tile else x
            y1 = max(0, y2 - tile) if y2 - y < tile and h >= tile else y
            wins.append((x1, y1, x2, y2))
    # unique
    seen = set()
    out = []
    for win in wins:
        if win not in seen:
            seen.add(win)
            out.append(win)
    return out


def yolo_to_xyxy(line_parts: list[float], w: int, h: int) -> tuple[int, np.ndarray]:
    cls_id, xc, yc, bw, bh = line_parts
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return int(cls_id), np.array([x1, y1, x2, y2], dtype=np.float32)


def load_yolo_labels(path, w: int, h: int) -> Dets:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return Dets.empty()
    cls_ids = []
    boxes = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        nums = [float(x) for x in raw.split()]
        if len(nums) < 5:
            continue
        cid, xyxy = yolo_to_xyxy(nums[:5], w, h)
        cls_ids.append(cid)
        boxes.append(xyxy)
    if not boxes:
        return Dets.empty()
    return Dets(
        xyxy=np.stack(boxes, axis=0),
        conf=np.ones(len(boxes), dtype=np.float32),
        cls=np.array(cls_ids, dtype=np.int32),
    )


def match_counts(pred: Dets, gt: Dets, iou_thr: float = 0.5, ignore_cls: set[int] | None = None) -> dict:
    ignore_cls = ignore_cls or set()
    p_mask = np.array([c not in ignore_cls for c in pred.cls], dtype=bool) if len(pred) else np.zeros((0,), dtype=bool)
    g_mask = np.array([c not in ignore_cls for c in gt.cls], dtype=bool) if len(gt) else np.zeros((0,), dtype=bool)
    p = Dets(pred.xyxy[p_mask], pred.conf[p_mask], pred.cls[p_mask], pred.names) if len(pred) else Dets.empty()
    g = Dets(gt.xyxy[g_mask], gt.conf[g_mask], gt.cls[g_mask], gt.names) if len(gt) else Dets.empty()
    if len(g) == 0:
        return {"tp": 0, "fp": int(len(p)), "fn": 0, "gt": 0, "pred": int(len(p))}
    if len(p) == 0:
        return {"tp": 0, "fp": 0, "fn": int(len(g)), "gt": int(len(g)), "pred": 0}

    ious = iou_matrix(p.xyxy, g.xyxy)
    order = np.argsort(-p.conf)
    used_g = np.zeros(len(g), dtype=bool)
    tp = 0
    for i in order:
        # same class
        cls_ok = g.cls == p.cls[i]
        cand = np.where(cls_ok & (~used_g))[0]
        if cand.size == 0:
            continue
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= iou_thr:
            used_g[j] = True
            tp += 1
    fp = int(len(p) - tp)
    fn = int(len(g) - tp)
    return {"tp": tp, "fp": fp, "fn": fn, "gt": int(len(g)), "pred": int(len(p))}


def pr(counts: dict) -> tuple[float, float]:
    p = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    r = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    return round(p, 4), round(r, 4)

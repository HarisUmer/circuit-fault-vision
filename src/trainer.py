"""YOLO trainer extras for tiny defects: CLAHE/sharpen on the batch + small-object hyps."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

from src.preprocess import enhance_bgr


def small_object_hyps() -> dict:
    """Safer defaults for millimetre opens/shorts. Not for home_wires color ids."""
    return {
        "copy_paste": 0.15,
        "copy_paste_mode": "mixup",
        "mixup": 0.08,
        "mosaic": 1.0,
        "close_mosaic": 4,
        "scale": 0.7,
        "degrees": 10.0,
        "flipud": 0.15,
        "fliplr": 0.5,
        "translate": 0.12,
        "multi_scale": True,
        "erasing": 0.0,
    }


class DefectTrainer(DetectionTrainer):
    """Random mild enhance on train batches so the net sees both raw and processed photos."""

    enhance_p = 0.4

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        model = getattr(self, "model", None)
        if model is None or not model.training:
            return batch
        imgs = batch["img"]
        out = imgs.clone()
        for i in range(imgs.shape[0]):
            if random.random() > self.enhance_p:
                continue
            rgb = (imgs[i].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            enh = enhance_bgr(bgr, recipe="auto")
            rgb2 = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
            out[i] = torch.from_numpy(rgb2).permute(2, 0, 1).float() / 255.0
        batch["img"] = out.to(imgs.device, non_blocking=True)
        return batch

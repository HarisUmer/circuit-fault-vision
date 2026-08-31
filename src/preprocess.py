"""Classical image processing so tiny opens, shorts, and burns are easier to see.

Used at inference (deterministic) and, with probability, as a train-time extra.
Recipes stay mild on purpose: aggressive sharpening invents false mouse-bites.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

Recipe = Literal["auto", "photo", "pcb", "clahe", "sharpen", "tophat", "off"]


def is_low_chroma(bgr: np.ndarray, sat_mean: float = 28.0) -> bool:
    """True for near-binary CCD traces (DeepPCB) and gray AOI patches."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return True
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < sat_mean


def auto_gamma(bgr: np.ndarray, lo: float = 0.32, hi: float = 0.72) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    mean = float(gray.mean()) / 255.0
    if lo < mean < hi:
        return bgr
    gamma = float(np.clip(np.log(0.5) / np.log(max(mean, 1e-4)), 0.65, 1.55))
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)
    return cv2.LUT(bgr, table)


def clahe_bgr(bgr: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    if bgr.ndim == 2 or bgr.shape[2] == 1:
        g = bgr if bgr.ndim == 2 else bgr[:, :, 0]
        out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def unsharp(bgr: np.ndarray, sigma: float = 1.15, amount: float = 0.75) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def light_denoise(bgr: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise. Stronger filters would wipe millimetre opens."""
    return cv2.bilateralFilter(bgr, d=5, sigmaColor=35, sigmaSpace=35)


def morph_boost(bgr: np.ndarray, k: int = 9, mix: float = 0.28) -> np.ndarray:
    """White tophat + blackhat: thin traces and dark gaps on PCB copper."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    odd = k if k % 2 else k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (odd, odd))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    boost = cv2.add(th, bh)
    boost_c = cv2.cvtColor(boost, cv2.COLOR_GRAY2BGR) if bgr.ndim == 3 else boost
    mixed = cv2.addWeighted(bgr, 1.0, boost_c, mix, 0)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def edge_residual(bgr: np.ndarray, mix: float = 0.18) -> np.ndarray:
    """Add a little Laplacian energy so hairline cracks occupy more pixels."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mag = cv2.convertScaleAbs(lap)
    mag_c = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR) if bgr.ndim == 3 else mag
    mixed = cv2.addWeighted(bgr, 1.0, mag_c, mix, 0)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def enhance_bgr(bgr: np.ndarray, recipe: Recipe = "auto") -> np.ndarray:
    """Return a BGR uint8 image. `off` is a copy of the input."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    img = bgr
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if recipe == "off":
        return img.copy()

    chosen: Recipe = recipe
    if recipe == "auto":
        chosen = "pcb" if is_low_chroma(img) else "photo"

    if chosen == "clahe":
        return clahe_bgr(img)
    if chosen == "sharpen":
        return unsharp(img)
    if chosen == "tophat":
        return morph_boost(img)

    if chosen == "pcb":
        # Binary-ish traces: CLAHE can posterize. Emphasize gaps instead.
        x = unsharp(img, sigma=1.0, amount=0.65)
        x = morph_boost(x, k=7, mix=0.32)
        return edge_residual(x, mix=0.12)

    # photo: color PCB, cables, panels
    x = light_denoise(img)
    x = auto_gamma(x)
    x = clahe_bgr(x, clip=2.2, tiles=8)
    x = unsharp(x, sigma=1.2, amount=0.7)
    return x


def stages_bgr(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Named intermediates for proof grids."""
    photo = not is_low_chroma(bgr)
    return {
        "original": bgr.copy(),
        "clahe": clahe_bgr(bgr),
        "sharpen": unsharp(bgr),
        "tophat": morph_boost(bgr),
        "enhanced": enhance_bgr(bgr, "photo" if photo else "pcb"),
    }

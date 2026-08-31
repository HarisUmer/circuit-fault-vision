# Image processing and extra detectors — what we built and what it measured

**Project:** `circuit_fault_vision`  
**Date:** 2026-08-28  
**Weights tested:** `models/circuit_faults.pt` (YOLOv8n, 320 px, not retrained this pass)  
**Test set:** 484 circuit-faults images, 1,769 boxes (1,706 if `complete` is ignored)  
**Proof shots:** `results/enhance_proof/`

---

## 1. Ask

Sharpen and otherwise process images so YOLO can see more of the defect, and add anything else that would make the detector better. Then measure it.

Product lock is unchanged: **box the problem**, do not classify complete vs incomplete, never print “safe.”

---

## 2. What we implemented

### 2.1 Classical processing (`src/preprocess.py`)

Mild on purpose. Hard unsharp / heavy denoise invents mouse-bites and wipes millimetre opens.

| Step | What it does | When |
|------|----------------|------|
| Light bilateral denoise | Drops JPEG noise, keeps edges | Color photos (cables, PKU) |
| Auto-gamma | Lifts dark panels, tames blown highlights | Photos whose mean luminance is outside 0.32–0.72 |
| CLAHE on L in LAB | Local contrast so faint burns and nicks occupy more pixels | Photos |
| Unsharp mask (σ≈1.2, amount≈0.7) | Sharpens trace/cable edges | Both |
| Morphological tophat + blackhat | Thin copper and dark gaps | Near-binary PCB (DeepPCB) |
| Laplacian residual (mix 0.12) | Hairline cracks get a little extra energy | PCB recipe |

`recipe="auto"` picks **photo** vs **pcb** from chroma. DeepPCB is almost gray; CLAHE there posterizes, so the PCB recipe skips it.

### 2.2 Dual-pass detect (`src/more_detect.py`, `src/boxes.py`)

Replace-the-image is the wrong move on a model trained on raw photos (see §3). The working pattern is:

1. Run YOLO on the **original** (conf 0.25).
2. Run YOLO on the **enhanced** copy (conf 0.40 — extras must be more sure).
3. **Keep original boxes.** If an enhanced box overlaps the same class, **raise confidence**. If it does not overlap, **add** it.

That cannot delete a box the raw image already found. Naive NMS (highest-conf wins) can: a shifted enhanced box with higher score can knock out a correct original box.

Tiled inference (SAHI-style 320 px windows) is implemented but **not** in default `--more`. At conf 0.25 it flooded the test set with false `damage` boxes (precision 0.12 on a 60-image subset). Opt-in: `python -m src.infer --preset circuit_faults --tiles --extra-conf 0.5`.

Horizontal-flip TTA is the same: opt-in `--tta-flip`.

### 2.3 Training extras (wired, not yet retrained)

`--preset circuit_faults` now, unless you pass `--no-enhance-aug`:

| Extra | Why |
|-------|-----|
| `DefectTrainer` | 40% of train batches get the same enhance recipe, so the next model is not surprised at infer |
| `copy_paste=0.15` + mixup | Rare tiny opens get pasted onto other boards |
| `scale=0.7`, `multi_scale=True` | More zoom / size jitter for millimetre defects |
| `degrees=10`, `flipud=0.15` | PCBs are not gravity-locked |
| `close_mosaic=4` | Last epochs train on full frames |
| `erasing=0.0` | Random erase would hide the defect |
| **albumentations 2.0.8** | Installed so Ultralytics can apply its built-in CLAHE/Blur (p=0.01) |

This box is CPU-only. A full 15-epoch retrain is ~2.5 h; we did **not** overwrite `circuit_faults.pt`. Next GPU train:

```
python -m src.train --preset circuit_faults --device 0 --epochs 40 --imgsz 640 --batch 16
```

---

## 3. Measured results (IoU 0.5 vs YOLO labels)

This is **not** Ultralytics `val()` mAP. Same weights, same 484 test images, greedy class-aware matching.

### 3.1 Defect boxes only (`complete` ignored) — this is the product metric

| Mode | TP | FP | FN | Precision | Recall | F1 |
|------|----|----|----|-----------|--------|-----|
| Original YOLO | 1254 | 296 | 452 | **0.809** | 0.735 | **0.770** |
| Enhanced image only (conf 0.40) | 1144 | 194 | 562 | 0.855 | 0.671 | 0.752 |
| Dual merge (keep original + extras @ 0.40) | **1257** | 342 | 449 | 0.786 | **0.737** | 0.761 |

- Dual recovered **3 extra true defects** (on 3 DeepPCB images) and added **46 extra false positives**.
- Enhanced-**only** is stricter and cleaner (P 0.855) but **misses 110 defects** the raw image already had. Domain shift: the net never saw CLAHE/tophat in training.
- Do **not** replace the camera frame with the processed one. Dual is the infer recipe.

### 3.2 All four classes (including `complete`)

Same pattern: original F1 0.777, dual F1 0.767, dual TP 1314 vs 1311. `complete` is not a product class; ignore that column for claims.

### 3.3 Naive NMS (what we threw away)

Merging original+enhanced at conf 0.25 with ordinary NMS: defect TP **1261** (+7) but FP **416** (+120). It also **lost** original TPs on some cable/PCB images because a shifted enhanced box won NMS. Archived: `results/enhance_proof/metrics_naive_nms.json`.

### 3.4 Tiles (60-image subset, conf 0.25)

Tiles alone: P **0.12**, 288 FP vs 40 TP. Too many repeated texture boxes. Not default.

### 3.5 What the pictures show anyway

Processing is still doing useful work on individual frames:

- Outdoor cable: damage conf **0.59 → 0.70** after CLAHE+sharpen (strands pop).
- DeepPCB: extra `incomplete` / `wires_touching` boxes appear after tophat; dual keeps the original high-conf short **and** the new open.

Stage strips (original / CLAHE / sharpen / tophat / full recipe) are in `results/enhance_proof/stages/`. Side-by-side detections: `results/enhance_proof/grids/`.

---

## 4. How to run it

```
python -m src.eval_enhance
python -m src.infer --preset circuit_faults --more --hide-complete
```

`--more` = enhance + dual merge. It does **not** turn tiles on.

`--enhance` alone = YOLO on the processed image only (worse recall on this checkpoint).

---

## 5. What actually makes the next model better

Infer-time sharpening on a raw-trained net is a **small** recall bump with a precision tax. The list that will move mAP:

1. **Retrain with `DefectTrainer`** so CLAHE/sharpen is in-distribution.
2. **640 px** (tiny opens vanish at 320).
3. **Drop `complete`** and merge PCB-IND (see `DEFECT_BOX_REPORT.md`).
4. Tiles only after that retrain, with `--extra-conf 0.5`.
5. Own labeled defect photos of real DBs — public PCB ≠ Pakistani consumer unit.

---

## 6. Safety and claims

- Screening aid. Live panels: electrician only.
- Extra boxes are still RGB guesses, not continuity tests.
- These F1 numbers are not field accuracy and not Ultralytics mAP50 0.831.
- Do not quote paper mAP as this product.

---

## 7. Files

| Path | Role |
|------|------|
| `src/preprocess.py` | CLAHE, unsharp, tophat, gamma |
| `src/more_detect.py` | Dual / tiles / hflip |
| `src/boxes.py` | IoU, NMS, keep-original merge |
| `src/trainer.py` | DefectTrainer + small-object hyps |
| `src/eval_enhance.py` | Test-set experiment |
| `src/train.py` | Uses DefectTrainer on `circuit_faults` |
| `src/infer.py` | `--more --hide-complete` |
| `results/enhance_proof/metrics.json` | Final dual-merge numbers |
| `results/enhance_proof/metrics_naive_nms.json` | Rejected merge policy |

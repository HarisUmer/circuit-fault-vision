# Real circuit / wire faults (WP1c)

**Product lock (2026-08-27):** the next train drops `complete`. Product is **defect boxes only** — see [`DEFECT_BOX_REPORT.md`](DEFECT_BOX_REPORT.md). This file is the **historical 4-class bootstrap**. Empty detections = no visible problem in this photo, not “circuit complete.”

**Ask (original):** not synthetic drawings. Detect **real** circuit and wire photos with:

| Class | Meaning |
|-------|---------|
| `complete` | Circuit looks intact in this region |
| `incomplete` | Open — gap / broken conductor, circuit not complete |
| `wires_touching` | Short — two conductors touching |
| `damage` | Bite, spur, missing hole, spurious copper, broken/burned cable |

## What exists in public data

There is still **no** labeled residential T&E / Pakistani consumer-unit set with those four names. Closest **photographs**:

| Source | What it is | Mapped to |
|--------|------------|-----------|
| PKU-Market-PCB (HF `RobotHuman/PCB_defect`) | 693 **color photos** of real PCB boards, VOC boxes | `incomplete` ← open_circuit; `wires_touching` ← short; `damage` ← mouse_bite, spur, missing_hole, spurious_copper; `complete` ← defect-free crops of the same boards |
| DeepPCB (HF `thangkt/PCB-Prune-YOLO-DeepPCB`) | 1,500 **real CCD** 640×640 traces (binarized) | `incomplete` ← open; `wires_touching` ← short; `damage` ← mousebite/spur/copper/pin-hole |
| RF100 cable-damage | 1,318 **real** outdoor cable photos | `damage` ← break, thunderbolt |

PCB copper traces are **not** home red/blue PVC. Outdoor cables are **not** a DB. This mix is the public bootstrap for the four labels the user asked for. Mix `data/own/` before claiming field accuracy.

**Merged set (this machine, 2026-08-27):** 4,041 images (train 3,032 / valid 525 / test 484).

| Class | Boxes |
|-------|-------|
| complete | 531 (defect-free crops of PKU boards) |
| incomplete | 2,424 |
| wires_touching | 1,997 |
| damage | 10,057 |

Sources: 693 PKU + 531 complete crops + 1,500 DeepPCB + 1,317 cable-damage.

PKU defects are often digitally added onto real board photos (Huang & Wei, arXiv:1901.08204). DeepPCB is real scans; some defects were also manually added. Cable-damage is field photos.

## Commands

```
python -m src.download_circuit_faults
python -m src.train --preset circuit_faults --device auto
python -m src.infer --preset circuit_faults
```

This box is CPU. On NVIDIA:

```
python -m src.train --preset circuit_faults --device 0 --epochs 40 --imgsz 640 --batch 16
```

Weights: `models/circuit_faults.pt`. Shots: `results/circuit_faults/`.

## Held-out test (this CPU run)

15 epochs, YOLOv8n, 320 px, **2.45 hours** on Intel i7-8550U:

| | |
|--|--|
| mAP50 | **0.831** |
| mAP50-95 | 0.605 |
| Precision / recall | 0.929 / 0.768 |
| complete mAP50-95 | 0.995 (easy full-crop box — do not over-read) |
| incomplete (open) | 0.481 |
| wires_touching (short) | 0.448 |
| damage | 0.496 |

Do **not** quote this mAP as home-wiring accuracy. Do **not** print “safe.”

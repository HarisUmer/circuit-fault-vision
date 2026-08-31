# Home-circuit wires (WP1b) — synthetic only

**Superseded for “real wires”:** photographs live in [`CIRCUIT_FAULTS.md`](CIRCUIT_FAULTS.md) (4-class bootstrap). **Product** is defect boxes only: [`DEFECT_BOX_REPORT.md`](DEFECT_BOX_REPORT.md).

**Ask (original):** detect **home** wiring, especially **red and blue** conductors, not only outdoor metal cables.

**Finding:** there is no freely downloadable labeled set of residential T&E / consumer-unit wires. Roboflow color-wire sets are tiny and gated; the 13 GB HF panel dump is auto-labeled components, not red/blue insulation.

**What we built:** a synthetic YOLO set plus GPU-ready train/infer. Mix real board photos from `data/own/` before claiming field accuracy.

## Color language (say this to investors)

| Class | Typical meaning |
|-------|-----------------|
| `red_live` | Legacy live (old UK / Pakistan / Australia) |
| `blue_neutral` | IEC neutral — the blue the user asked for |
| `brown_live` | IEC live (new installs) |
| `black_wire` | Legacy neutral or US hot |
| `earth` | Green/yellow CPC |
| `exposed_copper` / `burnt` | Visible damage |

Color is **not** a safety proof. Wrong-colored or reused conductors exist. Never print “this is the live” without a tester.

## Held-out test (this CPU run)

12 epochs, YOLOv8n, 320 px, **synthetic** test split (96 images):

| | |
|--|--|
| mAP50 | **0.76** |
| mAP50-95 | 0.57 |
| Precision / recall | 0.70 / 0.69 |
| red_live mAP50-95 | 0.73 |
| blue_neutral mAP50-95 | 0.69 |
| brown_live mAP50-95 | 0.21 (weak — mix real photos) |

Weights: `models/home_wires.pt`. Shots: `results/home_wires/`. Do **not** quote this as accuracy on a real consumer unit.

## Commands

```
python -m src.build_home_wires
python -m src.train --preset home_wires --device auto
python -m src.infer --preset home_wires
```

On a machine with NVIDIA:

```
python -m src.train --preset home_wires --device 0 --epochs 50 --imgsz 640 --batch 16
```

Weights: `models/home_wires.pt`. Shots: `results/home_wires/`.

The outdoor cable-damage model stays at `--preset cable_damage`. It will **not** reliably find red/blue PVC — different domain.

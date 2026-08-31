# Home-adjacent faults (WP1d) — search, fetch, defect-only train

**Date:** 2026-08-30  
**Ask:** more **home** data, search thoroughly, then train properly.

## Search result (public images)

There is still **no freely downloadable labeled set of Pakistani / UK-style consumer units with defect boxes**. What exists:

| Set | Home-like? | Downloadable here? | Why |
|-----|------------|--------------------|-----|
| Roboflow **electrical-hazards / HazardDetector** (~6.1–6.5k) | **Yes** — burned socket, damage wire, open copper, overloaded socket | **No** without `ROBOFLOW_API_KEY` | Best match. Public Domain listing. Export needs a free Roboflow account. |
| Indoor sockets / switches (Zenodo 18835199) | **Yes** — 3,459 wall sockets, switches, power strips | **Yes, ~180 MB** | Locator, not defects. Used as **empty-label backgrounds**. |
| Dutch consumer-unit YOLO (~4.1k, Automaat/RCD/WCD) | Locator (breakers/sockets) | No (51CTO / gated) | Useful later as component locator |
| EIFCD (MDPI Electronics 2026) | Burnt-in / burnt-out multi-sockets (~3.3k) | Paper only; no public zip | Korean fire-forensics set |
| WireWise (Zenodo 20108972) | Paper claims home wiring YOLO | **No images** — 18 kB .docx preprint | Do not quote 87.3% as ours |
| EnergAI fuses (Zenodo 7613424) | LV fuse photos | **3.4 GB** zip — skipped this pass | Locator, not defect boxes |
| `sriom1/electrical-panels-dataset` | Panel photos | **13 GB auto-labels** — skipped | Not trustworthy defect boxes |
| Mendeley aviation wiring (g6rbmc2ggc) | Damaged cables / unplugged | 403 / empty zip | Gated |
| Kaggle engrkarmat wire health | 60 healthy vs frayed wires | Needs Kaggle login; tiny | Segmentation, not YOLO |
| **Stripped Wire** (Zenodo 16686806) | Insulated-wire close-ups, cut/pulled strands | **Yes, 18 MB** | 167 defect photos → `damage` |
| PCB-IND (already on disk) | Factory AOI, not homes | Yes | Real open/short |
| PKU + DeepPCB + RF100 cable | PCB / outdoor cable | Already local | Drop `complete` |

Time-series “home electrical” sets (BLUED, SafeLeak-RCD, tracking CSV) are **not images**. Skipped.  
MVTec cable is CC BY-NC-SA — skip for anything that might be commercial.  
CableInspect-AD is **12 GB overhead HV**, not homes.

## What we built

`data/public/home-faults` — product classes only (`open` / `short` / `damage`). No `complete`.

| Source | Role |
|--------|------|
| circuit-faults minus `complete` | Real PCB + outdoor cable defects |
| PCB-IND | Extra real AOI open / short / damage |
| stripped-wire cut/pulled | Lab wire-end `damage` |
| indoor-sockets (subsample) | **Empty labels** — normal home sockets are not defects |

Images: **train 7,084 / val 1,028 / test 987**. Boxes: open 2,556 · short 2,357 · damage 15,664. Plus 760 home-socket backgrounds.

```
python -m src.download_home_data
python -m src.build_home_faults
python -m src.train --preset home_faults --epochs 15 --device cpu
python -m src.infer --preset home_faults --hide-complete
```

`--preset home_faults` **fine-tunes** `models/circuit_faults.pt` (not COCO from scratch) with DefectTrainer + small-object hyps. CPU, 320 px, YOLOv8n, 15 epochs (~5.6 h).

**Test (987 images):** mAP50 **0.713** · mAP50-95 **0.409** · P **0.810** · R **0.681**.

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| open | 0.747 | 0.435 |
| short | 0.629 | 0.328 |
| damage | 0.764 | 0.464 |

Headline 0.713 is **lower** than circuit-faults 0.831 because `complete` (easy full-crop boxes) was dropped. Weights: `models/home_faults.pt`. Proof: `results/home_faults/`. Gallery: `results/presentable/index.html`.

## Honest limit

This mix is still **PCB AOI + outdoor cable + lab wire ends**, plus unlabeled indoor sockets. It is **closer** to the product ontology and to home appearance, but it is **not** a burned-socket / Pakistani DB model until HazardDetector (or `data/own/`) is in the mix.

If you add a Roboflow API key, re-run download and rebuild to include burned sockets.

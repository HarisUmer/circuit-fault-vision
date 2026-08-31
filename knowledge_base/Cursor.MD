# Cursor.MD — circuit_fault_vision

> Shared memory for multi-agent work on this repo. **Read before every task. Update after any meaningful change.**

---

## Identity

| Field | Value |
|-------|--------|
| Repo | `F:\pange\circuit_fault_vision` |
| Parent | `F:\pange` (PanGe; sibling of `pange_dl`, ComfyUI, `automatic_email`) |
| Branch | `main` |
| Phase | WP1d home-faults **trained**. Test mAP50 **0.713** (defect-only; no complete). Product = **defect boxes only**. |
| Python | `D:\python\python.exe` 3.10.9, torch 2.2.2+cpu, **no NVIDIA GPU**, ultralytics 8.3.5, albumentations 2.0.8 |
| Presentable results | `results/presentable/index.html` — open in a browser |
| Date created | 2026-08-15 |

## Goal

Screen accessible electrical / PCB / cable faults from RGB (and later radiometric IR). Not an in-wall scanner. Not a licensed inspection.

## Locked decisions

| Topic | Decision |
|-------|----------|
| Product class | **Box the problem.** No `complete` / “circuit OK.” Empty boxes = nothing visible in this photo. |
| Safety | Screening aid. Never print “safe.” Live panels = electrician. |
| RGB + IR | RGB = visible damage. Heatmap = **radiometric IR + ΔT** (NETA/NFPA), not Ultralytics movement heatmaps. |
| Detector | Custom YOLO. COCO YOLO will not find these faults. |
| ByteTrack | Video IDs only. |
| Infer processing | Dual-pass: original + CLAHE/sharpen/tophat. **Keep original boxes.** Do not replace the photo. Tiles not default. |
| Public data | ≠ Pakistani residential DBs. PCB traces ≠ home T&E. |
| Papers | Do not quote Tao 98.7% or Chen 99% (sim) as this model’s score. |
| Synthetic wires | `--preset home_wires` is drawings. **Not** the real-wire demo. |
| Git | Personal GitHub **HarisUmer**. User asked 2026-08-31 to publish code + presentable + weights. |

## Current findings (2026-08-30)

### Models on disk

| Preset | Weights | Test mAP50 | mAP50-95 | P / R | Limit |
|--------|---------|------------|----------|-------|--------|
| `cable_damage` | `models/investor_proof.pt` | **0.8876** | 0.423 | 0.84 / 0.91 | Outdoor metal cables |
| `circuit_faults` | `models/circuit_faults.pt` | **0.8307** | 0.605 | 0.93 / 0.77 | PKU + DeepPCB + cable. 15 ep, 320 px, 2.45 h CPU |
| `home_faults` | `models/home_faults.pt` | **0.7134** | 0.409 | 0.81 / 0.68 | Defect-only. Fine-tune, 15 ep, ~5.6 h CPU. Short weakest (mAP50 0.629). Not a Pakistani DB. |

Circuit-faults **per-class mAP50-95:** complete **0.995** (easy full-board crops — drop this class) · incomplete/open **0.481** · wires_touching/short **0.448** · damage **0.496**. Splits 3032 / 525 / 484.

### Dual-pass processing (same `circuit_faults.pt`, 484 test images, IoU 0.5)

Defect-only (ignore `complete`):

| Mode | TP | FP | Precision | Recall | F1 |
|------|----|----|-----------|--------|-----|
| Original | 1254 | 296 | **0.809** | 0.735 | **0.770** |
| Enhanced only (conf 0.40) | 1144 | 194 | 0.855 | 0.671 | 0.752 |
| Dual merge | **1257** | 342 | 0.786 | **0.737** | 0.761 |

+3 true defects, +46 FPs. Enhance-only **hurts recall** (domain shift). Naive NMS merge was worse (can delete a correct original box). Tiles @ 0.25: precision **0.12** — not default.

Visual: cable damage conf 0.59→0.70 after sharpen. DeepPCB dual recovered extras on 3 images.

### Ontology (product)

Chip/PCB: `open`, `short`, `trace_damage` (+ later solder head). Panel: `burn`, `exposed_conductor`, `cable_damage`, IR `hotspot` on a **named lug**. Component locator (`breaker`, `lug`, `busbar`) is not a health class.

### Two-scale extras besides YOLO

PCB: golden-board difference, 640 px, continuity/TDR/ICT. Panel: radiometric ΔT + clamp. PD (TEV/ultrasonic) only for MV cubicles.

### Next (when user says go)

1. If `ROBOFLOW_API_KEY` is set, export HazardDetector (~6k burned sockets) and rebuild.
2. Own labeled defect boxes on real DBs (`data/own/`).
3. 640 px / GPU when available.
4. Radiometric IR + ΔT.

Commands:

```
python -m src.train --preset home_faults --epochs 15 --device cpu
python -m src.infer --preset home_faults --hide-complete
python -m src.infer --preset circuit_faults --more --hide-complete
python -m src.eval_enhance
```

## Where things live

| Path | What |
|------|------|
| `results/presentable/index.html` | **Show this.** Gallery + numbers |
| `knowledge_base/FINDINGS.md` | Snapshot of the same findings |
| `results/presentable/FINDINGS.md` | Same facts, markdown |
| `research/ENHANCE_REPORT.md` | Processing experiment |
| `research/DEFECT_BOX_REPORT.md` | Ontology lock |
| `research/TWO_SCALE_PLAN.md` | Chip vs switch sensors |
| `research/HOME_FAULTS.md` | Home dataset search + defect-only mix |
| `E:\Work\research` | Client pack. Send `PROPOSAL_DETAILED.html` |

## Client pack

`E:\Work\research` — **`PROPOSAL_DETAILED.html`** (Print to PDF). Short: `CLIENT_PROPOSAL.html`. Papers: `papers.md`. Presentable copy: `E:\Work\research\presentable\` (after sync).

## Agent log

- 2026-08-15: Repo + feasibility, datasets, architecture, capture protocol.
- 2026-08-16: Client pack + papers + SoW HTML.
- 2026-08-27: Cable-damage YOLO mAP50 0.888. Synthetic home-wires (later rejected as real-wire demo). Circuit-faults train mAP50 0.831. Two-scale plan. Ontology: defect boxes only.
- 2026-08-28: CLAHE/unsharp/tophat + dual merge. albumentations 2.0.8. DefectTrainer wired, **not** retrained. Presentable gallery: `results/presentable/`.
- 2026-08-31: Published https://github.com/HarisUmer/circuit-fault-vision (public). README + findings now embed proof stills.

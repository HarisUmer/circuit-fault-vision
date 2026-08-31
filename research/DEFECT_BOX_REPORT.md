# Defect-box report: mark the problem, not “complete / incomplete”

**Project:** `circuit_fault_vision`  
**Date:** 2026-08-27  
**Status:** product ontology lock + implementation plan  
**Audience:** engineering / investor technical note

---

## 1. Decision

On **chips, switch boards, and panels**, we will **not** classify whether the circuit is complete.

The model’s job is: **draw a box (or mask) on the problem** — damage, open trace, short, burn, poor solder, hot lug — and leave the rest of the photo unlabeled.

| Do | Do not |
|----|--------|
| Box the defect | Box the whole healthy board as `complete` |
| Empty labels = *no visible problem in this photo* | Print “circuit complete” or “safe” |
| Name the *kind* of problem | Ask YOLO to certify electrical continuity |

The earlier 4-class set (`complete` / `incomplete` / `wires_touching` / `damage`) was a bootstrap. **`complete` was the wrong product class.** It scored mAP50-95 **0.995** on test because those labels were huge easy crops, not because the model understands a healthy circuit. That number must not be sold as “we can tell if a board is OK.”

**Locked (2026-08-27):** defect-only detection. Completeness is an electrician’s test (continuity, TDR, ICT, insulation resistance), not a YOLO class.

---

## 2. Why this is more accurate

1. **Object detection is built for rares.** Faults occupy a few percent of pixels. Training a `complete` class teaches the network to paint the background. That fights localization.
2. **“Incomplete” is ambiguous.** An open on a PCB trace, a tripped breaker, and an in-wall cable break are three different objects. One class smears them.
3. **False “all clear” is the dangerous error.** If we never predict `complete`, we cannot falsely certify a board. No boxes → “nothing visible here,” which is honest.
4. **Factory AOI already works this way.** Inspectors mark defects. A clean board is the absence of marks, plus electrical test.

---

## 3. What to box (product classes)

Two **heads**, not one mixed soup. A small **scene router** first: `pcb_chip` vs `panel_switch` vs `cable` (can be a classifier or a coarse YOLO). Then the matching defect head.

### 3.1 Chip / small PCB (AOI)

Tight boxes on the defect, not the whole IC.

| Class | Box this | Typical size |
|-------|----------|--------------|
| `open` | Gap / broken copper | millimetres |
| `short` | Two conductors touching / solder bridge | millimetres |
| `trace_damage` | Mouse-bite, spur, scratch, missing copper, stain | millimetres |
| `solder_defect` | Insufficient / excess solder, spike | pin-scale |
| `part_misaligned` | SMT shifted off pads | component-scale |

**Not boxed:** healthy copper, healthy chips.  
**Not claimed from RGB:** inner-layer shorts, BGA voids (need X-ray / ICT).

Public data we can train on: PKU, DeepPCB, **PCB-IND** (4,789 real AOI patches, on disk), later SolDef_AI for solder.

### 3.2 Switch board / panel / MCCB

Tight boxes on the **visible** problem. Heat is a second stream.

| Class | Box this | Notes |
|-------|----------|--------|
| `burn` | Scorch, carbon, melted plastic | RGB |
| `exposed_conductor` | Bare copper / stripped insulation | RGB |
| `cable_damage` | Cut, chew, broken strand | RGB |
| `missing_cover` | Open knock-out, missing dead-front | RGB, optional |
| `hotspot` | Abnormal heat on a lug/pole | **IR only**, after a component box exists |

**Not boxed:** “this breaker is on,” “this circuit is complete.”  
**Component boxes** (separate locator head): `breaker`, `lug`, `busbar`, `cable_tail`. Those are *where to measure ΔT*, not defects.

---

## 4. Making the boxes acute (YOLO plus extras)

YOLO alone at 320 px is blunt on millimetre opens (our `incomplete` / `wires_touching` mAP50-95 was **0.48 / 0.45**). Accuracy comes from **how we look**, not from adding a `complete` class.

### 4.1 Inside the vision stack

| Add-on | Why it sharpens the box |
|--------|-------------------------|
| **Train/infer at 640–1280 px** | Tiny opens vanish at 320 px |
| **SAHI / tiled inference** | Slice the board, detect, stitch; standard for small defects |
| **YOLO-seg, not only detect** | Burns and mouse-bites are irregular; mask = better “where” |
| **Cascade (two-stage)** | Stage A: find the board/panel. Stage B: defects **only inside** that ROI. Cuts wall/background false positives (same idea as Liu IEEJ 2025 component→hotspot) |
| **Golden-board difference (PCBs)** | Align a known-good photo; subtract; YOLO runs on the residual. Factory AOI method; better than class YOLO for hairline opens |
| **RGB + IR box fusion (panels)** | RGB box = visible damage. IR box = hot lug. Overlay after alignment (MSX or homography). Same object ID |
| **ByteTrack (video only)** | Same burn/lug keeps one ID while you pan; no extra classes |

### 4.2 Beside YOLO (not instead of)

These do not replace the box. They **confirm or place** it when RGB is blind.

| Add-on | Scale | Puts the problem where? |
|--------|-------|-------------------------|
| Radiometric ΔT vs sibling poles | Panel | On the **hottest lug** of a named breaker |
| Clamp / load note | Panel | Stops “cold because unloaded” false all-clear |
| Continuity / TDR | Chip or outgoing cable | Along the **net** or metres down the cable |
| ICT / flying probe | Chip (factory) | Which pin failed ohms |
| TEV / ultrasonic PD | MV cubicle only | Inside the metal tank |

UX: one finding = **photo with box + evidence** (ΔT °C, ohms, or “visible only”). Human owns the decision.

---

## 5. Pipeline (target)

```
image or video
  → scene router: pcb_chip | panel_switch | cable
  → component locator (panel: breaker/lug/bus; pcb: optional board ROI)
  → defect YOLO-seg (problem boxes only)
  → optional: golden-board residual (PCB) or IR hotspot (panel, under load)
  → fuse boxes on the same ID
  → if video: ByteTrack
  → report: crop + class + confidence + extra sensor
  → never: “complete” / “safe”
```

If defect YOLO returns **no boxes:**  
`No visible defect in this view.`  
Not: `Circuit is complete.`

---

## 6. What we already have vs what to change

| Item | Today | Target |
|------|--------|--------|
| Classes | complete, incomplete, wires_touching, damage | **Drop complete.** Keep open/short/damage; split names by scene |
| `complete` mAP | 0.995 (misleading) | Remove from product metrics |
| Open / short mAP50-95 | 0.48 / 0.45 | Raise with 640 px, tiles, PCB-IND, template residual |
| Data | PKU + DeepPCB + cable | + PCB-IND (already downloaded); + SolDef for chips; own panel RGB+IR |
| Sensors | RGB only | Panel: radiometric IR. Chip: optional microscope + bench continuity |

WP1c weights (`models/circuit_faults.pt`) stay as an **investor bootstrap**. They are not the product ontology. Next train: **defect-only yaml**, no `complete` images.

---

## 7. Work plan

| WP | Work | Done when |
|----|------|-----------|
| **D1** | Rebuild dataset **without** `complete`. Map PCB-IND + PKU + DeepPCB + cable to `open` / `short` / `damage`. | `data/public/defects-only/` + yaml |
| **D2** | Retrain YOLOv8/v11n or s at **640 px** (GPU). Report per-class mAP on open/short/damage only. | `models/defects_only.pt` |
| **D3** | Tiled inference (SAHI) at test time; compare tiny-open recall vs full-frame | Table in this repo |
| **D4** | Cascade: detect board/panel ROI, then defects inside | Lower false positives on cluttered photos |
| **D5** | PCB golden-board prototype on DeepPCB pairs | Residual heatmap + YOLO |
| **D6** | Panel component locator (`breaker`, `lug`, …) | Needed before ΔT |
| **D7** | Radiometric IR + box on hot lug | Product localization for switches |
| **D8** | Own labeled **defect boxes** on real DBs and one chip board | Domain match |

Order: D1–D2 are software on data we already have. D5–D7 are the accuracy jump. D8 is the only path to field claims.

---

## 8. Safety and claims

- Screening aid. Live panels: electrician only.
- No boxes ≠ electrically sound.
- PCB-IND / DeepPCB mAP ≠ Pakistani consumer-unit accuracy.
- Do not quote paper mAP (Tao 98.7%, Chen 99% sim) as this product.

---

## 9. References (in-repo)

- Feasibility: [`REPORT.md`](REPORT.md)  
- Architecture (YOLO locates, ΔT judges): [`architecture.md`](architecture.md)  
- Two-scale sensors: [`TWO_SCALE_PLAN.md`](TWO_SCALE_PLAN.md)  
- Current 4-class run: [`CIRCUIT_FAULTS.md`](CIRCUIT_FAULTS.md) — historical; `complete` retired for product  
- Capture: [`capture_protocol.md`](capture_protocol.md)

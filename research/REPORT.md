# Feasibility report: vision-based electrical circuit / wiring fault detection

**Project:** `circuit_fault_vision` (`F:\pange\circuit_fault_vision`)  
**Date:** 2026-08-15  
**Status:** research only — no model trained  
**Question:** Can YOLO, ByteTrack, and heatmap / thermal images find problems in home electrical circuits and wiring? Do we need our own labeled data?

---

## 1. Short answer

**Yes, for a specific class of problems. No, as a general “scan the house and find any wiring issue” product.**

Computer vision can find:

1. **Heat-producing faults on accessible equipment** — loose lugs, overloaded breakers, unbalanced phases, hot joints on exposed cables — using **thermal infrared** plus a detector (YOLO or similar).
2. **Visually obvious damage** on exposed cables and boards — burns, melted plastic, broken strands, chewed insulation, missing covers — using **RGB** YOLO / segmentation.

It cannot, from a phone photo of a wall, find:

- A broken or high-resistance joint **inside** plaster, conduit, or insulation unless enough heat reaches the surface (usually it does not).
- An **open circuit** (no current → no heat).
- Insulation breakdown that has not yet heated.
- Intermittent faults that are not loaded at capture time.
- Code / safety-test failures (earth, polarity, insulation resistance).

That split is not a model-quality issue. It is physics. Thermal cameras measure **surface IR**, not X-ray of conductors. NFPA 70B and professional thermography treat IR as a **screening** method under load, not a replacement for electrical testing.

**You will need to train your own model** if the target is residential / building consumer units. Public datasets exist and are useful for bootstrapping, but they are almost all **substations, industrial switchgear, and overhead lines**. Domain gap to a home DB board in Pakistan (or anywhere) is large.

---

## 2. What the user idea maps to (and what it does not)

The proposed stack was: **images → YOLO (or similar) → ByteTrack → heatmaps → find circuit disparities**.

Those three tools solve **three different jobs**. Mixing them without that distinction is the main design risk.

| Tool | Real job | Role in this product |
|------|----------|----------------------|
| YOLO (detect / segment / classify) | Find objects and defects in a frame | **Core.** Component finder + visible-defect finder + hotspot box on thermal frames |
| ByteTrack | Keep the same object ID across video frames | **Helper.** Useful when walking a cable tray, panning a panel, UAV, or robot. Useless on a single still |
| “Heatmap” (Ultralytics / Supervision) | Accumulate *where tracked objects moved* | **Wrong signal** for electrical faults. That is crowd/traffic density |
| Thermal IR image (“heatmap” in common speech) | False-color picture of surface temperature | **Core input modality** |
| Radiometric ΔT | Per-pixel temperature vs a sibling component or ambient | **The actual fault score** for overheating |

So the right reading of the idea is:

> Dual-camera inspection: YOLO localizes electrical parts and visible damage; thermal (+ ΔT rules) finds abnormal heating; ByteTrack is used only when the input is video so the same breaker/cable is not counted twice.

That pipeline is real. It is close to published work on switchgear and transmission inspection, and to how FLIR / Fluke products are used — except those products still rely on a human thermographer for ΔT, and AI papers mostly detect *equipment* or *hot blobs*, not a full residential fault ontology.

---

## 3. Physics and inspection standards (this constrains the ML)

Electrical faults that vision can catch are mostly **I²R heating** at high resistance (loose/corroded connections), overload, or harmonic heating — plus **visible mechanical damage**.

### 3.1 Thermal does not see through walls

A thermal camera cannot X-ray plasterboard, brick, conduit, or insulation. It sees the **surface**. An overloaded cable in an insulated cavity often leaves little or no surface signature. Accessible panel interiors, exposed tails, busbars, and cable trays are the realistic targets.

### 3.2 Load is mandatory

NFPA 70B (2023) requires thermography to measure **Delta T** of similar components under similar loading, and vs ambient. Industry practice: inspect at **≥ ~40% of rated load**, ideally near peak. A lightly loaded loose lug looks “healthy.”

### 3.3 Severity is ΔT, not a YOLO class name

NETA MTS Table 100.18 (widely used guideline; not a physical law):

**Similar components, similar load**

| ΔT | Suggested action |
|----|------------------|
| 1–3 °C | Possible deficiency; investigate |
| 4–15 °C | Probable deficiency; repair as time permits |
| > 15 °C | Major; repair immediately |

**Component vs ambient**

| ΔT | Suggested action |
|----|------------------|
| 1–10 °C | Investigate |
| 11–20 °C | Repair as time permits |
| 21–40 °C | Monitor until repair |
| > 40 °C | Immediate repair |

A detector that only outputs `hotspot` without a temperature number cannot implement this. **False-color JPEGs are not enough.** We need **radiometric** files (FLIR radiometric JPEG / `.fff` / SDK) or at least calibrated temperature arrays.

### 3.4 What IR will miss even on an open panel

- Failing insulation with no heat
- Cracked busbar not yet arcing
- Wrong breaker rating / double-tap with no extra heat at that moment
- Anything behind a closed dead-front if heat does not couple to the cover (covers should be removed by a qualified person for a real survey)

---

## 4. Is it possible? By fault type

| Fault | RGB YOLO | Thermal YOLO / ΔT | Notes |
|-------|----------|-------------------|--------|
| Loose / corroded lug | Weak (maybe discoloration) | **Strong** under load | Best thermal use case |
| Overloaded circuit / undersized conductor | Weak | **Strong** if accessible | Compare siblings |
| Failing breaker (hot) | Weak | **Strong** | Classic home-inspection finding |
| Phase imbalance | No | **Medium** | Needs 3-phase comparison, not just a box |
| Burned / melted insulation | **Strong** | Medium (heat may be gone) | RGB after the fact |
| Exposed copper / missing sheath | **Strong** | No | Visible only |
| Broken strand / chew / cut | **Medium–strong** if close-up | No | Needs our own labels for home cables |
| Open / disconnected wire | Only if the break is visible | **No** (no current) | Electrical test, not vision |
| In-wall damaged NM / twin-and-earth | Almost never | Rare faint surface heat | Do not sell this as a feature |
| Arc tracking / carbonized bus | **Medium** if visible | **Strong** if still heating | |
| Ground / insulation resistance fail | No | No | Megger / EICR |

**Product-shaped version that is honest:** “AI-assisted thermal + visual screening of **distribution boards, accessible junctions, and exposed cable runs**.” Not “find every circuit break in the house.”

---

## 5. Do we train our own model?

**Yes**, for anything we would actually deploy on homes.

Reasons:

1. **COCO / ImageNet YOLO** has no classes for `mcb`, `busbar_lug`, `hot_neutral`, `burned_insulation`.
2. Public electrical datasets are **wrong domain**: 132 kV yards, UAV insulators, factory cable close-ups, Chinese switchgear IR.
3. **Local appearance** matters: Pakistani / South Asian consumer units, wiring colors, DB layouts, and camera types will not match FLIR T600 substation sets.
4. Papers that report 95–99% mAP almost always train **on their own** IR set (sometimes simulated). Those numbers will not transfer.
5. Thermal false-color palettes (iron, rainbow, white-hot) are a domain shift of their own. A model trained on rainbow palettes fails on ironbow unless you train on temperature arrays or many palettes.

**What we should not do:** train from scratch on ImageNet-scale data. Fine-tune Ultralytics YOLO (nano/small for edge, medium for accuracy) from COCO weights. That is standard and sufficient at this data scale (thousands of images, not millions).

**What we should do first:** a **baseline on public data** to prove the toolchain, then collect our own paired RGB+IR.

Unsupervised anomaly detection (MVTec-style: train only on “good” cables) is attractive because faults are rare. It is a good **second head** for close-up cable photos in a lab, not a replacement for YOLO on cluttered panels.

---

## 6. Public labeled data — what exists

See [`datasets.md`](datasets.md) for URLs and notes. Summary:

### Useful for bootstrapping (not for shipping)

| Dataset | Size (approx.) | Modality | What it actually labels | Home-wiring fit |
|---------|----------------|----------|-------------------------|-----------------|
| FIRC / 6-class IR faults (papers + Chinese mirrors) | 1,729 imgs, 6 classes | Thermal | Loose connection, insulation, CB fault, overload, phase imbalance, normal | **Best conceptual match** — industrial switchgear, not homes; availability / license must be checked |
| Roboflow Thermal Image V2 | 1,221 | Thermal | Busway, equipment, NFB (breaker) | Component location |
| Roboflow thermal substation | ~1,670 used in a 2025 paper | Thermal | Isolator, CB, bushing, CT, insulator | HV yard, not homes |
| ScienceDB HV IR (FLIR C5) | 895 | Thermal | 5 HV equipment types | Recognition, not fault boxes |
| SCITD (paper, 2026) | 5,600 annotated | Thermal | Normal vs abnormal heating | Likely **not public** |
| Switchgear IR GitHub (~5.5k) | ~5,500 | Thermal | 8 *parts* (core, body, CT…) | Locates parts, not “fault vs OK” |
| RF100 cable-damage | 1,318 | RGB | `break`, `thunderbolt` | Outdoor cable damage, not NM/T&E |
| PowerLine-MTYOLO MPCD | 1,871 | RGB | Cable masks + broken strands | UAV overhead |
| TTPLA | 1,100 (4K) | RGB | Towers + lines (instance seg) | HV aerial |
| CPLID | 848 | RGB | Insulator / broken insulator | HV |
| MVTec AD cable | ~388 | RGB | 8 factory cable defects + masks | Lab bench, CC-BY-NC |
| Roboflow electrical panel | 256 | RGB | Contactor, timer, MCB | Tiny; components only |
| Kaggle “Electrical Wiring Faults Detection” | listed | ? | Name sounds on-topic | Page was not readable here; inspect before relying |
| Cable-Thermo (YOLO26 paper, 2026) | simulated | Thermal (ANSYS) | 4 underground duct defects | Sim-to-real gap; not homes |

**There is no strong, public, YOLO-ready residential dataset** of paired RGB+IR home consumer units with fault labels. That is the gap we would fill.

Published YOLO-on-IR papers (YOLOv7/v8/v26, GD-YOLO, YOLO_ViT_CNN) confirm the **method** works on their sets (often mAP@0.5 in the high 80s–90s). Treat those scores as “IR hot blobs are learnable,” not as expected production accuracy on our cameras.

---

## 7. Recommended technical approach

Detail in [`architecture.md`](architecture.md). Condensed:

### Stage 0 — Hardware

- RGB: any 1080p+ phone or USB cam.
- Thermal: **radiometric** camera. Minimum viable: FLIR One Pro (160×120 native, radiometric JPEG, ±3 °C — OK for relative ΔT, weak for certified reports). Better: FLIR C5 / E5–E8 class. Lab/edge papers use Lepton 3.5 + Jetson.
- Capture **under load**. Record load if possible (clamp meter).
- Safety: qualified person for open live panels.

### Stage 1 — Two YOLO heads (or one multi-task)

1. **RGB:** classes such as `panel`, `breaker`, `busbar`, `cable`, `outlet`, `junction`, plus defect classes `burn`, `exposed_conductor`, `broken`, `melted`, `chew` (start smaller; grow later).
2. **Thermal:** component classes + `thermal_anomaly` **or** skip the anomaly class and run ΔT on segmented components.

YOLO-seg (instance masks) is better than boxes for cables and for averaging temperature inside a mask.

### Stage 2 — Physics layer (do not learn this if we have numbers)

For each thermal instance:

1. Read mean / max temperature in the mask from radiometric data.
2. Compare to sibling components (other breakers on the same bus) and to ambient.
3. Map ΔT → NETA-style severity.
4. Suppress “hot” detections that are actually reflections on shiny metal (emissivity trap). Prefer comparing identical components.

This is how the 2026 Scientific Reports breaker paper does it (segmentation + temperature KDE), and it is more trustworthy than asking YOLO to name `overload` vs `loose_connection` from a blob. Those two can look identical in IR.

### Stage 3 — ByteTrack (video only)

Use Ultralytics `model.track(..., tracker="bytetrack.yaml")` so a hotspot on breaker #4 stays ID 4 while the inspector pans. Optional: accumulate a **spatial hotspot map of that panel**, not a movement heatmap of the inspector.

### Stage 4 — Fusion

RGB defect **or** thermal severity ≥ threshold → flag. Show both photos (NFPA-style: thermal + visible). Never auto-diagnose “replace this circuit” from CV alone.

### Model family

Start with **Ultralytics YOLOv8/v11** (stable ecosystem, built-in ByteTrack). A 2026 paper used YOLO26 on simulated cable IR; we can swap when the stack is mature. Do not invent a ViT hybrid until a plain YOLO baseline exists.

---

## 8. What I would not do

- Train only on RGB and hope to find hidden wiring faults.
- Use ByteTrack heatmaps as the electrical signal.
- Treat rainbow JPEGs as temperature.
- Trust 99% mAP from simulated or single-site IR papers.
- Open live boards without a qualified electrician.
- Promise fire-prevention or “certified inspection.”

---

## 9. Suggested phases

| Phase | Work | Outcome | Effort (order of magnitude) |
|-------|------|---------|-----------------------------|
| **0** | This research | Go / no-go | Done |
| **1** | Download 2–3 public sets; train a throwaway YOLO | Toolchain works; see domain gap | Days |
| **2** | Camera + capture protocol + 200–500 paired board images (normal + staged/known faults) | First real dataset | Weeks + electrician time |
| **3** | Label (CVAT/Roboflow); train RGB + IR YOLO-seg; ΔT rules | Internal demo on our boards | Weeks |
| **4** | More sites, more load conditions, hard negatives (reflections, sunlight) | Something that might generalize | Months |
| **5** | Video + ByteTrack + simple report UI | Inspector-style app | After 3 is solid |

Minimum labeled set for a *demo* on one board type: on the order of **300–800 images per modality**, with faults oversampled. For a product that generalizes across homes: **thousands**, multiple boards, multiple cameras, multiple loads. Faults are rare — we will need staged faults (controlled loose lug, extra load) plus real inspection leftovers, not only “wait for fires.”

---

## 10. Risks

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Domain gap | Public IR ≠ home DBs | Collect own data; public only for pretrain |
| Palette / camera shift | Ironbow vs rainbow vs raw temp | Train on temperature arrays, not only RGB-mapped IR |
| Class confusion | Overload vs loose lug look similar in IR | ΔT + RGB context; don’t over-claim class names |
| Reflections / emissivity | Shiny copper looks “cold” or “hot” | Prefer relative sibling ΔT; dull tape dots in lab |
| Safety / liability | Wrong “all clear” | Screening UX; no certification language |
| Rare faults | Models bias to “normal” | Staged faults, heavy augmentation, anomaly head |
| MVTec NC license | Cannot use commercially | Keep MVTec out of any commercial train set |
| Live capture | Arc flash | Electrician-only open-panel protocol |

---

## 11. Bottom line

The idea is **sound if we scope it as thermal + visual screening of accessible electrical equipment**, which is already how professional thermography works, with YOLO automating localization and ByteTrack helping on video.

It is **not sound** as “YOLO looks at a house photo and finds circuit breaks in the walls.”

**Train our own model: yes.** Public data: use as a bootstrap, not as the product dataset. **Heatmaps: thermal radiometry, not tracking density.** **ByteTrack: later, for video.**

If we proceed, Phase 1 is a public-data YOLO smoke test; Phase 2 is a radiometric camera and a capture session on real boards. Everything else is downstream of those two.

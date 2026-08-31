# Two-scale fault localization — beyond RGB YOLO

**Date:** 2026-08-27  
**Question:** After the 4-class vision bootstrap (test mAP50 0.831), how do we actually **pinpoint where** a fault is — on **small PCBs / chips** and on **large switches / boards** — and what else besides a vision model should we add?

**Locked:** YOLO draws **defect boxes** (where the problem is). Do **not** classify complete vs incomplete on chips or panels. Physics / electrical tests judge continuity. Never print “safe.” Screening aid only.

Product ontology: [`DEFECT_BOX_REPORT.md`](DEFECT_BOX_REPORT.md).

---

## 1. Why vision alone is not enough

The WP1c bootstrap still has boxes of `complete` / `incomplete` / `wires_touching` / `damage` on PCB traces and outdoor cables. **Product ontology is defect-only** ([`DEFECT_BOX_REPORT.md`](DEFECT_BOX_REPORT.md)): drop `complete`; box the problem. That bootstrap is still not a locator for a live consumer unit or a populated chip board.

| What RGB YOLO can do | What it cannot do |
|----------------------|-------------------|
| See an open/short **on a visible copper trace** | See a short **under** an IC, BGA, or solder mask |
| See burnt insulation, missing covers | Find a high-resistance lug that looks fine until it is **hot under load** |
| Roughly where on the **photo** | Where on the **net / pole / cubicle** in electrical terms |
| Cheap, camera-only demo | Replace ICT, TDR, megger, or NETA thermography |

Two physical scales need **different extra sensors**. One YOLO head can still **route** the photo (“this is a PCB” vs “this is a breaker cubicle”), then specialist heads + instruments run.

```
photo / video
    → YOLO locator (pcb | ic | solder | breaker | lug | bus | cable | mccb)
    → if pcb/chip  → AOI defects + optional thermal + electrical net test
    → if switch    → RGB damage + radiometric ΔT + optional PD / clamp / TDR
    → fuse on the same component ID
    → report: photo + location + evidence + “electrician to confirm”
```

---

## 2. Chip / small PCB path

Goal: *which pad, pin, or net is bad?*

### 2.1 Keep and improve vision

1. **AOI YOLO** (what we have): open, short, mouse-bite, spur, missing copper. Next: merge **PCB-IND** (4,789 real AOI 300×300 patches, CC BY 4.0, eight classes including open/short). Real production line, better than PKU’s often-photoshopped defects.
2. **Golden-board compare** (DeepPCB idea): align a defect-free template to the board under test. Difference image finds **where** a trace changed. YOLO then **names** the defect. This is how factory AOI actually works; class-only YOLO is weaker on tiny opens.
3. **Solder / SMT** (chips): SolDef_AI (~1,150 images, CC BY 4.0) — misalignment, poor solder, spikes. Separate head: `solder_poor`, `misaligned`, `bridge` (no healthy/`solder_ok` class). Do not mix these ids with the RGB defect head.
4. **Close-up, not phone-from-2m.** Chip faults need 5–20 cm, decent light, maybe a cheap USB microscope.

### 2.2 Non-vision (this is how factories find the *net*)

| Method | What it localizes | When to add |
|--------|-------------------|-------------|
| **Continuity / TDR on a trace** | Distance-to-fault along a conductor | Lab / repair bench; not a walk-by camera |
| **ICT / flying probe** | Which component or net is open/short | Manufacturing; fixture cost |
| **Boundary scan (JTAG)** | Digital nets on scan-capable ICs | If the board supports it |
| **AXI / X-ray** | Hidden BGA voids, inner-layer shorts | Industrial only |
| **Point thermal of ICs** | Hot die / shorted regulator | Cheap IR after YOLO finds the package |
| **ICT vs AOI** | AOI sees solder shape; ICT measures ohms | Complementary, not competing |

**Product implication:** for **chip boards**, vision is AOI screening. The *where* on the schematic is electrical. A v2 feature is: YOLO box → operator probes that pad with a continuity beep / TDR, result stored on the same ID.

---

## 3. Large switch / DB / MCCB path

Goal: *which pole, lug, or cubicle is bad?*

This is closer to the original PanGe product (home/building boards) than PCB AOI is.

### 3.1 Vision (keep)

- RGB: `burn_mark`, `exposed_conductor`, `missing_cover`, `melted_plastic`
- **Component locator** (missing today): `breaker`, `busbar`, `lug`, `cable_tail`, `meter`, `mccb` — so ΔT is averaged on the **right object**, not a random hot blob
- ByteTrack only for walkthrough video (same breaker ID)

### 3.2 Non-vision (this is how thermographers and switchgear techs work)

| Method | Localizes | Fit for us |
|--------|-----------|------------|
| **Radiometric IR + NETA ΔT** | Which lug/pole is hotter than siblings under load | **Next real product sensor.** Already locked. |
| **Clamp meter / CT / PQ** | Overload, imbalance, harmonics on a named circuit | Tag reading to the YOLO breaker ID |
| **Insulation resistance / loop / RCD test** | Pass/fail of a circuit, not a photo box | Electrician workflow; we store the result, we do not fake it |
| **TDR from the board** | Distance to open/short on the **outgoing cable** | Complements RGB `open` when the break is in-wall |
| **TEV / ultrasonic / UHF partial discharge** | Insulation defect **inside** a metal-clad cubicle | Industrial MV switchgear, not a home DB v1 |
| **UV corona camera** | Surface corona on HV | Substation, not homes |
| **Arc-flash / EMI audio** | Live arcing | Later; noisy; safety-critical |

**Product implication:** for **big switches**, the stack is **YOLO (where is the part) + IR (is it hot) + optional clamp (is it loaded)**. PD sensors only if the client is MV switchgear, not Pakistani residential DBs.

---

## 4. Datasets — have vs get next

### Already local

| Set | Scale | Role |
|-----|--------|------|
| RF100 cable-damage | Cables | `damage` |
| PKU-Market-PCB | Bare PCB | open/short/damage (some defects synthetic) |
| DeepPCB | Bare PCB traces | open/short/damage; template-compare research |
| circuit-faults merge | Mix | 4-class YOLO, test mAP50 0.831 |
| home-wires synthetic | Drawings | Do not use as the real-wire demo |

### Gather next (public, licensed)

| Priority | Set | Size | License | Why |
|----------|-----|------|---------|-----|
| **1 now** | **PCB-IND** (Zenodo 19723114) | 4,789 AOI patches, 101 MB | CC BY 4.0 | Real production open/short; merge into **defect-only** (no `complete`) |
| **2** | **SolDef_AI** (MDPI / Kaggle) | ~1,150 SMT | CC BY 4.0 | Chip solder / misalignment |
| **3** | Roboflow Thermal Image V2 | 1,221 IR | check export | Breaker/NFB localization in IR |
| **4** | FIRC-style 6-class IR | ~1.7k | messy | Switchgear fault *types* — only if license is clean |
| **5** | Own capture | TBD | ours | Home DBs + local switchgear; only set that matches the product |

Skip unless asked: `sriom1/electrical-panels-dataset` (13 GB auto-labels), MVTec cable (non-commercial).

PCB-IND download: `python -m src.download_pcb_ind` → `data/public/pcb-ind`.

---

## 5. Work packages (after WP1c)

| WP | What | Outcome |
|----|------|---------|
| **WP2 / D1–D2** | Rebuild **without** `complete`; merge PCB-IND as open/short/damage; GPU retrain **640 px** | Stronger **chip/PCB** defect boxes |
| **WP3** | Separate **component-locator** YOLO: pcb vs breaker vs cable vs IC | Router for the two-scale pipeline |
| **WP4** | Radiometric camera + ΔT rules on named components | **Switch** localization that vision cannot do |
| **WP5** | SolDef_AI solder head (do not mix class ids) | Chip assembly defects |
| **WP6** | Capture protocol: paired RGB+IR of local DBs **and** one industrial switchgear cubicle | Domain data |
| **WP7** | Sidecar measurements: clamp + optional TDR, stored on YOLO ID | “Where on the circuit” not just “where in the photo” |
| **WP8** | PD (TEV/ultrasonic) **only** if a client is MV switchgear | Out of home-DB v1 |

---

## 6. Honest limits (repeat)

- Chip AOI mAP ≠ field accuracy on a Pakistani DB.
- Complete-class 0.995 on full-crop boxes is not “the board is electrically complete.”
- In-wall opens need **TDR or a megger**, not a thermal photo.
- Live panels: electrician only.

# Public dataset catalog

Inventoried 2026-08-15 for `circuit_fault_vision`. None of these is a drop-in residential home-wiring set. Use them to **bootstrap** YOLO, then collect our own paired RGB+IR.

Download targets later into `data/public/` (gitignored). Do not commit images.

---

## A. Thermal / infrared (closest to “heatmap” faults)

### A1. FIRC-style 6-class electrical IR faults

- **Why it matters:** Same ontology we care about: loose/corroded connection, deteriorated insulation, faulty circuit breaker, overloaded circuit, phase imbalance, normal.
- **Size:** 1,729 images, 2,616 boxes (reported on Chinese dataset mirrors). Box counts: normal 666, loose/corroded 616, overload 528, CB 371, phase imbalance 219, insulation 216.
- **Format:** VOC + YOLO on some mirrors.
- **Used by:** YOLO_ViT_CNN paper (IJIRT; claims mAP@0.5 95.6% after 5 epochs — treat as optimistic / possibly overfit).
- **Fit:** High conceptually, **industrial switchgear**, not home DBs. License and canonical download URL are messy (mirrors / paid Chinese blogs). Verify license before training.
- **Action:** Hunt a clean copy (Roboflow / paper supplementary / authors). If found, this is the first thermal pretrain.

### A2. Roboflow — Thermal Image V2

- URL: https://universe.roboflow.com/electrical-equipments-detection/thermal-image-v2
- **Size:** 1,221 images (version 1).
- **Classes:** Bus_Way, electrical-equipment, NFB (no-fuse breaker).
- **Fit:** Component localization in IR, not fault types.
- **Action:** Easy YOLO export. Good Phase-1 “can we detect a breaker in IR?” baseline.

### A3. Roboflow — Thermal substation components (ATPD)

- URL: https://universe.roboflow.com/atpd/thermal-substation-components-mlyrb
- **Classes:** Circuit breakers, disconnectors, power transformers, surge arresters, wave traps.
- **Also:** ~1,670 FLIR T600 images (7 classes) used in *Appl. Sci.* 2025, 15, 328 (improved YOLOv8). Mix of normal and fault *states* at equipment level.
- **Fit:** HV substation. Useful IR pretrain, wrong objects for homes.

### A4. ScienceDB — HV equipment IR (FLIR C5)

- DOI: https://doi.org/10.57760/sciencedb.10185
- **Size:** 203 CB + 178 transformers + 181 arresters + 180 disconnectors + 153 wave traps = **895** rainbow-palette 640×480 IR images, folder-labeled (classification, not boxes).
- **Fit:** Equipment ID under varying load/time of day. Would need us to add boxes / hotspot labels. Rainbow palette ≠ radiometric.

### A5. SCITD — Switchgear Cabinet Infrared Thermal Imaging Dataset

- Paper: *A method for detecting abnormal heating in switchgear based on SAHDA* (2026).
- **Size:** 5,600 annotated (600 normal, 5,000 abnormal). Built from >10k raw (field + lab + atlases), then augmented.
- **Fit:** Excellent task match (abnormal heating). **Assume not public** until authors release it. Do not scrape papers.

### A6. Switchgear IR overheat (GitHub example repo)

- https://github.com/QQ767172261/Use-YOLOv8-to-train-switchgear-infrared-superheat-image-dataset-to-build-a-comprehensive-deep-learni
- **Size:** ~5,500 VOC images, 8 **part** classes (core, connection, body, load switch, arrester, CT, VT, MCCB).
- **Fit:** Localizes *which part* is in view on an overheat photo; not a fault taxonomy. Dataset files may live on CSDN, not in the repo.

### A7. Cable-Thermo (simulated underground duct IR)

- Paper: *Enhanced YOLO26 for Thermographic Fault Detection in Underground Duct Cables*, Appl. Sci. 2026.
- **Classes:** hollow damage, conductor burnout, sheath damage, severe damage. ANSYS thermoelectric simulation.
- **Reported:** mAP50 ~99% on the sim set; authors themselves flag sim-to-real as future work.
- **Fit:** Method reference only. Do not expect 99% on real cameras.

### A8. MDPI Energies 2020 HV thermal (binary)

- https://www.mdpi.com/1996-1073/13/2/392
- **Size:** 1,075 defective + 925 non-defective = 2,000 instances (classification).
- **Fit:** Binary IR “bad vs good” on HV gear. Old backbone (AlexNet-era). Low priority.

---

## B. RGB visible cable / wire damage

### B1. Roboflow 100 — cable-damage (best public RGB start)

- Universe: https://universe.roboflow.com/roboflow-100/cable-damage
- Hugging Face: https://huggingface.co/datasets/LibreYOLO/cable-damage
- **Size:** 1,318 (train 919 / val 265 / test 134).
- **Classes:** `break`, `thunderbolt`.
- **Fit:** Outdoor / utility cable damage, not household T&E. Fine for a RGB YOLO smoke test.
- **Local copy (2026-08-27):** `data/public/cable-damage` (gitignored). Train: `python -m src.train`. Investor note: `research/INVESTOR_PROOF.md`.

### B2. PowerLine-MTYOLO — MPCD

- https://github.com/phd-benel/powerline-mtyolo
- **Size:** 1,871 images; 2,501 cable masks; 1,906 broken-strand boxes. Merged from 5 Roboflow sets.
- **Fit:** UAV overhead. Multitask detect+seg is a good architecture reference.

### B3. TTPLA — towers and lines

- Paper: ACCV 2020. GitHub: https://github.com/R3ab/ttpla_dataset
- **Size:** 1,100 images at 3840×2160; 8,987 instances. Instance segmentation.
- **Fit:** Asset mapping, not defect detection. Baseline mask AP was low (~16%) — thin cables are hard.

### B4. CPLID — broken insulators

- 600 real normal + 248 **synthetic** defective insulators.
- **Fit:** HV insulators only. Synthetic defects = optimistic bias.

### B5. MVTec AD — cable subset

- https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- **Size:** train 224 good; test 58 good + ~106 defects across `bent_wire`, `cable_swap`, `combined`, `cut_inner_insulation`, `cut_outer_insulation`, `missing_cable`, `missing_wire`, `poke_insulation`. Pixel masks.
- **License:** **CC BY-NC-SA 4.0 — no commercial use.**
- **Fit:** Unsupervised anomaly-detection experiments only. Lab lighting, aligned cables. Keep out of any commercial train mix.

### B6. Other small Roboflow sets

- Cable Anomaly (oumayma tajir): 136 images, MVTec-like classes — too small alone.
- `wire_defect_ins`: 261 images, `broken` / `wire` / `extrusion` / `twist`.
- Electrical panel (IMGPRO): 256 images, `contactor`, timers, `mcb`, overload relay — **component** labels, useful later for panel parsing.

### B7. Kaggle — Electrical Wiring Faults Detection

- https://www.kaggle.com/datasets/warcoder/electrical-wiring-faults-detection
- Name is on-topic. Treat as **inspect before use** (license / whether it is images vs tabular).

## B8. Home-circuit conductors (red / blue / IEC) — 2026-08-27

There is **no good public YOLO set of residential T&E** (red live / blue neutral / brown / earth). Closest public hits:

| Set | Why it does not replace ours |
|-----|------------------------------|
| Roboflow `wire-detection-mupmh` | Color codes (red, LBU, BK, YG) but ~140 images; Roboflow login to export |
| HF `sriom1/electrical-panels-dataset` | 13 GB, 107 classes, **auto YOLOE labels**, not wire colors |
| REMODEL electric wires | 39 GB chroma-key robot cables, segmentation, not home DBs |
| RF100 cable-damage | Outdoor metal strands — **wrong domain** for red/blue PVC |

**What we did first:** generate `data/public/home-wires` via `python -m src.build_home_wires` (synthetic IEC + legacy colors). That is **not** real-world wire photos.

**What we did next (2026-08-27):** merge **real photographs** into `data/public/circuit-faults` with classes `complete` / `incomplete` / `wires_touching` / `damage`:

| Source | Local path | Role |
|--------|------------|------|
| PKU-Market-PCB / HRIPCB | `data/public/pku-pcb-raw` (HF `RobotHuman/PCB_defect`) | Color PCB photos; open → incomplete, short → wires_touching, other defects → damage; intact crops → complete |
| DeepPCB | `data/public/deeppcb-raw` (HF `thangkt/PCB-Prune-YOLO-DeepPCB`) | Real CCD traces; same open/short/damage map |
| RF100 cable-damage | `data/public/cable-damage` | Real outdoor cables → damage |

Train: `python -m src.train --preset circuit_faults --device 0`. Note: [`research/CIRCUIT_FAULTS.md`](CIRCUIT_FAULTS.md).

Honest limit: PCB traces ≠ home T&E. Synthetic home-wires mAP is a **color/shape bootstrap**. Circuit-faults mAP is **not** field accuracy on a Pakistani consumer unit. Mix `data/own/`.

**2026-08-30:** defect-only mix `home-faults` (drop complete + PCB-IND + stripped-wire). See [`HOME_FAULTS.md`](HOME_FAULTS.md). Best real **home socket** set (HazardDetector, ~6k) needs a Roboflow API key.

---

## B10. Home sockets / wires (2026-08-30 search)

| Set | n | Classes | License / get |
|-----|---|---------|----------------|
| Roboflow electrical-hazards / HazardDetector | ~6.1–6.5k | burned socket, damage wire, open copper, overloaded socket | Public Domain listing; **export needs Roboflow key** |
| Stripped Wire (Zenodo 16686806) | 503 jpg (167 defect) | good / pulled / cut strands | Downloaded → `data/public/stripped-wire` |
| EnergAI fuses | ~4.3k | fuse types | 3.4 GB; skipped (locator) |
| WireWise preprint | — | claimed home wiring YOLO | No image files on Zenodo |
| Indoor sockets / switches (Zenodo 18835199) | 3,459 | power socket, light switch, power strip | **Downloaded** → `data/public/indoor-sockets`. Locator, not defects. Used as empty-label backgrounds in home-faults. |

Merged train set: [`HOME_FAULTS.md`](HOME_FAULTS.md). Command: `python -m src.train --preset home_faults`.

---

## B9. Chip / industrial PCB (2026-08-27)

- Zenodo: https://doi.org/10.5281/zenodo.19723114
- **Size:** 4,789 real AOI 300×300 patches (3833/478/478). ~101 MB.
- **Classes:** mouse_bite, missing_copper, scratch, spurious_copper, copper_burr, stain, short, open.
- **License:** CC BY 4.0.
- **Fit:** Best public **real production-line** PCB set for open/short. **Product map (defect-only):** open→`open`, short→`short`, rest→`damage`. Do not add a `complete` class.
- **Command:** `python -m src.download_pcb_ind` → `data/public/pcb-ind`.
- **Local (2026-08-27):** downloaded. YOLO/VOC/COCO copies on disk (same 4,789 patches in three formats).

### SolDef_AI (solder / SMT chips)

- Paper: *J. Manuf. Mater. Process.* 2024, 8(3), 117. CC BY 4.0.
- **Size:** ~1,150 images of 0603/0805/1206 parts, three viewpoints.
- **Classes:** position/misalignment + poor solder (excess/insufficient), spikes.
- **Fit:** **Chip** assembly, not bare traces. Separate YOLO head. Kaggle mirror cited in follow-on YOLO11-seg papers.
- **Action:** download when WP5 starts; do not mix ids with circuit-faults.

### PKU + DeepPCB (already local)

See B8 / CIRCUIT_FAULTS.md. DeepPCB templates also support **golden-board difference**, not only class YOLO.

---

## C. Survey paper (more datasets)

*Overview of Image Datasets for Deep Learning Applications in Diagnostics of Power Infrastructure* — PMC10459611. Catalog of TTPLA, CPLID, IDID, PLAD, etc. Almost entirely **overhead / HV**. Confirms the residential gap.

---

## D. Recommended download order (when we leave research)

1. ~~RF100 cable-damage~~ **Done.**
2. ~~PKU + DeepPCB 4-class merge~~ **Done** (`--preset circuit_faults`, test mAP50 0.831).
3. **PCB-IND** (real AOI open/short) — `python -m src.download_pcb_ind`. **On disk** 2026-08-27.
4. ~~Stripped-wire + indoor sockets~~ **Done** 2026-08-30 (`--preset home_faults`).
5. **HazardDetector** if `ROBOFLOW_API_KEY` is available (~6k burned sockets).
6. Thermal Image V2 (IR component detection).
7. SolDef_AI for chip solder.
8. FIRC 6-class IR only if a clean licensed copy exists.
9. Skip MVTec for anything that might be commercial.

**Do not wait on public data to be “enough.”** It will not be. Start the capture protocol in parallel: [`capture_protocol.md`](capture_protocol.md). Two-scale plan: [`TWO_SCALE_PLAN.md`](TWO_SCALE_PLAN.md).

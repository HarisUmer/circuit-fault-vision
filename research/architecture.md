# Recommended architecture

Research-phase design for `circuit_fault_vision`. Not implemented yet.

## Design principle

**YOLO localizes. Radiometry judges heat. RGB judges visible damage. ByteTrack is only for video identity. Rules assign severity. A human (electrician) owns the decision.**

Do not ask a single classifier to output `loose_connection` vs `overload` from a thermal blob. Those look alike. Output `component` + `ΔT` + `visible_defect` and let rules + a person interpret.

```
                    ┌──────────── RGB frame ────────────┐
                    │  YOLO-seg: components + defects   │
                    │  burn / exposed / broken / chew   │
                    └─────────────────┬─────────────────┘
                                      │
Inspection ──► split RGB / IR ──► fuse by homography or MSX overlay
                                      │
                    ┌──────────── IR frame ─────────────┐
                    │  YOLO-seg: components             │
                    │  radiometric T inside each mask   │
                    │  ΔT vs siblings and vs ambient    │
                    │  NETA-style severity              │
                    └─────────────────┬─────────────────┘
                                      │
              optional video: ByteTrack on component boxes
                                      │
                          report: photo pair + ID + ΔT + flags
```

## Modality

| Stream | Sensor | Model job |
|--------|--------|-----------|
| RGB | Phone / USB | Find board geometry and **visible** defects |
| Thermal | Radiometric IR | Find components; measure heat |
| Alignment | FLIR MSX or chessboard / manual crop | Same object in both images |
| Video | Either stream | ByteTrack IDs while panning |

If we only have one cheap IR camera that outputs a fused MSX JPEG **without** radiometric data, we can still train a hotspot detector, but we **cannot** implement NETA ΔT. That is a prototype, not a measurement tool.

## Models

**Start:** Ultralytics YOLO (v8 or v11) `*-seg` small/medium.

| Head | Classes (v1 — keep small) |
|------|---------------------------|
| RGB components | `board`, `breaker`, `busbar`, `cable`, `outlet`, `lug` |
| RGB defects | `burn_mark`, `exposed_conductor`, `broken_cable`, `melted_plastic` |
| IR components | same component names (transfer RGB names where possible) |

v2 additions only after data exists: `thermal_hotspot` as an extra IR class (optional), `chew`, `double_tap`, `scorch_bus`.

**Why YOLO not a classifier:** panels contain many objects; we need *where*.  
**Why seg not just detect:** temperature should be averaged inside the breaker body, not a loose box that includes adjacent poles.  
**Why not ViT hybrid first:** extra complexity; papers that bolt ViT onto YOLO still need the same labels.

Pretrain: COCO → optional public IR/cable sets → **our** residential set.

## Physics layer (IR)

For each IR mask of class `breaker` / `lug` / `cable`:

1. Extract radiometric temperature array (FLIR SDK / `flirimageextractor` / camera API).
2. `T_max`, `T_mean` in mask. Ignore the 1% hottest pixels if they are likely reflection spikes (optional).
3. `ΔT_sib = T − median(siblings of same class in frame)`.
4. `ΔT_amb = T − T_ambient` (spot on enclosure or camera ambient).
5. Map through NETA 100.18 bands (see REPORT).
6. Require documented load ≥ 40% for “clear” findings; otherwise tag `low_load_unreliable`.

This layer is deterministic. It should be unit-tested with synthetic temperature maps.

## ByteTrack

Use only on video:

```text
model.track(source=video, tracker="bytetrack.yaml", persist=True)
```

Purpose:

- Same breaker keeps ID `3` across frames.
- Build a **dwell map of anomalies on the panel**, not a crowd heatmap.
- Deduplicate alerts (don’t fire 30 times for one hot lug).

Skip ByteTrack for stills and for the first demo.

Wrong use: Ultralytics `solutions.Heatmap` / Supervision movement heatmaps. Those encode **camera motion and object traffic**, not watts.

## Fusion and UX

A finding is:

- RGB defect confidence ≥ threshold, **or**
- IR severity ≥ “investigate”, **or**
- both (highest trust)

Always show **visible photo + thermal photo + ΔT + load note**. That matches NFPA 70B reporting spirit (thermal + visible, document ΔT).

Never auto-print “safe.” Missing heat ≠ healthy wiring.

## Runtime (later)

| Target | Model |
|--------|--------|
| Laptop demo | YOLO11s-seg, FP16 GPU |
| Phone + FLIR One | YOLO11n, or server-side inference |
| Edge (Jetson) | YOLO11n INT8 — papers already run IR YOLO on Orin NX / Nano |

## Training recipe (when we have labels)

- Ultralytics default augment is fine; add palette jitter if training on false-color IR.
- Prefer **raw temperature as a 1-channel input** (normalized) over palette RGB if we control the camera.
- Imbalance: oversample fault images; class weights; don’t only train on hotspots.
- Split **by site / board**, not random images, or leakage will fake high mAP.
- Metrics: mAP@0.5 for boxes/masks; plus **ΔT MAE** on a radiometric holdout; plus false-clear rate (the dangerous error).

## Out of scope for v1

- In-wall tomography
- Full EICR automation
- UAV HV inspection (data exists; not the stated home-circuit goal)

Partial discharge (TEV / ultrasonic / UHF) and ICT/X-ray are **out of home-DB v1** but **in scope later** for industrial switchgear and factory PCB repair. See [`TWO_SCALE_PLAN.md`](TWO_SCALE_PLAN.md).

## Two-scale routing (WP2+)

One photo does not mean one specialist model. A **locator** YOLO should first say whether the frame is a **small PCB/chip** or a **large switch/DB**, then:

- **PCB/chip:** AOI defect YOLO + optional golden-board difference + solder head; electrical *where* via continuity/TDR/ICT on the same ID.
- **Switch/DB:** RGB damage + radiometric ΔT on the named pole/lug; clamp for load; TDR only for outgoing cables; PD sensors only for MV cubicles.

Do not train one 20-class soup that mixes `mouse_bite` with `MCCB_hotspot`. Separate heads, shared tracker ID. Do **not** add a `complete` / healthy class — box problems only ([`DEFECT_BOX_REPORT.md`](DEFECT_BOX_REPORT.md)).

RGB infer: run YOLO on the original frame **and** on a mild CLAHE/sharpen/tophat copy, then keep original boxes ([`ENHANCE_REPORT.md`](ENHANCE_REPORT.md)). Do not replace the photo with the processed one.

Full plan: [`TWO_SCALE_PLAN.md`](TWO_SCALE_PLAN.md).

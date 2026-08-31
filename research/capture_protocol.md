# Capture protocol (for when we collect our own data)

This is the critical path. Public datasets will not cover residential boards.

**Do not open live panels unless a qualified electrician is doing it under their safety rules.** Arc flash and shock are real. This protocol assumes that person is present.

## Goal

Build a **paired RGB + radiometric IR** set of:

1. Local distribution boards / consumer units (normal, under typical load).
2. Accessible junctions, sockets, and exposed cable runs.
3. Known or staged faults (loose lug, extra load on one circuit) so the minority class exists.

Target for a first model: **300–800 paired frames** across ≥ 5 different boards, plus video clips of pans for ByteTrack later.

## Hardware

| Item | Minimum | Better |
|------|---------|--------|
| Thermal | FLIR One Pro (radiometric JPEG, 160×120) | FLIR C5 / Exx, or Lepton 3.5 with SDK |
| RGB | Same phone (MSX) or a second 1080p camera | Dedicated RGB aligned to IR |
| Load | Clamp meter on feeder / circuits | Logged current per circuit |
| Notes | Paper / phone log | Same filename stem for RGB, IR, JSON meta |

Reject cameras that only save a pretty false-color PNG with no temperature.

## Conditions

- Equipment **energized and loaded**. Aim for ≥ 40% of typical circuit load (NFPA 70B practice). A Sunday-empty house will hide faults.
- Capture peak-use if possible (evening AC / cooking).
- Indoor ambient noted (°C). Avoid sun on the board and blowing AC directly on the copper.
- Shiny metal: IR lies. Prefer comparing identical breakers, not absolute °C. Optional: high-emissivity tape dots on lugs in a **lab** setting only.

## Shot list (per board)

1. Closed dead-front, RGB + IR (what a homeowner sees).
2. Open dead-front (electrician), whole-board RGB + IR.
3. Close-ups: incoming lugs, neutral bar, each bank of breakers, any obvious discoloration.
4. Repeat at a second load level if possible.
5. 10–20 s pan video (RGB and/or IR) for later tracking tests.

## Metadata JSON (one per capture)

```json
{
  "id": "board03_shot07",
  "site": "anonymized",
  "board_type": "single_phase_db",
  "cover": "open",
  "load_note": "evening, AC on, ~estimated 45%",
  "ambient_c": 32,
  "camera": "flir_one_pro",
  "palette": "ironbow",
  "radiometric": true,
  "labels_later": true
}
```

No homeowner names, addresses, or meter numbers in the repo.

## Labeling (after capture)

Tool: Roboflow or CVAT. Export YOLO-seg.

**RGB:** components (locator) + **visible defects only** (`open` / `short` / `damage` — see `data/own/README.md`). Do not label healthy boards as `complete`.  
**IR:** components (same names). Hotspot evidence is ΔT on that component, not a “safe” class. Do **not** spend weeks labeling `overload` vs `loose` on IR. Optionally tag `known_fault=true` at image level when the electrician confirms a problem.

Split **by board ID** into train/val/test (e.g. 3 boards train, 1 val, 1 test).

## Staged faults (lab / training board only)

On a dedicated dead training board, not a customer’s house:

- Slightly loosened lug under modest load (electrician).
- Extra load on one branch vs neighbors.
- Sample of burned / chewed cable (already discarded pieces) for RGB close-ups.

Never stage faults that create fire risk. If in doubt, skip.

## Home RGB wires (red / blue) — 2026-08-27

Public data does not cover this. Use the synthetic bootstrap (`python -m src.build_home_wires`), then replace it with real photos.

**Shot list (RGB only, cover off only with an electrician):**

1. Open consumer unit: whole board, then close-ups of incoming tails (red/brown live, blue/black neutral, earth).
2. Junction box / ceiling rose with visible red + blue.
3. Socket back (if already open for other work): line / neutral / earth.
4. Any visible damage: burnt, nicked insulation, exposed copper.

Label with the class table in `data/own/README.md`. Filename stem shared across image + `.txt`.

This RGB color model **does not** see in-wall faults and **does not** measure heat. Keep the thermal protocol above for the product path.

## Ethics and safety

- Screening dataset, not a medical/legal inspection archive.
- Do not publish identifiable interiors without consent.
- Do not claim the dataset “proves” a board is safe.

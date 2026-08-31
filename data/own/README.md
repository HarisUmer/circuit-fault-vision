# Own labeled photos

Put real JPEGs here (gitignored except this README). Public PCB/cable sets are not Pakistani consumer units.

```
data/own/
  images/
  labels/     YOLO txt, same stem as image
```

One line per box: `class_id x_center y_center width height` (all 0–1).

## Product classes (new labels)

**Problems only.** See `research/DEFECT_BOX_REPORT.md`. Do **not** box a healthy board as `complete`. No boxes means “nothing visible in this photo,” not “safe.”

| id | name | meaning |
|----|------|---------|
| 0 | open | Gap, broken trace or disconnected visible conductor |
| 1 | short | Two conductors touching / solder bridge |
| 2 | damage | Burn, chew, mouse-bite, missing copper, broken insulation |

Chip solder (`solder_defect`, `part_misaligned`) and panel `hotspot` (IR) are later heads — do not mix those ids into this table until WP D2/D7.

## Legacy (do not add more of these)

- Synthetic color ids (`red_live`, `blue_neutral`, …) — only `--preset home_wires`.
- WP1c four-class (`complete` / `incomplete` / `wires_touching` / `damage`) — bootstrap only. Map `incomplete` → `open`, `wires_touching` → `short` if you reuse those labels.

Live panels: electrician present. Never print “safe.”

# Investor baseline (WP1)

**What this proves:** a small YOLO detector can be trained on public cable photos and find visible damage. The toolchain works.

**What this does not prove:** home consumer-unit screening, in-wall faults, or radiometric thermal ΔT. That still needs our own labeled boards and a radiometric camera.

## Dataset

| | |
|--|--|
| Name | Roboflow 100 **cable-damage** |
| Source | [Hugging Face LibreYOLO/cable-damage](https://huggingface.co/datasets/LibreYOLO/cable-damage) · [Roboflow Universe](https://universe.roboflow.com/roboflow-100/cable-damage) |
| License | CC BY 4.0 (Roboflow 100) |
| Images | 1,318 (train 919 / val 265 / test 134) |
| Classes | `break`, `thunderbolt` |
| Modality | RGB close-ups of damaged conductors (utility / outdoor). Not thermal. Not residential boards. |

FIRC 6-class thermal (loose connection, overload, breaker, …) is the closer product ontology but is **not freely downloadable**. Using it would over-claim. Cable-damage is the honest public proof.

## Model

YOLOv8n (nano), COCO-pretrained, 320 px, 15 epochs, CPU (`torch 2.2.2+cpu`). Weights: `models/investor_proof.pt`.

Reproduce:

```
python -m src.download_cable_damage
python -m src.train
python -m src.infer
```

## Held-out test metrics

Filled by `src/train.py` → `results/investor_proof/metrics.json` after the run. Do **not** quote literature mAP (Tao 98.7%, Chen 99% simulation) as this model's score.

## How to show investors

1. Open 4–8 annotated shots in `results/investor_proof/pred_*.jpg` (boxes on `break` / `thunderbolt`).
2. State the metric from `metrics.json` (mAP50 on the **test** split).
3. Say the next paid step is **own paired RGB + radiometric IR of local boards**, with an electrician, then NETA ΔT rules.

## Honest limits (say these out loud)

- Domain gap: overhead / lab cables ≠ Pakistani residential DBs.
- RGB cannot see hidden joints. Thermal under load is the other half of the product.
- Never print “safe.” Screening aid only.

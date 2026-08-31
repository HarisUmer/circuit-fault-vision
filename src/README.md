# src/

Three investor baselines plus a defect-only home mix. Same scripts, `--preset` switches the dataset.

**Outdoor metal cables** (trained): `break` / `thunderbolt` on RF100 cable-damage.

**Home circuits** (red / blue PVC): synthetic IEC + legacy colors. Not real photos.

**Circuit faults** (real photos): `complete` / `incomplete` / `wires_touching` / `damage` from PKU PCB + DeepPCB + cable-damage.

```
python -m src.download_cable_damage
python -m src.train --preset cable_damage
python -m src.infer --preset cable_damage

python -m src.build_home_wires
python -m src.train --preset home_wires --device auto
python -m src.infer --preset home_wires

python -m src.download_circuit_faults
python -m src.train --preset circuit_faults --device auto
python -m src.infer --preset circuit_faults --more --hide-complete

# Defect-only mix (no complete): PCB-IND + stripped-wire + indoor-socket backgrounds.
# Fine-tunes models/circuit_faults.pt. Still not a Pakistani consumer-unit model.
python -m src.download_home_data
python -m src.build_home_faults
python -m src.train --preset home_faults --epochs 15 --device cpu
python -m src.infer --preset home_faults --hide-complete

python -m src.download_pcb_ind
python -m src.eval_enhance
python -m src.infer --preset circuit_faults --more --hide-complete
```

GPU later (this box is CPU-only):

```
python -m src.train --preset circuit_faults --device 0 --epochs 40 --imgsz 640 --batch 16
```

`--device auto` uses CUDA if `torch.cuda.is_available()`, else CPU (320 px, batch 8, amp off).

| Script | Job |
|--------|-----|
| `download_cable_damage.py` | Hugging Face cable-damage (CC BY 4.0) |
| `build_home_wires.py` | Generate red/blue/brown/black/earth YOLO set (drawings) |
| `download_circuit_faults.py` | PKU PCB + DeepPCB photos, merge 4-class labels |
| `download_pcb_ind.py` | PCB-IND real AOI patches (Zenodo, CC BY 4.0) |
| `build_circuit_faults.py` | Remap existing downloads without re-fetching |
| `train.py` | YOLOv8n; `--preset cable_damage` / `home_wires` / `circuit_faults`. Circuit-faults adds CLAHE/sharpen train aug + small-object hyps |
| `infer.py` | Annotated proof shots. `--more` = enhance + dual NMS + tiles |
| `preprocess.py` | CLAHE, unsharp, tophat, auto-gamma |
| `more_detect.py` | Dual-pass / tiled / hflip merge |
| `eval_enhance.py` | Test-set comparison vs ground truth |
| `trainer.py` | DefectTrainer (random enhance on train batches) |
| `device.py` | CPU vs GPU defaults |

Real board photos: `data/own/README.md`. Later: ByteTrack, radiometric ΔT.

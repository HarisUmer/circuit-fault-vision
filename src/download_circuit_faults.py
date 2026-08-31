"""Download real PCB photos (PKU-Market-PCB) and DeepPCB YOLO images, then merge labels."""
from __future__ import annotations

import socket
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download, snapshot_download

from src.build_circuit_faults import DEEPPCB_RAW, PKU_RAW, build

ROOT = Path(__file__).resolve().parents[1]
DEEPPCB_URLS = (
    "http://www.pami.sjtu.edu.cn/Upload/Files/2018-12-18-02-58-58-687495.zip",
)


def _download_pku() -> Path:
    PKU_RAW.mkdir(parents=True, exist_ok=True)
    n_jpg = len(list(PKU_RAW.rglob("*.jpg"))) if PKU_RAW.exists() else 0
    if n_jpg >= 600:
        print("PKU-Market-PCB already present:", n_jpg, "jpg")
        return PKU_RAW
    path = snapshot_download(
        repo_id="RobotHuman/PCB_defect",
        repo_type="dataset",
        local_dir=str(PKU_RAW),
    )
    print("PKU-Market-PCB ->", path)
    return Path(path)


def _download_deeppcb_yolo() -> Path | None:
    DEEPPCB_RAW.mkdir(parents=True, exist_ok=True)
    processed_imgs = DEEPPCB_RAW / "processed"
    n_test = len(list(processed_imgs.rglob("*_test.jpg"))) if processed_imgs.exists() else 0
    if n_test >= 1000:
        print("DeepPCB YOLO images already present:", n_test)
        return DEEPPCB_RAW
    zip_path = Path(
        hf_hub_download(
            repo_id="thangkt/PCB-Prune-YOLO-DeepPCB",
            repo_type="dataset",
            filename="deeppcb_processed.zip",
            local_dir=str(DEEPPCB_RAW),
        )
    )
    out = DEEPPCB_RAW / "processed"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    print("DeepPCB YOLO extracted ->", out)
    return DEEPPCB_RAW


def _try_deeppcb_templates() -> Path | None:
    """Optional original zip (includes defect-free *_temp.jpg). Often blocked outside CN."""
    if any(DEEPPCB_RAW.rglob("*_temp.jpg")):
        print("DeepPCB templates already present")
        return DEEPPCB_RAW
    zip_path = DEEPPCB_RAW / "deeppcb_official.zip"
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(20)
    for url in DEEPPCB_URLS:
        try:
            print("Trying DeepPCB templates", url)
            urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(DEEPPCB_RAW)
            print("DeepPCB templates extracted ->", DEEPPCB_RAW)
            return DEEPPCB_RAW
        except Exception as exc:  # noqa: BLE001
            print("DeepPCB templates skip:", type(exc).__name__, exc)
        finally:
            socket.setdefaulttimeout(old_timeout)
    print("No *_temp.jpg templates. `complete` uses defect-free crops from PKU boards.")
    return None


def main() -> None:
    _download_pku()
    _download_deeppcb_yolo()
    _try_deeppcb_templates()
    cable = ROOT / "data" / "public" / "cable-damage" / "data.yaml"
    if not cable.exists():
        print("cable-damage missing; run python -m src.download_cable_damage for real wire photos")
    dest = build()
    print("Merged dataset:", dest)
    print("Train on GPU:")
    print("  python -m src.train --preset circuit_faults --device 0 --epochs 40 --imgsz 640")


if __name__ == "__main__":
    main()

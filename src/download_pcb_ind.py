"""Download PCB-IND: real industrial AOI PCB defect photos (CC BY 4.0, Zenodo)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public" / "pcb-ind"
ZIP_URL = "https://zenodo.org/records/19723114/files/PCB-IND_v4.zip?download=1"


def main() -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    yaml_hit = list(DEST.rglob("data.yaml")) + list(DEST.rglob("*.yaml"))
    n_jpg = len(list(DEST.rglob("*.jpg"))) + len(list(DEST.rglob("*.png")))
    if n_jpg >= 4000:
        print("PCB-IND already present:", n_jpg, "images")
        return DEST
    zip_path = DEST / "PCB-IND_v4.zip"
    print("Downloading PCB-IND v4 (~101 MB) from Zenodo…")
    urlretrieve(ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DEST)
    print("Extracted ->", DEST)
    print("Map later: open -> incomplete, short -> wires_touching, rest -> damage")
    return DEST


if __name__ == "__main__":
    main()

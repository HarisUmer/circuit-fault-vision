"""Download extra HOME-adjacent public image sets (not 13 GB auto-panel dumps)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public"

STRIPPED_URL = "https://zenodo.org/records/16686806/files/Insulated_wire_dataset.zip?download=1"
STRIPPED_DEST = PUBLIC / "stripped-wire"

# Indoor home sockets / switches / power strips (locator, not defects). CC BY 4.0.
SOCKETS_URL = (
    "https://zenodo.org/records/18835199/files/combined_datasets_yolo_annotations.zip?download=1"
)
SOCKETS_DEST = PUBLIC / "indoor-sockets"

MENDELEY_CANDIDATES = (
    "https://data.mendeley.com/public-api/datasets/g6rbmc2ggc/files/zip?download=1",
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/g6rbmc2ggc-1.zip",
)
MENDELEY_DEST = PUBLIC / "aviation-wiring"

# Public-domain Roboflow Universe set (home sockets / wires). Needs a key if gated.
ROBOFLOW_HAZARD = PUBLIC / "home-hazards"


def _get(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "circuit_fault_vision/1.0"})
    with urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)


def download_stripped_wire() -> Path:
    STRIPPED_DEST.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in STRIPPED_DEST.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if n >= 20:
        print("stripped-wire already present:", n, "images")
        return STRIPPED_DEST
    zip_path = STRIPPED_DEST / "Insulated_wire_dataset.zip"
    print("Downloading Stripped Wire Dataset (~18 MB, Zenodo 16686806)...")
    urlretrieve(STRIPPED_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(STRIPPED_DEST)
    print("Extracted ->", STRIPPED_DEST)
    return STRIPPED_DEST


def try_mendeley_wiring() -> Path | None:
    MENDELEY_DEST.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in MENDELEY_DEST.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if n >= 20:
        print("aviation-wiring already present:", n)
        return MENDELEY_DEST
    for url in MENDELEY_CANDIDATES:
        zip_path = MENDELEY_DEST / "g6rbmc2ggc.zip"
        try:
            print("Trying", url[:80], "...")
            _get(url, zip_path, timeout=60)
            if zip_path.stat().st_size < 10_000:
                print("  too small", zip_path.stat().st_size)
                continue
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(MENDELEY_DEST)
            print("Extracted aviation wiring ->", MENDELEY_DEST)
            return MENDELEY_DEST
        except Exception as exc:  # noqa: BLE001
            print("  failed:", type(exc).__name__, exc)
    print("Mendeley aviation-wiring not downloaded (gated or URL changed).")
    return None


def download_indoor_sockets() -> Path:
    """~3.5k indoor power-socket / switch / strip photos. Use as locator or empty-label backgrounds."""
    SOCKETS_DEST.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in SOCKETS_DEST.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if n >= 100:
        print("indoor-sockets already present:", n, "images")
        return SOCKETS_DEST
    zip_path = SOCKETS_DEST / "combined_datasets_yolo_annotations.zip"
    print("Downloading indoor sockets / switches (~180 MB, Zenodo 18835199)...")
    urlretrieve(SOCKETS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(SOCKETS_DEST)
    n = sum(1 for _ in SOCKETS_DEST.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    print("Extracted indoor-sockets ->", SOCKETS_DEST, "images", n)
    return SOCKETS_DEST


def try_roboflow_home_hazards() -> Path | None:
    """Home sockets: burned / damaged wire / open copper / overloaded. Usually needs Roboflow key."""
    import os

    key = os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_KEY")
    if not key:
        print("No ROBOFLOW_API_KEY — cannot export HazardDetector / electrical-hazards (home sockets).")
        print("That is the best public HOME set (~6k images). Sign up at universe.roboflow.com and set the key.")
        return None
    # workspace/project/version — public domain electrical-hazards
    url = f"https://universe.roboflow.com/ds/electrical-hazards?api_key={key}"
    print("Roboflow export URL shape is version-specific; skip auto without a known version id.")
    return None


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    download_stripped_wire()
    download_indoor_sockets()
    try_mendeley_wiring()
    try_roboflow_home_hazards()
    note = {
        "fetched": [
            "stripped-wire (Zenodo 16686806)",
            "indoor-sockets (Zenodo 18835199, CC BY 4.0, locator)",
        ],
        "already_local": ["pcb-ind", "circuit-faults", "cable-damage"],
        "gated_best_home_set": {
            "name": "HazardDetector / electrical-hazards",
            "url": "https://universe.roboflow.com/electrical-hazard-zcack/electrical-hazards",
            "classes": ["burned socket", "damage wire", "open copper", "overloaded socket"],
            "n_images": "~6100-6500",
            "license": "Public Domain (Roboflow listing)",
            "need": "ROBOFLOW_API_KEY then export YOLOv8",
        },
        "skipped": [
            "sriom1/electrical-panels-dataset (13 GB auto YOLOE labels)",
            "energAI-fuses images.zip (3.4 GB fuse catalog, locator not defects)",
            "WireWise Zenodo (preprint .docx only, no images)",
        ],
    }
    (PUBLIC / "home_data_search.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print("Wrote", PUBLIC / "home_data_search.json")


if __name__ == "__main__":
    main()

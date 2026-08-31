"""Download the public RF100 cable-damage dataset (YOLO format)."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "public" / "cable-damage"
REPOS = ("LibreYOLO/cable-damage", "Libre-YOLO/cable-damage")


def main() -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for repo in REPOS:
        try:
            path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(DEST))
            print(f"Downloaded {repo} -> {path}")
            return Path(path)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"Failed {repo}: {exc}")
    raise RuntimeError(f"Could not download cable-damage dataset: {last_err}")


if __name__ == "__main__":
    main()

"""YOLO data.yaml helpers shared by train/infer."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_yaml(dataset_dir: Path) -> Path:
    for name in ("data.yaml", "data.yml"):
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No data.yaml in {dataset_dir}")


def rewrite_paths(yaml_path: Path, out_name: str) -> Path:
    """Make train/val/test paths absolute so Ultralytics finds them from any cwd."""
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    base = yaml_path.parent.resolve()

    def abs_split(value: str | None) -> str | None:
        if not value:
            return value
        p = Path(str(value))
        if not p.is_absolute():
            p = (base / p).resolve()
        return str(p)

    for key in ("path", "train", "val", "valid", "test"):
        if key in cfg and cfg[key]:
            cfg[key] = abs_split(cfg[key])
    if "val" not in cfg and "valid" in cfg:
        cfg["val"] = cfg["valid"]

    out = ROOT / "data" / "public" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def class_names(cfg: dict) -> list[str]:
    names = cfg.get("names") or {}
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def count_split_images(dataset_dir: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in ("train", "valid", "val", "test"):
        img = dataset_dir / split / "images"
        if img.is_dir():
            n = sum(1 for p in img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            key = "val" if split in {"valid", "val"} else split
            out[key] = n
    return out

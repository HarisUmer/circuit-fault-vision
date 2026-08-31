"""Prepare the home-circuit wire dataset (generate bootstrap set).

There is no freely downloadable labeled set of Pakistani/UK residential red/blue
T&E. The 13 GB HF 'electrical-panels' dump is noisy auto-labels, not wire colors.
This script builds the synthetic YOLO set used by --preset home_wires.
"""
from __future__ import annotations

from src.build_home_wires import DEST, main as build_main


def main() -> None:
    build_main()
    print("Train on GPU later:")
    print("  python -m src.train --preset home_wires --device 0")
    print("Or auto (CUDA if present, else CPU):")
    print("  python -m src.train --preset home_wires --device auto")
    print("Dataset:", DEST)


if __name__ == "__main__":
    main()

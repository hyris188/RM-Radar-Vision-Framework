#!/usr/bin/env python3
"""Generic Ultralytics YOLO training entry point driven by a YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = config.pop("model")
    data = Path(config.pop("data"))
    if not data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data}. Create your private dataset first."
        )
    if args.resume:
        config["resume"] = True
    model = YOLO(model_name)
    result = model.train(data=str(data), **config)
    print(f"Training output: {result.save_dir}")


if __name__ == "__main__":
    main()

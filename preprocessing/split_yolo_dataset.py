#!/usr/bin/env python3
"""Create a deterministic, duplicate-safe YOLO train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")

    groups = defaultdict(list)
    for image in sorted(args.images.rglob("*")):
        if image.suffix.lower() in IMAGE_SUFFIXES:
            label = args.labels / image.relative_to(args.images).with_suffix(".txt")
            if label.is_file():
                groups[digest(image)].append((image, label))
    unique_groups = list(groups.values())
    random.Random(args.seed).shuffle(unique_groups)
    train_target = round(sum(len(group) for group in unique_groups) * args.train_ratio)
    counts = {"train": 0, "val": 0}
    for group in unique_groups:
        split = "train" if counts["train"] < train_target else "val"
        for image, label in group:
            relative = image.relative_to(args.images)
            link_or_copy(image, args.output / "images" / split / relative)
            link_or_copy(label, args.output / "labels" / split / relative.with_suffix(".txt"))
            counts[split] += 1
    print(f"train={counts['train']} val={counts['val']} duplicate_groups={len(unique_groups)}")


if __name__ == "__main__":
    main()

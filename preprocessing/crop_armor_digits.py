#!/usr/bin/env python3
"""Crop armor regions from YOLO labels into digit-classification folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "B1", "B2", "B3", "B4", "B0", "BS",
            "R1", "R2", "R3", "R4", "R0", "RS",
        ],
    )
    parser.add_argument("--padding", type=float, default=0.05)
    args = parser.parse_args()

    written = skipped = 0
    for image_path in sorted(args.images.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(args.images)
        label_path = args.labels / relative.with_suffix(".txt")
        if not label_path.is_file():
            skipped += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split()
            if len(fields) != 5:
                skipped += 1
                continue
            class_id = int(fields[0])
            if not 0 <= class_id < len(args.classes):
                skipped += 1
                continue
            cx, cy, box_width, box_height = map(float, fields[1:])
            pad_x = box_width * args.padding
            pad_y = box_height * args.padding
            x1 = max(0, round((cx - box_width / 2 - pad_x) * width))
            y1 = max(0, round((cy - box_height / 2 - pad_y) * height))
            x2 = min(width, round((cx + box_width / 2 + pad_x) * width))
            y2 = min(height, round((cy + box_height / 2 + pad_y) * height))
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            target_dir = args.output / args.classes[class_id]
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{relative.stem}_{index:02d}.jpg"
            if not cv2.imwrite(str(target), image[y1:y2, x1:x2]):
                raise RuntimeError(f"Cannot write crop: {target}")
            written += 1
    print(f"written={written} skipped={skipped} output={args.output.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create offline YOLO detection augmentations without changing validation data."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(path: Path):
    boxes, class_ids = [], []
    if not path.is_file():
        return boxes, class_ids
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        box = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in box):
            raise ValueError(f"{path}:{line_number}: coordinates must be normalized")
        class_ids.append(class_id)
        boxes.append(box)
    return boxes, class_ids


def transform_pipeline():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                rotate=(-5, 5),
                shear=(-2, 2),
                p=0.7,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.2),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.15,
                        hue=0.03,
                        p=1.0,
                    ),
                    A.RandomBrightnessContrast(p=1.0),
                ],
                p=0.6,
            ),
            A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3, 5))], p=0.2),
            A.GaussNoise(p=0.15),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.25,
            clip=True,
        ),
    )


def write_labels(path: Path, boxes, class_ids) -> None:
    lines = [
        f"{int(class_id)} " + " ".join(f"{float(value):.6f}" for value in box)
        for box, class_id in zip(boxes, class_ids)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    image_dir = args.dataset / "images" / "train"
    label_dir = args.dataset / "labels" / "train"
    output_images = args.output / "images" / "train"
    output_labels = args.output / "labels" / "train"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    pipeline = transform_pipeline()

    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    written = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue
        relative = image_path.relative_to(image_dir)
        label_path = label_dir / relative.with_suffix(".txt")
        boxes, class_ids = read_labels(label_path)
        for index in range(args.copies):
            random.seed(args.seed + written)
            result = pipeline(image=image, bboxes=boxes, class_ids=class_ids)
            stem = f"{relative.stem}_aug{index:02d}"
            target_image = output_images / relative.parent / f"{stem}.jpg"
            target_label = output_labels / relative.parent / f"{stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target_image), result["image"])
            write_labels(target_label, result["bboxes"], result["class_ids"])
            written += 1
    print(f"generated={written} output={args.output.resolve()}")


if __name__ == "__main__":
    main()

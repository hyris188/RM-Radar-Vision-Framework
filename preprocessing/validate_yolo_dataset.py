#!/usr/bin/env python3
"""Validate YOLO labels and detect exact train/validation image leakage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_label(path: Path, class_count: int, pose: bool) -> list[str]:
    errors = []
    expected = 8 if pose else 5
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != expected:
            errors.append(f"{path}:{line_number}: expected {expected} fields, got {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            values = [float(item) for item in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric field")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{path}:{line_number}: invalid class {class_id}")
        normalized = values[:4] + (values[4:6] if pose else [])
        if not all(0.0 <= value <= 1.0 for value in normalized):
            errors.append(f"{path}:{line_number}: normalized coordinate outside [0,1]")
        if pose and values[6] not in (0.0, 1.0, 2.0):
            errors.append(f"{path}:{line_number}: keypoint visibility must be 0, 1 or 2")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--pose", action="store_true", help="expect one x/y/visibility keypoint")
    args = parser.parse_args()
    errors = []
    hashes = {}
    counts = {}
    for split in ("train", "val"):
        image_root = args.dataset / "images" / split
        label_root = args.dataset / "labels" / split
        images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        counts[split] = len(images)
        for image in images:
            relative = image.relative_to(image_root)
            label = label_root / relative.with_suffix(".txt")
            if not label.is_file():
                errors.append(f"missing label: {label}")
                continue
            errors.extend(validate_label(label, args.classes, args.pose))
            value = digest(image)
            previous = hashes.get(value)
            if previous and previous[0] != split:
                errors.append(f"train/val duplicate: {previous[1]} == {image}")
            hashes[value] = (split, image)
    for error in errors[:100]:
        print(error)
    print(f"train={counts.get('train', 0)} val={counts.get('val', 0)} errors={len(errors)}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()

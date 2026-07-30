#!/usr/bin/env python3
"""Extract uniformly sampled frames for subsequent manual annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=15, help="keep every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prefix", default="frame")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    args.output.mkdir(parents=True, exist_ok=True)
    source_index = saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % max(1, args.step) == 0:
            path = args.output / f"{args.prefix}_{source_index:08d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Cannot write image: {path}")
            saved += 1
            if args.max_frames and saved >= args.max_frames:
                break
        source_index += 1
    capture.release()
    print(f"source_frames={source_index} saved={saved} output={args.output.resolve()}")


if __name__ == "__main__":
    main()

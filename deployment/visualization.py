"""OpenCV drawing helpers shared by the deployment pipelines."""

from __future__ import annotations

import cv2


BGR_GRAY = (160, 160, 160)
BGR_RED = (0, 0, 255)
BGR_BLUE = (255, 0, 0)


def get_armor_style(class_id: int, class_names) -> tuple[str, tuple[int, int, int]]:
    """Return a normalized armor label and an exact OpenCV BGR color."""
    if isinstance(class_names, dict):
        model_name = str(class_names.get(class_id, "armor"))
    else:
        model_name = str(class_names[class_id])
    normalized = model_name.lower()
    if "red" in normalized:
        return "red", BGR_RED
    if "blue" in normalized:
        return "blue", BGR_BLUE
    if "dead" in normalized:
        return "dead", BGR_GRAY
    return model_name, (0, 255, 255)


def validate_armor_classes(class_names) -> None:
    resolved = {
        get_armor_style(class_id, class_names)[0]
        for class_id in range(len(class_names))
    }
    expected = {"dead", "red", "blue"}
    if not expected.issubset(resolved):
        raise ValueError(
            f"Armor model classes must contain dead/red/blue, got: {class_names}"
        )


def put_label(frame, text, x, y, color, scale=0.55) -> None:
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
    )
    top = max(0, y - height - baseline - 6)
    cv2.rectangle(frame, (x, top), (x + width + 6, y), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 3, y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

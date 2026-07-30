#!/usr/bin/env python3
"""Run the trained three-stage recognition pipeline on a Hikrobot USB camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from PIL import Image
import torch
from ultralytics import YOLO

from model.digit_classifier.predictor import DigitClassifier
from utils.config import load_cfg_from_cfg_file
from deployment.visualization import (
    get_armor_style,
    put_label,
    validate_armor_classes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-config", default="config/device.yaml")
    parser.add_argument(
        "--car-weights",
        default="weights/car_detector.pt",
    )
    parser.add_argument(
        "--armor-weights",
        default="weights/armor_detector.pt",
    )
    parser.add_argument(
        "--digit-weights",
        default="weights/armor_digit_classifier.pth",
    )
    parser.add_argument("--car-conf", type=float, default=0.25)
    parser.add_argument("--armor-conf", type=float, default=0.20)
    parser.add_argument(
        "--car-imgsz",
        type=int,
        default=1280,
        help="Car detector inference size; 640 is recommended for real-time use",
    )
    parser.add_argument(
        "--armor-imgsz",
        type=int,
        default=192,
        help="Armor detector inference size",
    )
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Use FP16 detector inference on CUDA",
    )
    parser.add_argument("--camera-width", type=int, default=4024)
    parser.add_argument("--camera-height", type=int, default=3036)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument(
        "--pixel-format",
        choices=("BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8"),
        default="BayerRG8",
    )
    parser.add_argument("--exposure", type=float, default=None, help="Exposure time in us")
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument(
        "--process-width",
        type=int,
        default=1920,
        help="Resize before inference; 0 keeps the camera resolution",
    )
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--output", default="", help="Optional output MP4 path")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    return parser.parse_args()


def resize_to_width(image, width: int):
    if width <= 0 or image.shape[1] <= width:
        return image
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def find_hikrobot_usb_devices():
    devices = []
    for vendor_file in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            if vendor_file.read_text().strip().lower() != "2bdf":
                continue
            device_dir = vendor_file.parent
            product_file = device_dir / "idProduct"
            speed_file = device_dir / "speed"
            devices.append(
                {
                    "path": device_dir.name,
                    "product": product_file.read_text().strip() if product_file.exists() else "unknown",
                    "speed": float(speed_file.read_text().strip()) if speed_file.exists() else 0.0,
                }
            )
        except (OSError, ValueError):
            continue
    return devices


def annotate_frame(frame, car_model, armor_model, digit_model, args):
    height, width = frame.shape[:2]
    car_result = car_model.predict(
        frame,
        imgsz=args.car_imgsz,
        conf=args.car_conf,
        device=args.device,
        half=args.half,
        verbose=False,
    )[0]

    car_entries = []
    for box in car_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        car_entries.append((x1, y1, x2, y2, float(box.conf[0]), frame[y1:y2, x1:x2]))

    armor_results = []
    if car_entries:
        armor_results = armor_model.predict(
            [entry[5] for entry in car_entries],
            imgsz=args.armor_imgsz,
            conf=args.armor_conf,
            device=args.device,
            half=args.half,
            verbose=False,
        )

    digit_crops = []
    armor_entries = []
    for car_entry, result in zip(car_entries, armor_results):
        cx1, cy1, _, _, _, car_crop = car_entry
        for box in result.boxes:
            ax1, ay1, ax2, ay2 = map(int, box.xyxy[0].tolist())
            ax1, ay1 = max(0, ax1), max(0, ay1)
            ax2 = min(car_crop.shape[1], ax2)
            ay2 = min(car_crop.shape[0], ay2)
            if ax2 <= ax1 or ay2 <= ay1:
                continue
            class_id = int(box.cls[0])
            armor_entries.append(
                (cx1 + ax1, cy1 + ay1, cx1 + ax2, cy1 + ay2, class_id, float(box.conf[0]))
            )
            crop = car_crop[ay1:ay2, ax1:ax2]
            digit_crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    digit_names = []
    digit_probabilities = []
    if digit_crops:
        digit_names, digit_probabilities = digit_model.predict_batch(
            digit_crops, return_names=True
        )

    for x1, y1, x2, y2, confidence, _ in car_entries:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
        put_label(frame, f"car {confidence:.2f}", x1, y1, (0, 150, 0))

    for entry, digit, probabilities in zip(armor_entries, digit_names, digit_probabilities):
        x1, y1, x2, y2, class_id, armor_confidence = entry
        armor_name, color = get_armor_style(class_id, armor_model.names)
        digit_confidence = max(probabilities)
        label = (
            f"{armor_name}_{digit} "
            f"A:{armor_confidence:.2f} D:{digit_confidence:.2f}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        put_label(frame, label, x1, y1, color, scale=0.45)

    return len(car_entries), len(armor_entries)


def main() -> None:
    args = parse_args()
    required = (
        args.device_config,
        args.car_weights,
        args.armor_weights,
        args.digit_weights,
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    sdk_library = Path("/opt/MVS/lib/64/libMvCameraControl.so")
    if not sdk_library.is_file():
        raise RuntimeError(
            "Hikrobot MVS SDK is not installed. Install the bundled x86_64 .deb first."
        )

    usb_devices = find_hikrobot_usb_devices()
    if not usb_devices:
        raise SystemExit(
            "ERROR: Linux has not detected a Hikrobot USB camera (USB vendor 2bdf).\n"
            "Reconnect the camera and first verify: lsusb | grep 2bdf:0001"
        )
    print(f"Detected Hikrobot USB device(s): {usb_devices}")
    if all(device["speed"] < 5000 for device in usb_devices):
        print("WARNING: Camera is not on a 5 Gbit/s USB 3.x link; real-time FPS will be limited.")

    import rclpy
    from rclpy.signals import SignalHandlerOptions

    # Import only after checking the SDK so a missing installation gives a useful error.
    from driver.hik_camera.hik import SimpleHikCamera

    camera_config = load_cfg_from_cfg_file(args.device_config)
    camera_config.width = args.camera_width
    camera_config.height = args.camera_height
    camera_config.acquisition_rate = args.camera_fps
    camera_config.display_fps = False
    camera_config.pixel_format = args.pixel_format
    if args.exposure is not None:
        camera_config.exposure_time = args.exposure
    if args.gain is not None:
        camera_config.gain = args.gain

    car_model = YOLO(args.car_weights)
    armor_model = YOLO(args.armor_weights)
    validate_armor_classes(armor_model.names)
    digit_model = DigitClassifier("mobilenet", args.digit_weights)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    camera = None
    writer = None
    frame_count = 0
    fps_window_started = time.perf_counter()
    fps_window_frames = 0
    try:
        camera = SimpleHikCamera(camera_config)
        camera.register_group("realtime_detection")
        camera.start_streaming()
        last_frame_at = time.monotonic()

        while True:
            image_rgb, _ = camera.get_image_latest("realtime_detection", timeout=2.0)
            if image_rgb is None:
                print("Waiting for a camera frame...")
                if time.monotonic() - last_frame_at >= args.connect_timeout:
                    raise RuntimeError(
                        "Hikrobot USB device exists, but MVS did not deliver a frame "
                        f"within {args.connect_timeout:g} seconds."
                    )
                continue
            last_frame_at = time.monotonic()

            started = time.perf_counter()
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            frame = resize_to_width(frame, args.process_width)
            cars, armors = annotate_frame(
                frame, car_model, armor_model, digit_model, args
            )
            inference_fps = 1.0 / max(time.perf_counter() - started, 1e-6)
            cv2.putText(
                frame,
                f"Infer {inference_fps:.1f} FPS | Camera {camera.get_fps():.1f} FPS | "
                f"Cars {cars} | Armors {armors}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.output:
                if writer is None:
                    output = Path(args.output)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.camera_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {output}")
                writer.write(frame)

            display = resize_to_width(frame, args.display_width)
            cv2.imshow("Hikrobot Full Pipeline - press q to quit", display)
            frame_count += 1
            fps_window_frames += 1
            if fps_window_frames >= 30:
                elapsed = time.perf_counter() - fps_window_started
                print(
                    f"[PERF] pipeline={fps_window_frames / elapsed:.1f} FPS, "
                    f"last_infer={inference_fps:.1f} FPS, "
                    f"camera={camera.get_fps():.1f} FPS",
                    flush=True,
                )
                fps_window_started = time.perf_counter()
                fps_window_frames = 0
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if camera is not None:
            camera.close()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"Processed {frame_count} camera frames")
    if args.output:
        print(f"Saved visualization to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

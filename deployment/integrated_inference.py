#!/usr/bin/env python3
"""Run armor, drone and laser-module recognition on one Hikrobot stream."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from ultralytics import YOLO

from model.digit_classifier.predictor import DigitClassifier
from utils.config import load_cfg_from_cfg_file
from deployment.armor_pipeline import (
    annotate_frame,
    find_hikrobot_usb_devices,
    resize_to_width,
)
from deployment.visualization import put_label, validate_armor_classes


# Load the pip-installed CUDA/cuDNN libraries before Ultralytics creates the
# ONNX Runtime sessions for the two PFA models.
ort.preload_dlls(directory="")

BGR_DRONE = (0, 165, 255)
BGR_LASER = (255, 0, 255)
BGR_CENTER = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The repository intentionally contains no weights. Users must train their
    # own models and place them in this ignored local directory (or override
    # every path from the command line).
    model_dir = Path("weights")
    parser.add_argument("--device-config", default="config/device.yaml")
    parser.add_argument("--car-weights", default=str(model_dir / "car_detector_best.pt"))
    parser.add_argument(
        "--armor-weights", default=str(model_dir / "armor_detector_best.pt")
    )
    parser.add_argument(
        "--digit-weights", default=str(model_dir / "armor_digit_classifier_best.pth")
    )
    parser.add_argument(
        "--drone-weights", default=str(model_dir / "drone_detector_best.onnx")
    )
    parser.add_argument(
        "--laser-weights", default=str(model_dir / "laser_module_pose_best.onnx")
    )
    parser.add_argument("--car-conf", type=float, default=0.25)
    parser.add_argument("--armor-conf", type=float, default=0.20)
    parser.add_argument("--drone-conf", type=float, default=0.60)
    parser.add_argument("--laser-conf", type=float, default=0.50)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--car-imgsz", type=int, default=640)
    parser.add_argument("--armor-imgsz", type=int, default=192)
    parser.add_argument("--air-imgsz", type=int, default=640)
    parser.add_argument(
        "--air-interval",
        type=int,
        default=1,
        help="Run drone and laser models every N frames; 1 runs all models every frame",
    )
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument("--camera-width", type=int, default=4024)
    parser.add_argument("--camera-height", type=int, default=3036)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument(
        "--pixel-format",
        choices=("BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8"),
        default="BayerRG8",
    )
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--process-width", type=int, default=960)
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip model warmup (the first displayed frames will be slower)",
    )
    args = parser.parse_args()
    if args.air_interval < 1:
        parser.error("--air-interval must be at least 1")
    return args


def warmup_models(car_model, armor_model, digit_model, drone_model, laser_model, args) -> None:
    print("Warming up all five models on the GPU...", flush=True)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    car_model.predict(
        dummy,
        imgsz=args.car_imgsz,
        conf=args.car_conf,
        device=args.device,
        half=args.half,
        verbose=False,
    )
    armor_model.predict(
        dummy,
        imgsz=args.armor_imgsz,
        conf=args.armor_conf,
        device=args.device,
        half=args.half,
        verbose=False,
    )
    digit_model.predict(Image.fromarray(dummy), return_names=True)
    drone_model.predict(
        dummy,
        imgsz=args.air_imgsz,
        conf=args.drone_conf,
        device=args.device,
        half=args.half,
        verbose=False,
    )
    laser_model.predict(
        dummy,
        imgsz=args.air_imgsz,
        conf=args.laser_conf,
        device=args.device,
        half=args.half,
        verbose=False,
    )
    print("Model warmup complete.", flush=True)


def predict_air(frame, drone_model, laser_model, args):
    drone_result = drone_model.predict(
        frame,
        imgsz=args.air_imgsz,
        conf=args.drone_conf,
        max_det=1,
        device=args.device,
        half=args.half,
        verbose=False,
    )[0]
    laser_result = laser_model.predict(
        frame,
        imgsz=args.air_imgsz,
        conf=args.laser_conf,
        max_det=1,
        device=args.device,
        half=args.half,
        verbose=False,
    )[0]
    return drone_result, laser_result


def draw_air(frame, drone_result, laser_result, keypoint_conf: float) -> tuple[int, int]:
    height, width = frame.shape[:2]
    drone_count = 0
    laser_count = 0

    if drone_result is not None:
        for box in drone_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            confidence = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BGR_DRONE, 2)
            put_label(frame, f"drone {confidence:.2f}", x1, y1, BGR_DRONE)
            drone_count += 1

    keypoint_data = None
    if laser_result is not None and laser_result.keypoints is not None:
        keypoint_data = laser_result.keypoints.data

    if laser_result is not None:
        for index, box in enumerate(laser_result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            confidence = float(box.conf[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_confidence = 0.0
            if keypoint_data is not None and index < len(keypoint_data):
                point = keypoint_data[index, 0]
                center_confidence = float(point[2])
                if center_confidence >= keypoint_conf:
                    center_x = int(point[0])
                    center_y = int(point[1])

            cv2.rectangle(frame, (x1, y1), (x2, y2), BGR_LASER, 2)
            cv2.drawMarker(
                frame,
                (center_x, center_y),
                BGR_CENTER,
                cv2.MARKER_CROSS,
                18,
                2,
            )
            label = (
                f"laser_module {confidence:.2f} "
                f"center=({center_x},{center_y}) K:{center_confidence:.2f}"
            )
            put_label(frame, label, x1, y1, BGR_LASER, scale=0.42)
            laser_count += 1

    return drone_count, laser_count


def main() -> None:
    args = parse_args()
    required = (
        args.device_config,
        args.car_weights,
        args.armor_weights,
        args.digit_weights,
        args.drone_weights,
        args.laser_weights,
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    sdk_library = Path("/opt/MVS/lib/64/libMvCameraControl.so")
    if not sdk_library.is_file():
        raise RuntimeError("Hikrobot MVS SDK is not installed")
    usb_devices = find_hikrobot_usb_devices()
    if not usb_devices:
        raise SystemExit(
            "ERROR: Linux has not detected a Hikrobot USB camera (USB vendor 2bdf)."
        )
    print(f"Detected Hikrobot USB device(s): {usb_devices}")
    if all(device["speed"] < 5000 for device in usb_devices):
        print("WARNING: The camera is not using a 5 Gbit/s USB 3.x link.")

    car_model = YOLO(args.car_weights)
    armor_model = YOLO(args.armor_weights)
    validate_armor_classes(armor_model.names)
    digit_model = DigitClassifier("mobilenet", args.digit_weights)
    drone_model = YOLO(args.drone_weights, task="detect")
    laser_model = YOLO(args.laser_weights, task="pose")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if not args.no_warmup:
        warmup_models(
            car_model, armor_model, digit_model, drone_model, laser_model, args
        )

    import rclpy
    from rclpy.signals import SignalHandlerOptions
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

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    camera = None
    writer = None
    frame_count = 0
    cached_drone_result = None
    cached_laser_result = None
    fps_window_started = time.perf_counter()
    fps_window_frames = 0
    try:
        camera = SimpleHikCamera(camera_config)
        camera.register_group("integrated_detection")
        camera.start_streaming()
        last_frame_at = time.monotonic()

        while True:
            image_rgb, _ = camera.get_image_latest("integrated_detection", timeout=2.0)
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

            # Every model receives pixels without visualization overlays.
            if frame_count % args.air_interval == 0:
                cached_drone_result, cached_laser_result = predict_air(
                    frame, drone_model, laser_model, args
                )
            cars, armors = annotate_frame(
                frame, car_model, armor_model, digit_model, args
            )
            drones, lasers = draw_air(
                frame, cached_drone_result, cached_laser_result, args.keypoint_conf
            )

            inference_fps = 1.0 / max(time.perf_counter() - started, 1e-6)
            cv2.putText(
                frame,
                f"All {inference_fps:.1f} FPS | Camera {camera.get_fps():.1f} FPS | "
                f"Cars {cars} Armors {armors} Drones {drones} Lasers {lasers}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "GREEN car | RED/BLUE armor | ORANGE drone | MAGENTA laser | YELLOW center",
                (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
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
            cv2.imshow("Armor + Drone + Laser Module - press q to quit", display)
            frame_count += 1
            fps_window_frames += 1
            if fps_window_frames >= 30:
                elapsed = time.perf_counter() - fps_window_started
                print(
                    f"[PERF] all={fps_window_frames / elapsed:.1f} FPS, "
                    f"last={inference_fps:.1f} FPS, camera={camera.get_fps():.1f} FPS, "
                    f"cars={cars}, armors={armors}, drones={drones}, lasers={lasers}",
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

    print(f"Processed {frame_count} integrated camera frames")
    if args.output:
        print(f"Saved visualization to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

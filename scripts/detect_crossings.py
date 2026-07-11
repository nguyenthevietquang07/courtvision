"""Detect ball/net crossings in a video.

The script records a crossing event when the best ball box and best net box
overlap at or above an IoU threshold. It also records whether the ball center
is inside the net box expanded by a configurable padding, which is useful for
noisy sports footage where a tiny ball box may not overlap much.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect CourtVision ball/net crossing events")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights, usually runs/.../best.pt")
    parser.add_argument("--source", default="sample.mp4", help="Input video path")
    parser.add_argument("--out-dir", default="runs/crossings", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--iou-threshold", type=float, default=0.01, help="Box IoU needed to record a crossing")
    parser.add_argument("--net-padding", type=int, default=20, help="Pixels to expand the net box for center checks")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests")
    parser.add_argument("--save-video", action="store_true", help="Write annotated video")
    return parser.parse_args()


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_inside_expanded_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
    padding: int,
) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding)


def get_detections(result) -> list[Detection]:
    detections: list[Detection] = []
    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = str(names[class_id]).lower()
        confidence = float(box.conf[0].item())
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        detections.append(Detection(class_name=class_name, confidence=confidence, box=xyxy))

    return detections


def best_detection(detections: Iterable[Detection], class_name: str) -> Detection | None:
    candidates = [d for d in detections if d.class_name == class_name]
    return max(candidates, key=lambda d: d.confidence, default=None)


def draw_detection(frame, detection: Detection, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in detection.box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{detection.class_name} {detection.confidence:.2f}"
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Could not find input video: {source}")

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        video_path = out_dir / "annotated_crossings.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    frames_csv = out_dir / "frame_metrics.csv"
    events_csv = out_dir / "crossing_events.csv"
    events_json = out_dir / "crossing_events.json"

    events: list[dict] = []
    previous_crossing = False
    event_id = 0
    frame_index = 0

    with frames_csv.open("w", newline="", encoding="utf-8") as frame_file:
        frame_writer = csv.DictWriter(
            frame_file,
            fieldnames=[
                "frame",
                "time_seconds",
                "ball_confidence",
                "net_confidence",
                "iou",
                "ball_center_x",
                "ball_center_y",
                "net_center_x",
                "net_center_y",
                "center_inside_expanded_net",
                "crossing_active",
                "event_id",
            ],
        )
        frame_writer.writeheader()

        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detections = get_detections(result)
            ball = best_detection(detections, "ball")
            net = best_detection(detections, "net")

            iou = 0.0
            ball_center = ("", "")
            net_center = ("", "")
            center_inside_net = False
            crossing_active = False
            current_event_id = ""

            if ball is not None:
                ball_center = center(ball.box)
                draw_detection(frame, ball, (0, 255, 255))

            if net is not None:
                net_center = center(net.box)
                draw_detection(frame, net, (255, 0, 255))

            if ball is not None and net is not None:
                iou = box_iou(ball.box, net.box)
                center_inside_net = point_inside_expanded_box(ball_center, net.box, args.net_padding)
                crossing_active = iou >= args.iou_threshold or center_inside_net

                if crossing_active and not previous_crossing:
                    event_id += 1
                    current_event_id = event_id
                    event = {
                        "event_id": event_id,
                        "frame": frame_index,
                        "time_seconds": round(frame_index / fps, 3),
                        "iou": round(iou, 5),
                        "ball_confidence": round(ball.confidence, 4),
                        "net_confidence": round(net.confidence, 4),
                        "ball_box": [round(v, 2) for v in ball.box],
                        "net_box": [round(v, 2) for v in net.box],
                        "center_inside_expanded_net": center_inside_net,
                    }
                    events.append(event)

                if crossing_active:
                    cv2.putText(
                        frame,
                        "CROSSING",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6,
                        (0, 0, 255),
                        4,
                    )

            frame_writer.writerow(
                {
                    "frame": frame_index,
                    "time_seconds": round(frame_index / fps, 3),
                    "ball_confidence": round(ball.confidence, 4) if ball else "",
                    "net_confidence": round(net.confidence, 4) if net else "",
                    "iou": round(iou, 5),
                    "ball_center_x": round(ball_center[0], 2) if ball else "",
                    "ball_center_y": round(ball_center[1], 2) if ball else "",
                    "net_center_x": round(net_center[0], 2) if net else "",
                    "net_center_y": round(net_center[1], 2) if net else "",
                    "center_inside_expanded_net": center_inside_net,
                    "crossing_active": crossing_active,
                    "event_id": current_event_id,
                }
            )

            previous_crossing = crossing_active
            if writer is not None:
                writer.write(frame)
            frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    with events_csv.open("w", newline="", encoding="utf-8") as event_file:
        event_writer = csv.DictWriter(
            event_file,
            fieldnames=[
                "event_id",
                "frame",
                "time_seconds",
                "iou",
                "ball_confidence",
                "net_confidence",
                "ball_box",
                "net_box",
                "center_inside_expanded_net",
            ],
        )
        event_writer.writeheader()
        event_writer.writerows(events)

    events_json.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"Processed {frame_index} frames")
    print(f"Recorded {len(events)} crossing events")
    print(f"Frame metrics: {frames_csv}")
    print(f"Crossing events: {events_csv}")
    print(f"Crossing events JSON: {events_json}")
    if args.save_video:
        print(f"Annotated video: {out_dir / 'annotated_crossings.mp4'}")


if __name__ == "__main__":
    main()

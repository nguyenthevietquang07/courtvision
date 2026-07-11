"""Train a YOLO model for CourtVision ball/net detection.

This script is intentionally small so it can run locally or in Google Colab.
For serious training, use a GPU runtime and start from yolov8s/yolov8m or the
provided yolov8x.pt checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CourtVision ball/net detector")
    parser.add_argument("--data", default="data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base YOLO model or checkpoint")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None, help="Device, for example 0, cpu, or cuda")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics project directory")
    parser.add_argument("--name", default="courtvision_ball_net", help="Run name")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=7, help="Training seed")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers; use 0 on Windows issues")
    return parser.parse_args()


def resolved_data_config(data_path: Path, project_dir: Path, run_name: str) -> Path:
    """Write a YOLO data config with an absolute dataset root.

    Ultralytics may resolve relative paths against its global datasets
    directory. Writing an absolute `path` keeps local Windows runs and Colab
    runs pointed at the checked-out repository.
    """

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    raw_root = Path(data.get("path", data_path.parent))
    dataset_root = raw_root if raw_root.is_absolute() else (data_path.parent / raw_root)
    data["path"] = str(dataset_root.resolve())

    project_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = project_dir / f"{run_name}_resolved_data.yaml"
    resolved_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolved_path


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset config: {data_path}")

    project_dir = Path(args.project)
    train_data = resolved_data_config(data_path, project_dir, args.name)

    model = YOLO(args.model)
    results = model.train(
        data=str(train_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        plots=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_path = save_dir / "weights" / "best.pt"
    print(f"Training finished. Best weights: {best_path}")
    print(results)


if __name__ == "__main__":
    main()

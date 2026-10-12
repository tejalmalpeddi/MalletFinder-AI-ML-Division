"""Train YOLOv8n on the bottle/mallet dataset.

"""
import argparse
from pathlib import Path

import albumentations as A
import torch
import yaml
from ultralytics import YOLO

DATA = Path("Sampled-YD-Object-Detection-2").resolve()
RUNS = Path("runs").resolve()

COMMON = dict(model="yolov8n.pt", imgsz=640, batch=16, epochs=50, patience=20, seed=0)

CONFIGS = {
    "baseline": {},
    "tuned": dict(
        hsv_h=0.01,
        hsv_v=0.5,
        scale=0.3,
        augmentations=[
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
            A.ImageCompression(quality_range=(40, 90), p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.1),
        ],
    ),
}


def write_data_yaml():
    names = yaml.safe_load((DATA / "data.yaml").read_text())["names"]
    config = {"path": str(DATA), "train": "train/images", "val": "valid/images", "names": dict(enumerate(names))}
    path = RUNS / "data.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, default="tuned")
    parser.add_argument("--epochs", type=int, default=COMMON["epochs"])
    parser.add_argument("--name", help="run folder under runs/, defaults to the config name")
    args = parser.parse_args()

    settings = {**COMMON, **CONFIGS[args.config], "epochs": args.epochs}
    model = YOLO(settings.pop("model"))
    model.train(
        data=str(write_data_yaml()),
        project=str(RUNS),
        name=args.name or args.config,
        exist_ok=True,
        device=pick_device(),
        **settings,
    )


if __name__ == "__main__":
    main()

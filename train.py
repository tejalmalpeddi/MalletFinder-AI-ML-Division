"""Train YOLOv8n on the bottle/mallet dataset.

Two configs, identical except for augmentation, so the tuned choices can be judged
against Ultralytics' defaults:

    python train.py --config baseline
    python train.py --config tuned

Weights land in runs/<config>/weights/best.pt.
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
        hsv_h=0.01,  # mallet and sand are ~7 hue units apart; this shifts at most 1.8 (default 2.7)
        hsv_v=0.5,   # mean brightness spans 52-187 across the dataset
        scale=0.3,   # default 0.5 can shrink a 16 px object to 8 px, one grid cell
        augmentations=[
            A.MotionBlur(blur_limit=(3, 7), p=0.2),             # camera on a moving rover
            A.ImageCompression(quality_range=(40, 90), p=0.2),  # compressed onboard video frames
            A.CLAHE(clip_limit=2.0, p=0.1),                     # local contrast in harsh sun / shadow
        ],
    ),
}


def write_data_yaml():
    # the downloaded data.yaml lists a test split that doesn't exist, so use our own
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
    args = parser.parse_args()

    settings = {**COMMON, **CONFIGS[args.config], "epochs": args.epochs}
    model = YOLO(settings.pop("model"))
    model.train(
        data=str(write_data_yaml()),
        project=str(RUNS),
        name=args.config,
        exist_ok=True,
        device=pick_device(),
        **settings,
    )


if __name__ == "__main__":
    main()

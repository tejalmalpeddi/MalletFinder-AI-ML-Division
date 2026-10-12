"""Rank training images by how badly the model does on them, to decide what data to collect next.
"""

import argparse
import csv
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate import OUT, draw, match, side
from explore_data import NAMES, SMALL, load_split, tile_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    model = YOLO(args.weights)
    rows = load_split("train")

    scored = []
    for result, (img_path, labels) in zip(
        model.predict([str(p) for p, _ in rows], conf=args.conf, stream=True, verbose=False), rows
    ):
        preds = [(int(c), tuple(map(int, b)), float(s))
                 for c, b, s in zip(result.boxes.cls, result.boxes.xyxy.tolist(), result.boxes.conf)]
        tp, fp, fn = match(preds, labels)
        score = len(fn) + len(fp) + sum(1 - conf for *_, conf in tp)
        scored.append((score, img_path, labels, fp, fn))
    scored.sort(key=lambda r: -r[0])

    with open(OUT / "hard_examples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "score", "false_positives", "missed"])
        for score, img_path, _, fp, fn in scored:
            writer.writerow([img_path.name, f"{score:.2f}", len(fp), len(fn)])

    worst = scored[: len(scored) // 10]
    missed = [l for *_, fn in worst for l in fn]
    false_pos = sum(len(fp) for *_, fp, _ in worst)
    clean = sum(not fp and not fn for *_, fp, fn in scored)
    print(f"top 10% hardest training images ({len(worst)}): {false_pos} FPs, {len(missed)} misses")
    print("  missed by class:", dict(Counter(NAMES[l[0]] for l in missed)))
    small = np.mean([side(l) < SMALL for l in missed]) if missed else 0
    print(f"  missed objects that are COCO-small: {small:.0%}")
    print(f"  images with no errors at all: {clean}/{len(scored)}")

    tiles = [draw(img_path, labels, fp, fn) for _, img_path, labels, fp, fn in scored[:12]]
    cv2.imwrite(str(OUT / "hard_examples.jpg"), tile_grid(tiles, cols=4))
    print(f"ranking in {OUT}/hard_examples.csv, worst 12 in {OUT}/hard_examples.jpg")


if __name__ == "__main__":
    main()

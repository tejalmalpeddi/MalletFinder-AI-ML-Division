"""Run the detector on frames sampled from a phone video, to test outside the dataset's distribution.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from explore_data import COLORS, NAMES, tile_grid

OUT = Path("video_test")
MALLET = NAMES.index("mallet")


def sample_frames(video, n):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append((i, frame))
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    tiles, hits = [], 0
    for i, frame in sample_frames(args.video, args.frames):
        name = f"frame_{i:05d}"
        cv2.imwrite(str(OUT / "frames" / f"{name}.jpg"), frame)
        boxes = model.predict(frame, conf=args.conf, verbose=False)[0].boxes
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.4f}"
                 for c, (x, y, w, h), s in zip(boxes.cls, boxes.xywhn.tolist(), boxes.conf)]
        (OUT / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        hits += any(int(c) == MALLET for c in boxes.cls)
        for c, (x1, y1, x2, y2), s in zip(boxes.cls, boxes.xyxy.tolist(), boxes.conf):
            c = int(c)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[c], 4)
            cv2.putText(frame, f"{NAMES[c]} {s:.2f}", (int(x1), max(int(y1) - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS[c], 3)
        cv2.putText(frame, name, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        tiles.append(cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1]))))

    Path("analysis").mkdir(exist_ok=True)
    sheet_path = f"analysis/video_{args.video.stem}.jpg"
    cv2.imwrite(sheet_path, tile_grid(tiles, cols=5))
    print(f"mallet detected in {hits}/{len(tiles)} sampled frames; sheet in {sheet_path}")


if __name__ == "__main__":
    main()

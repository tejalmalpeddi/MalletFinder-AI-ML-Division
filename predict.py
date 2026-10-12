"""Run the detector on a folder of images and write one YOLO-format .txt per image.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in EXTENSIONS)
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    for result in model.predict([str(p) for p in paths], conf=args.conf, stream=True, verbose=False):
        lines = [
            f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}"
            for c, (x, y, w, h), conf in zip(result.boxes.cls, result.boxes.xywhn.tolist(), result.boxes.conf)
        ]
        out = args.output / (Path(result.path).stem + ".txt")
        out.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"wrote {len(paths)} files to {args.output}/ (conf >= {args.conf})")


if __name__ == "__main__":
    main()

"""Per-class metrics, threshold sweep and error analysis on the validation split.

    python evaluate.py [--weights weights/best.pt] [--conf 0.35]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from explore_data import NAMES, SIZE, SMALL, MEDIUM, load_split, tile_grid, to_pixels
from train import pick_device, write_data_yaml

OUT = Path("analysis")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def match(preds, labels, threshold=0.5):
    used = set()
    tp, fp = [], []
    for c, box, conf in sorted(preds, key=lambda p: -p[2]):
        candidates = [(iou(box, to_pixels(l)), i) for i, l in enumerate(labels) if l[0] == c and i not in used]
        best, i = max(candidates, default=(0, None))
        if best >= threshold:
            used.add(i)
            tp.append((c, box, conf))
        else:
            fp.append((c, box, conf))
    fn = [l for i, l in enumerate(labels) if i not in used]
    return tp, fp, fn


def side(label):
    return np.sqrt(label[3] * label[4]) * SIZE


def draw(img_path, labels, fp, fn):
    img = cv2.imread(str(img_path))
    for l in labels:
        cv2.rectangle(img, to_pixels(l)[:2], to_pixels(l)[2:], (0, 200, 0), 2)
    for c, (x1, y1, x2, y2), conf in fp:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{NAMES[c]} {conf:.2f}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for l in fn:
        cv2.rectangle(img, to_pixels(l)[:2], to_pixels(l)[2:], (0, 220, 255), 3)
        cv2.putText(img, f"missed {NAMES[l[0]]}", to_pixels(l)[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(img, img_path.name[:40], (5, SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.resize(img, (384, 384))


def threshold_sweep(all_preds, rows, thresholds=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)):
    print("\nthreshold sweep (P / R):")
    print("  conf   " + "".join(f"{n:>15s}" for n in NAMES))
    for t in thresholds:
        tp, fp, fn = [0] * len(NAMES), [0] * len(NAMES), [0] * len(NAMES)
        for preds, (_, labels) in zip(all_preds, rows):
            hits, false, missed = match([p for p in preds if p[2] >= t], labels)
            for c, *_ in hits:
                tp[c] += 1
            for c, *_ in false:
                fp[c] += 1
            for l in missed:
                fn[l[0]] += 1
        cells = [f"{tp[c] / max(tp[c] + fp[c], 1):.2f} / {tp[c] / max(tp[c] + fn[c], 1):.2f}" for c in range(len(NAMES))]
        print(f"  {t:4.2f}   " + "".join(f"{x:>15s}" for x in cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    model = YOLO(args.weights)

    metrics = model.val(data=str(write_data_yaml()), split="val", imgsz=640, device=pick_device(),
                        project=str(Path("runs").resolve()), name="val", exist_ok=True, verbose=False)
    print(f"\n{'class':8s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>9s}")
    for c, name in enumerate(NAMES):
        p, r, ap50, ap = metrics.box.class_result(c)
        print(f"{name:8s} {p:6.3f} {r:6.3f} {ap50:6.3f} {ap:9.3f}")
    print(f"{'all':8s} {metrics.box.mp:6.3f} {metrics.box.mr:6.3f} {metrics.box.map50:6.3f} {metrics.box.map:9.3f}")
    print("(P/R above are at Ultralytics' max-F1 threshold; plots in runs/val/)")

    rows = load_split("valid")
    all_preds = [
        [(int(c), tuple(map(int, b)), float(s))
         for c, b, s in zip(result.boxes.cls, result.boxes.xyxy.tolist(), result.boxes.conf)]
        for result in model.predict([str(p) for p, _ in rows], conf=0.05, stream=True, verbose=False)
    ]
    threshold_sweep(all_preds, rows)

    counts = {c: dict(tp=0, fp=0, fn=0, wrong_class=0) for c in range(len(NAMES))}
    missed_sides, errors = [], []
    for preds, (img_path, labels) in zip(all_preds, rows):
        preds = [p for p in preds if p[2] >= args.conf]
        tp, fp, fn = match(preds, labels)
        for c, *_ in tp:
            counts[c]["tp"] += 1
        for c, box, _ in fp:
            counts[c]["fp"] += 1
            if any(l[0] != c and iou(box, to_pixels(l)) >= 0.5 for l in labels):
                counts[c]["wrong_class"] += 1
        for l in fn:
            counts[l[0]]["fn"] += 1
            missed_sides.append(side(l))
        if fp or fn:
            errors.append((len(fp) + len(fn), img_path, labels, fp, fn))

    print(f"\nat conf >= {args.conf}, IoU >= 0.5:")
    for c, k in counts.items():
        p = k["tp"] / max(k["tp"] + k["fp"], 1)
        r = k["tp"] / max(k["tp"] + k["fn"], 1)
        print(f"  {NAMES[c]:7s} P {p:.3f}  R {r:.3f} | TP {k['tp']}  FP {k['fp']} "
              f"({k['wrong_class']} wrong class)  FN {k['fn']}")

    all_sides = np.array([side(l) for _, ls in rows for l in ls])
    missed = np.array(missed_sides)
    print("\nrecall by object size:")
    for name, lo, hi in [("small", 0, SMALL), ("medium", SMALL, MEDIUM), ("large", MEDIUM, 1e9)]:
        n = np.sum((all_sides >= lo) & (all_sides < hi))
        m = np.sum((missed >= lo) & (missed < hi)) if len(missed) else 0
        print(f"  {name:6s} ({n:3d} objects): recall {1 - m / max(n, 1):.2f}")

    errors.sort(key=lambda e: -e[0])
    if errors:
        cv2.imwrite(str(OUT / "errors.jpg"), tile_grid([draw(*e[1:]) for e in errors[:12]], cols=4))
    print(f"\n{len(errors)} of {len(rows)} images have an error; worst 12 in {OUT}/errors.jpg")


if __name__ == "__main__":
    main()

"""Look at the raw dataset before training anything.

Answers the questions that decide the training setup:
  - how big are the objects in pixels, and will the detector's grid even see them?
  - how varied is the lighting, and how much rotation is already baked in?
  - is colour a reliable cue for the mallet, and does the background share it?
  - are there near-duplicate photos across (or within) the train/valid split?

Prints a summary and writes plots to analysis/. Run from the repo root:

    python explore_data.py
"""
import random
from pathlib import Path

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA = Path("Sampled-YD-Object-Detection-2")
OUT = Path("analysis")
NAMES = ["bottle", "mallet"]
COLORS = [(255, 128, 0), (0, 200, 0)]  # BGR: bottle blue, mallet green
SIZE = 512  # every image in the dataset is 512x512

# COCO's size buckets, in pixels of sqrt(box area)
SMALL, MEDIUM = 32, 96


def read_labels(path):
    boxes = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, x, y, w, h = line.split()
            boxes.append((int(c), float(x), float(y), float(w), float(h)))
    return boxes


def load_split(split):
    rows = []
    for img_path in sorted((DATA / split / "images").glob("*.jpg")):
        label_path = DATA / split / "labels" / (img_path.stem + ".txt")
        rows.append((img_path, read_labels(label_path)))
    return rows


def to_pixels(box, W=SIZE, H=SIZE):
    _, x, y, w, h = box
    return int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)


def overview(splits):
    for split, rows in splits.items():
        per_class = [sum(b[0] == c for _, bs in rows for b in bs) for c in range(len(NAMES))]
        both = sum({b[0] for b in bs} == {0, 1} for _, bs in rows)
        crowded = sum(len(bs) >= 4 for _, bs in rows)
        print(f"{split}: {len(rows)} images | "
              + ", ".join(f"{n} {k}" for n, k in zip(NAMES, per_class))
              + f" | {both} images with both classes, {crowded} with 4+ objects")


def box_sizes(splits):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    print(f"\nbox size, sqrt(area) in pixels at {SIZE}x{SIZE}:")
    for c, name in enumerate(NAMES):
        boxes = np.array([b[1:] for rows in splits.values() for _, bs in rows for b in bs if b[0] == c])
        x, y, w, h = boxes.T
        side = np.sqrt(w * h) * SIZE
        small, medium = np.mean(side < SMALL), np.mean((side >= SMALL) & (side < MEDIUM))
        print(f"  {name:7s} median {np.median(side):4.0f}px | small {small:.0%}  medium {medium:.0%}  "
              f"large {1 - small - medium:.0%} | under 16px: {np.mean(side < 16):.0%}")

        axes[0].hist(side, bins=np.logspace(np.log10(4), np.log10(SIZE), 40), alpha=0.6, label=name)
        # log2 so a box twice as tall (-1) and one twice as wide (+1) sit symmetric around 0
        axes[1].hist(np.log2(w / h), bins=40, alpha=0.6, label=name)
        axes[2].scatter(x, y, s=4, alpha=0.5, label=name)

    axes[0].set_xscale("log")
    for edge in (SMALL, MEDIUM):
        axes[0].axvline(edge, color="gray", ls="--", lw=1)
    axes[0].set_title("sqrt(box area) in px, COCO small/medium/large cut-offs dashed")
    axes[1].set_title("log2(width / height)")
    axes[2].set_title("box centers")
    axes[2].invert_yaxis()
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "boxes.png", dpi=120)


def black_corners(gray, patch=6, threshold=8):
    # Roboflow's rotation augmentation fills the uncovered corners with pure black
    corners = [gray[:patch, :patch], gray[:patch, -patch:], gray[-patch:, :patch], gray[-patch:, -patch:]]
    return sum(c.max() <= threshold for c in corners)


def lighting(splits):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    print("\nlighting:")
    for split, rows in splits.items():
        brightness, contrast, rotated = [], [], 0
        for img_path, _ in rows:
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            brightness.append(gray.mean())
            contrast.append(gray.std())
            rotated += black_corners(gray) >= 2
        brightness = np.array(brightness)
        lo, hi = np.percentile(brightness, [5, 95])
        print(f"  {split}: brightness 5th-95th pct {lo:.0f}-{hi:.0f}, contrast median {np.median(contrast):.0f}, "
              f"{rotated / len(rows):.0%} already rotated (2+ black corners)")
        axes[0].hist(brightness, bins=40, alpha=0.6, density=True, label=split)
        axes[1].hist(contrast, bins=40, alpha=0.6, density=True, label=split)

    axes[0].set_title("mean brightness (0-255)")
    axes[1].set_title("contrast (std of gray)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lighting.png", dpi=120)


def hue(rows):
    """Hue of saturated pixels inside mallet boxes, bottle boxes, and outside all boxes.

    Low-saturation pixels (grey sand, white sky) have no meaningful hue, so they're dropped.
    OpenCV hue runs 0-179, so red wraps around: 175 and 2 are neighbours.
    """
    inside = [[] for _ in NAMES]
    background = []
    for img_path, boxes in rows:
        hsv = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2HSV)
        saturated = (hsv[..., 1] >= 80) & (hsv[..., 2] >= 50)
        outside = np.ones(saturated.shape, bool)
        for box in boxes:
            x1, y1, x2, y2 = to_pixels(box)
            region = hsv[y1:y2, x1:x2]
            inside[box[0]].append(region[..., 0][saturated[y1:y2, x1:x2]])
            outside[y1:y2, x1:x2] = False
        background.append(hsv[..., 0][saturated & outside][::10])  # subsample, background is huge

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 181, 3)
    print("\nhue of saturated pixels:")
    for label, values in [*zip(NAMES, inside), ("background", background)]:
        values = np.concatenate(values)
        warm = np.mean((values <= 20) | (values >= 170))
        print(f"  {label:10s} {warm:.0%} red/orange (hue <=20 or >=170)")
        ax.hist(values, bins=bins, density=True, histtype="step", lw=2, label=label)
    ax.set_xlabel("OpenCV hue (0-179)")
    ax.set_title("hue of saturated pixels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "hue.png", dpi=120)


def phash_bits(rows):
    return np.array([imagehash.phash(Image.open(p)).hash.flatten() for p, _ in rows])


def near_duplicates(splits):
    """Count photo pairs whose perceptual hashes are within a Hamming distance, for a range of cut-offs.

    The dataset authors kept one augmented copy per source photo, but the same scene can still
    appear twice under different names (e.g. '01_THEWATERBOTTLE' and '01_THEWATERBOTTLE-copy').
    Sweeping the cut-off instead of picking one shows where 'same photo' ends; closest_pairs.jpg
    lets you check that by eye.
    """
    train, valid = phash_bits(splits["train"]), phash_bits(splits["valid"])
    cross = (train[:, None, :] != valid[None, :, :]).sum(-1)
    within = (train[:, None, :] != train[None, :, :]).sum(-1)
    within = within[np.triu_indices(len(train), k=1)]  # each pair once, no self-pairs

    print("\nnear-duplicate pairs by phash distance (64 bits):")
    print("  <=d   train-valid   train-train")
    for d in (0, 2, 4, 6, 8, 10, 12):
        print(f"  {d:3d}   {np.sum(cross <= d):11d}   {np.sum(within <= d):11d}")

    i, j = np.unravel_index(np.argsort(cross, axis=None)[:6], cross.shape)
    pairs = [(splits["train"][a][0], splits["valid"][b][0], cross[a, b]) for a, b in zip(i, j)]
    tiles = []
    for a, b, d in pairs:
        row = np.hstack([cv2.resize(cv2.imread(str(p)), (256, 256)) for p in (a, b)])
        cv2.putText(row, f"d={d}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(row)
    cv2.imwrite(str(OUT / "closest_pairs.jpg"), np.vstack(tiles))


def draw_grid(items, path, cols=4, tile=320):
    tiles = []
    for img_path, boxes in items:
        img = cv2.imread(str(img_path))
        for box in boxes:
            x1, y1, x2, y2 = to_pixels(box)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[box[0]], 2)
            cv2.putText(img, NAMES[box[0]], (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[box[0]], 2)
        tiles.append(cv2.resize(img, (tile, tile)))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid)


def main():
    OUT.mkdir(exist_ok=True)
    splits = {"train": load_split("train"), "valid": load_split("valid")}
    everything = splits["train"] + splits["valid"]

    overview(splits)
    box_sizes(splits)
    lighting(splits)
    hue(everything)
    near_duplicates(splits)

    draw_grid(random.Random(0).sample(splits["train"], 16), OUT / "samples.jpg")
    # the objects most likely to be missed, to look at before the model ever misses them
    smallest = sorted(everything, key=lambda r: min(b[3] * b[4] for b in r[1]))[:16]
    draw_grid(smallest, OUT / "smallest.jpg")
    print(f"\nplots written to {OUT}/")


if __name__ == "__main__":
    main()

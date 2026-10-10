# Methodology

## 1. How to run it

Tested on macOS 15.5 (Apple M4), Python 3.14.7.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dataset.** Copy `.env.example` to `.env` and set `ROBOFLOW_API_KEY` to your own Roboflow key. The download lands in `Sampled-YD-Object-Detection-2/`, which is git-ignored and treated as read-only.

> TODO: download script, training, evaluation and inference commands.

**Data exploration** (about 5 s, writes plots to `analysis/`):

```bash
python explore_data.py
```

## 2. Thought process

### Looking at the data first

Before choosing a model or any augmentation I wanted to know what the data actually looks like, so `explore_data.py` answers four questions. Numbers below are from that script.

- **Size and balance.** 800 train / 199 valid images, all 512x512. 685 bottle and 537 mallet boxes, so the classes are close enough to balanced that I'm not reweighting anything. 76 images contain both classes and 32 have four or more objects.

- **How big are the objects?** I measured box size as sqrt(area) in pixels rather than as a fraction of the image, because the detector works in pixels: YOLO's finest feature map has one cell per 8 px. Median object is about 58 px for both classes. Using COCO's buckets, about 17% are small (<32 px), and 2-3% are under 16 px, i.e. about 2x2 cells on the finest grid. So small objects will cause some misses, but most objects are medium-sized. My first read of the data ("42% of boxes are under 1% of the image") made this sound worse than it is; 1% of a 512x512 image is still a 51 px square.
  - *Effect on training:* train at `imgsz=640` (upscales the 512 images a bit) and don't push the scale augmentation toward shrinking objects further.

- **Lighting and rotation.** Mean brightness ranges from 52 to 187 (5th-95th percentile), so there are genuinely dark and washed-out images. About 35% of images already have black corners from Roboflow's rotation augmentation. I detected that by checking whether at least two corner patches are pure black.
  - *Effect on training:* brightness jitter is worth keeping. Adding a lot more rotation is not, since a third of the data is already rotated.

- **Is colour a shortcut for the mallet?** The mallet is always orange-red, so I expected colour to make it easy. I measured the hue of saturated pixels inside mallet boxes, inside bottle boxes and in the background (`analysis/hue.png`). 97% of saturated mallet pixels are red/orange, but so are 71% of saturated background pixels, because the desert sand is orange too. The two peaks are only about 7 OpenCV hue units apart (mallet around 5-10, sand around 12-20, on a 0-179 scale). So colour helps, but only narrowly.
  - *Effect on training:* hue augmentation has to stay small. In the installed Ultralytics version the hue shift is `± hsv_h * 180` (I checked `ultralytics/data/augment.py`), so the default `hsv_h=0.015` shifts by at most about 2.7 units. Raising it would make the model learn that sand-coloured things can be mallets. I'm keeping it at or below the default.

- **Is the train/valid split leaking?** The README says many images are augmented copies of the same photos. The sampling script (`sampling/sample-roboflow.py`) already keeps only one augmented copy per source photo, so exact copies across the split shouldn't exist. To check for the same photo uploaded under a different name, I compared perceptual hashes (64-bit pHash, Hamming distance) for every train/valid pair and every train/train pair, sweeping the cut-off instead of picking one:

  | distance <= | train-valid pairs | train-train pairs |
  |---:|---:|---:|
  | 6 | 0 | 1 |
  | 8 | 3 | 4 |
  | 10 | 12 | 22 |
  | 12 | 31 | 76 |

  Nothing matches up to distance 6. Looking at the closest pairs (`analysis/closest_pairs.jpg`), the ones at 8-10 are not copies, but some are **the same scene shot twice** (same bottle on the same rocks, same desk). So I kept the provided split, since it has no duplicates. But valid images are not fully independent of train, and the validation scores are probably somewhat optimistic. The video test is the more honest check of generalisation.

- **The hardest examples.** `analysis/smallest.jpg` shows the 16 images with the smallest objects: distant bottles on gravel, mallets a few pixels wide, one image with heavy Roboflow noise, and a group selfie where the bottle is a minor detail. I expect most false negatives to come from images like these.

### Model, preprocessing, evaluation

> TODO once training is done.

## 3. Known limitations

- Some scenes appear in both train and valid (see above), so validation metrics likely overestimate real-world performance.
- The hue thresholds in `explore_data.py` (saturation >= 80, value >= 50, "red/orange" = hue <= 20 or >= 170) are hand-picked. The hue histogram plot doesn't depend on the red/orange cut-off, but the 97% / 71% numbers do.
- The rotation count is a heuristic: a genuinely dark photo could have black corners without being rotated.

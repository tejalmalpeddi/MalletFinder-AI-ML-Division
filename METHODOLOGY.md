# Methodology

## 1. How to run it

Tested on macOS 15.5 (M4, MPS), Python 3.14.7. Falls back to CUDA or CPU automatically.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # set ROBOFLOW_API_KEY
# download: run roboflow_data.ipynb (reads ROBOFLOW_API_KEY) -> Sampled-YD-Object-Detection-2/
python explore_data.py               # data stats + plots -> analysis/
python train.py --config tuned --epochs 20   # ~13 min on M4 -> runs/tuned/weights/best.pt
python evaluate.py                   # metrics + analysis/errors.jpg (uses weights/best.pt)
python predict.py <image_dir> <out_dir>      # conf 0.25, one .txt per image
```

Trained weights are committed at `weights/best.pt` (YOLOv8n, 6.5 MB). Inference output per line: `class_id x_center y_center width height confidence`, normalized 0-1. **Confidence threshold: 0.25** (`--conf` to change).

## 2. Thought process

**Data** (`explore_data.py`)
- 800 train / 199 valid, all 512x512. 685 bottle / 537 mallet boxes: balanced enough, no reweighting.
- Median object ~58 px (sqrt area); 17% COCO-small, 2-3% under 16 px. Train at `imgsz=640`, don't add shrinking scale aug.
- Brightness 52-187 (5th-95th pct): keep value jitter. ~35% of images are already rotated (black corners): no extra rotation.
- Mallet hue ~5-10, sand background ~12-20 (OpenCV 0-179). Colour separates them, but narrowly. Ultralytics shifts hue by `±hsv_h*180` (checked in `augment.py`), so `hsv_h` stays <= the default 0.015.
- Leakage: the sampling script keeps one augmented copy per source photo. pHash check: 0 train/valid pairs within distance 6. At 8-10 some pairs are the same scene shot twice. Kept the given split.

**Model and augmentation** (`train.py`)
- YOLOv8n, COCO-pretrained: 3.1M params, 8.6 GFLOPs, ~2 ms/img on M4. Small enough for the RKNN/NPU target, and the most widely supported YOLO for that toolchain.
- Tuned augmentation on top of the Ultralytics defaults: `hsv_h=0.01` (mallet/sand hue gap), `hsv_v=0.5` (lighting range), `scale=0.3` (protect small objects), plus MotionBlur, JPEG compression and CLAHE (moving rover camera, compressed onboard frames, harsh sun/shadow).
- 20 epochs, seed 0. Planned a baseline-vs-tuned comparison but cut it for time (see limitations).

**Results** (`evaluate.py`, valid set)

| class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bottle | 0.877 | 0.781 | 0.829 | 0.512 |
| mallet | 0.776 | 0.686 | 0.833 | 0.494 |
| all | 0.827 | 0.734 | 0.831 | 0.503 |

P/R are at Ultralytics' max-F1 threshold. At the deployment threshold of 0.25, using my own IoU >= 0.5 matching:

| class | P | R | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| bottle | 0.75 | 0.80 | 120 | 41 | 31 |
| mallet | 0.54 | 0.88 | 93 | 79 | 13 |

Recall by size: small 0.62, medium 0.88, large 0.87.
- 0.25 favours mallet recall. Missing the mallet costs the rover more than a false alarm it can re-check on the next frame.
- mAP50-95 (~0.5) is much lower than mAP50: detections are found, but box placement is loose.

**Error analysis** (`analysis/errors.jpg`, 12 worst valid images)
- **Missing labels inflate FPs.** In several outdoor scenes the model boxes real bottles that aren't labeled (one image has four bottles, only one labeled). Some of the 41 bottle "FPs" are label errors, not model errors.
- **Mallet localisation.** Most mallet FPs are stacked or loose boxes on a real mallet: thin, rotated objects in axis-aligned boxes, where IoU < 0.5 counts as both an FP and an FN. This explains the low mallet precision (0.54) next to its high recall.
- **Small and distant objects.** Recall is 0.62 for small objects vs 0.88 for medium: distant bottles on gravel and far-off clusters are missed.
- **Out-of-domain images.** A close-up mallet on a pegboard (indoor, greyscale) gets boxed only on its head, and a synthetic drawing on a white background is missed. Few training images look like this.
- Only 1 cross-class confusion: bottle vs mallet is not the problem; background is.

## 3. Known limitations

- No baseline run. The tuned config was never compared against default augmentation, so I can't claim the augmentation helped. Next: `python train.py --config baseline --epochs 20` and compare.
- Only 20 epochs (time budget). mAP50-95 rose from 0.46 to 0.50 over the last 5 epochs and only levelled off at 19-20; more epochs would likely help.
- Video test (Part 1.6) not done.
- Some scenes appear in both train and valid, so valid metrics are likely optimistic.
- Missing labels in valid bias precision down. A relabel pass on the FP images would give truer numbers.
- MPS ops are non-deterministic, so `seed=0` doesn't make runs bit-identical on Apple silicon.
- The dataset download is still the notebook; it should be a script for a one-command fresh-clone run.

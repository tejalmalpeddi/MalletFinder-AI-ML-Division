# Methodology

## 1. How to run it

Tested on macOS 15.5 (M4, uses MPS) with Python 3.14.7. `train.py` picks CUDA, then MPS, then CPU, so it should run anywhere.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                          # put your key in ROBOFLOW_API_KEY
# run roboflow_data.ipynb to download the data -> Sampled-YD-Object-Detection-2/

python explore_data.py                        # data stats, plots go to analysis/
python train.py --config tuned --epochs 20    # ~13 min on M4 -> runs/tuned/weights/best.pt
python train.py --config baseline --epochs 20 # same thing with default augmentation
python evaluate.py                            # metrics + analysis/errors.jpg, uses weights/best.pt
python predict.py <image_dir> <out_dir>       # inference
python video_test.py <video>                  # 20 frames from a phone video -> analysis/video_<name>.jpg
```

The trained model is in the repo at `weights/best.pt` (YOLOv8n, 6.5 MB), so you can skip training and go straight to `predict.py`.

`predict.py` writes one `.txt` per image, one detection per line: `class_id x_center y_center width height confidence`, all normalized 0-1. Images with nothing detected get an empty file. **Confidence threshold is 0.25**; change it with `--conf`.

## 2. Thought process

**Looking at the data first** (`explore_data.py`)
- 800 train / 199 valid images, all 512x512. 685 bottle boxes vs 537 mallet boxes: close enough that I didn't reweight anything.
- I measured box size in pixels (sqrt of area), because YOLO's finest grid cell is 8 px. Median object is ~58 px, 17% are COCO-small (<32 px), and 2-3% are under 16 px (about 2x2 grid cells). So I train at 640 instead of 512, and I didn't want augmentation that shrinks objects more.
- Brightness ranges from 52 to 187 (5th-95th percentile), so brightness jitter makes sense. About 35% of images already have black corners from Roboflow's rotation, so I didn't add more rotation.
- I assumed the orange mallet would be easy to spot by colour. Then I plotted hue: the sand in the background is orange too. Mallet pixels peak around hue 5-10, sand around 12-20 (OpenCV's 0-179 scale). I checked the Ultralytics source, and hue augmentation shifts by up to `hsv_h * 180`. So the default 0.015 can already move a colour 2.7 units, and anything bigger starts turning mallets into sand. That's why I lowered it.
- Leakage: the README says many images are augmented copies. The sampling script already keeps only one copy per source photo, but I still checked with perceptual hashes. There are no train/valid pairs within distance 6. At 8-10, some pairs are the same scene shot twice, not the same photo. I kept the given split but treat the valid scores as a bit optimistic.

**Model and augmentation** (`train.py`)
- YOLOv8n, pretrained on COCO: 3.1M params, 8.6 GFLOPs, ~2 ms per image on my M4. The real target is an NPU through RKNN, so a small model that the toolchain supports well mattered more than squeezing out accuracy.
- Two configs that differ only in augmentation:
  - `baseline`: Ultralytics defaults.
  - `tuned`: `hsv_h=0.01` (mallet/sand hue gap), `hsv_v=0.5` (wide lighting range) and `scale=0.3` (don't shrink small objects as much), plus motion blur, JPEG compression and CLAHE. Those three are aimed at the rover: a moving camera, compressed video frames, harsh sun and shadow.
- Both trained for 20 epochs with seed 0. 20 epochs was a time call (~13 min each).

**Baseline vs tuned** (valid set, mAP50 / mAP50-95)

| | bottle | mallet | all |
|---|---|---|---|
| baseline | 0.835 / 0.515 | **0.911 / 0.564** | **0.873 / 0.540** |
| tuned | 0.829 / 0.512 | 0.833 / 0.494 | 0.831 / 0.503 |

- My tuned augmentation lost, mostly on the mallet. My best guesses:
  - Blur, compression and CLAHE make training harder, and 20 epochs probably isn't enough to make that back.
  - The valid set looks like the train set (no blur, no compression), so robustness augmentation can't show a benefit on it.
  - `scale=0.3` cut down scale variety.
- I changed four things at once, so I can't say which one hurt. That's the main thing I'd fix next.
- The baseline wasn't better on my videos either (see below).
- `weights/best.pt` is still the tuned model, because I did the threshold choice and error analysis on it. Going by these numbers I'd ship the baseline: `cp runs/baseline/weights/best.pt weights/best.pt`.

**Results for the submitted (tuned) model** (`evaluate.py`, valid set)

| class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bottle | 0.877 | 0.781 | 0.829 | 0.512 |
| mallet | 0.776 | 0.686 | 0.833 | 0.494 |
| all | 0.827 | 0.734 | 0.831 | 0.503 |

These P/R values are at Ultralytics' best-F1 threshold. I also matched predictions to labels myself (IoU >= 0.5, same class) at the threshold I actually use, 0.25:

| class | P | R | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| bottle | 0.75 | 0.80 | 120 | 41 | 31 |
| mallet | 0.54 | 0.88 | 93 | 79 | 13 |

Recall by object size: small 0.62, medium 0.88, large 0.87.
- I picked 0.25 to favour mallet recall. For the rover, missing the mallet is worse than a false alarm it can re-check on the next frame.
- mAP50-95 (~0.5) is a lot lower than mAP50 (~0.83). The model finds things, but its boxes are loose.

**Error analysis** (`analysis/errors.jpg`, the 12 valid images with the most errors)
- **Missing labels.** In some outdoor scenes the model boxes real bottles that nobody labeled. One image has four bottles and only one label. So some of the 41 bottle "false positives" are labeling mistakes, not model mistakes.
- **Loose boxes on mallets.** Most mallet false positives sit on a real mallet: either several stacked boxes or one box that's too loose. Mallets are thin and often diagonal, so an axis-aligned box around them is mostly background, and small shifts drop IoU under 0.5. That one box then counts as both a false positive and a miss. This is why mallet precision is 0.54 while its recall is 0.88.
- **Small, far-away objects.** Recall drops to 0.62 for small objects: distant bottles on gravel, clusters far off.
- **Images unlike the training set.** A close-up mallet on a grey pegboard only gets its head boxed, and a drawing on a white background is missed completely. There are very few indoor or synthetic images in training.
- Bottle vs mallet confusion happened once. The problem is the background, not telling the two classes apart.

**Video test** (`video_test.py`, 20 evenly spaced frames per video)
- I had no mallet, so I filmed two indoor phone videos on the same rug: a yellow and black screwdriver (`analysis/video_IMG_4278.jpg`), then a light wooden hairbrush, which is closer to a mallet shape (`analysis/video_IMG_4279.jpg`).
- The mallet was detected in **0 of 20** frames in both. To check it wasn't just the threshold, I looked at the highest mallet confidence in any frame: 0.09 (tuned) and 0.17 (baseline). So it isn't close. The only detection was the whole rug called a "bottle" (0.34 tuned, 0.60 baseline).
- This matches the hue finding: every mallet in training is orange-red on dirt, so the model leans on colour and background more than shape. Neither of my objects is orange, and a patterned rug looks nothing like the training backgrounds.
- What I didn't get to test: an orange object on the same rug. That would show whether it's the colour or the background. Either way the fix is more mallet colours and backgrounds in the data. Another option is hue augmentation on the mallet class only; doing it globally would make sand look like mallets.

## 3. Known limitations

- One run per config, and MPS isn't deterministic, so I don't know how much of the baseline/tuned gap is noise. Next I'd test one augmentation at a time, train longer, and use 2-3 seeds.
- Only 20 epochs. mAP50-95 went from 0.46 to 0.50 over the last 5 epochs and only flattened at the very end, so longer training would probably help, but my computer was crashing and running slow.
- The video test used non-orange objects indoors. It shows the model doesn't generalise beyond "orange on dirt", but not about how it does on the real mallet.
- Some scenes appear in both train and valid, so the valid numbers are probably a bit high.
- Missing labels in valid push precision down. Relabeling the false-positive images would give more accurate numbers.


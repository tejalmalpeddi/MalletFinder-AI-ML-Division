# Methodology

## 1. How to run it

I tested this on my MacBook (M4, macOS) with Python 3.14.7. The training script uses CUDA if you have an NVIDIA GPU, otherwise MPS on a Mac, otherwise CPU, so it should work on other machines too. I've only actually run it on my Mac though.

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env`, paste your Roboflow API key into `ROBOFLOW_API_KEY`, and run `roboflow_data.ipynb` to download the dataset.

Running everything:

```bash
python explore_data.py          # dataset stats and plots (saved in analysis/)
python train.py --config baseline --epochs 60 --name baseline_60   # the final model, took me ~35 min
python train.py --config tuned --epochs 20      # my augmentation experiment, optional
python evaluate.py              # metrics, threshold sweep, error images
python hard_examples.py         # finds the hardest training images
python video_test.py <video>    # runs the model on 20 frames from a video
```

You don't have to train it yourself. The final model is already saved at `weights/best.pt` (YOLOv8n, 6.5 MB).

To run inference on a folder of test images:

```bash
python predict.py path/to/test_images path/to/output
```

It makes one `.txt` file per image. Each line is one detection: `class_id x_center y_center width height confidence`, with everything scaled 0 to 1. If nothing is detected the file is empty. The confidence threshold is 0.35 by default, and you can change it with `--conf`.

## 2. Thought process

### Looking at the data

- There are 800 training images and 199 validation images, all 512x512. There are 685 bottle boxes and 537 mallet boxes, which seemed close enough that I didn't do anything special to balance the classes.
- **Train/val split:** I used the split that came with the dataset. The README mentioned that a lot of images are augmented copies of each other, so I was worried the same photo might be in both train and val. I compared images with perceptual hashing and didn't find any exact duplicates across the split. A few pairs were the same scene taken twice, so I think the validation scores are probably a little higher than they would be on totally new data.
- **Object size:** A lot of the objects are pretty small. The median is around 58 pixels across, 17% are under 32 pixels, and a few are under 16 pixels. Because of that I trained at 640 instead of a smaller size, and I tried not to use augmentations that make objects even smaller.
- **Brightness:** Brightness varies a lot between images, so random brightness changes during training made sense. Around 35% of the images already had black corners from being rotated, so I didn't add any more rotation.
- **Colour:** I thought the orange mallet would be easy to detect because of its colour, but when I plotted the hue values I found out the sand in a lot of the images is also orange. The mallet and sand hues are really close together. I looked at how Ultralytics does hue augmentation and the default setting already shifts colours a little bit. If I turned it up more, the model might start confusing mallets with sand, so I decided not to increase it.

### Choosing a model

- I used YOLOv8n (the smallest YOLOv8) pretrained on COCO. The model eventually needs to run on the rover's NPU, so I wanted something small and fast that is well supported, instead of a bigger model that might be a bit more accurate but too slow.
- I trained two versions that only differ in augmentation:
  - **baseline:** the default Ultralytics settings.
  - **tuned:** less hue shift (because of the sand problem), more brightness change, less zooming out, plus motion blur, JPEG compression, and contrast adjustment (CLAHE). I added those last three because the rover camera will be moving, the video will be compressed, and the lighting outside can be harsh.

### Augmentation results

mAP50 / mAP50-95 on the validation set:

| | bottle | mallet | all |
|---|---|---|---|
| tuned, 20 epochs | 0.829 / 0.512 | 0.833 / 0.494 | 0.831 / 0.503 |
| baseline, 20 epochs | 0.835 / 0.515 | 0.911 / 0.564 | 0.873 / 0.540 |
| baseline, 60 epochs (final) | 0.871 / 0.565 | 0.975 / 0.648 | 0.923 / 0.606 |

- My tuned version actually did worse than the defaults, especially on mallets. I'm not totally sure why, but my guesses are:
  - The extra augmentations make training harder, and 20 epochs might not have been enough for it to catch up.
  - The validation images don't have blur or compression, so the validation set can't really show if those augmentations help.
  - Reducing the zoom range might have made the model see less variety in object sizes.
- One mistake I made was changing several settings at the same time, so I can't tell which change hurt. Next time I'd change one thing at a time.
- Since the default settings won, I trained that version for 60 epochs. That ended up being the biggest improvement of anything I tried, and the scores were still slowly going up at the end.

### Final model results

| class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bottle | 0.918 | 0.828 | 0.871 | 0.565 |
| mallet | 0.925 | 0.962 | 0.975 | 0.648 |
| all | 0.922 | 0.895 | 0.923 | 0.606 |

To choose a confidence threshold, I tried a few values and checked precision and recall for each class:

| conf | bottle P / R | mallet P / R |
|---|---|---|
| 0.15 | 0.84 / 0.87 | 0.80 / 0.98 |
| 0.25 | 0.89 / 0.84 | 0.92 / 0.96 |
| 0.35 | 0.93 / 0.82 | 0.94 / 0.96 |
| 0.50 | 0.96 / 0.81 | 0.97 / 0.92 |

- I picked 0.35. Mallet recall stays the same between 0.25 and 0.40, so raising it to 0.35 gets rid of some false positives without missing any more mallets. Above 0.40 it starts missing mallets. I think missing a mallet is worse than a false alarm for the rover, since it can just check again on the next frame.
- At 0.35 the model gets 124 bottles right, with 10 false positives and 27 misses. For mallets it gets 102 right, with 7 false positives and 4 misses.
- It's much worse at small objects (0.68 recall) than medium ones (0.94).
- mAP50-95 is a lot lower than mAP50, which means the model usually finds the object but the boxes aren't very tight, especially for mallets that are thin and diagonal.

### Error analysis

I looked at the 12 validation images with the most mistakes (`analysis/errors.jpg`):

- **Bottles are harder than mallets.** Most of the bottle errors are misses. I think this is because bottles look really different from each other (different colours, see-through vs not, standing up vs lying down), while all the mallets basically look the same.
- **Small, far away objects** get missed a lot, like bottles far away on gravel.
- **Close-up mallets get called bottles.** When the mallet fills the whole picture you can't see the head, so it just looks like an orange tube, which is kind of like a bottle.
- **Some labels are missing.** In a few images the model found real bottles that weren't labeled, so some of the "false positives" are actually correct.
- **Long thin things** sometimes get called mallets, like a pole on a roof.
- **Drawings or fake-looking images** get missed, probably because there aren't many of them in the training data.

### Video test

I didn't have an actual mallet, so I filmed three objects on the same rug at home, moving the camera the same way each time, and ran the model on 20 frames from each video:

- a yellow and black screwdriver
- a wooden hairbrush (kind of mallet-shaped)
- an orange bottle

Results:

- **Screwdriver and hairbrush:** nothing was detected in any frame. So it didn't find a mallet, but it also didn't have any false positives. (My earlier 20-epoch model labeled the whole rug as a bottle in one frame, but the final model didn't.)
- **Orange bottle:** detected as a bottle in 14 out of 20 frames, and never as a mallet. The frames it missed were when the bottle was far away or the frame was blurry.

At first I thought the rug background or the colour was the problem, but since it found the bottle fine on the same rug, I think the model mostly cares about shape. The screwdriver and hairbrush just don't look like the mallets it was trained on. I still can't say how well it would do on a real orange mallet since I couldn't test one.

### Extra: speed for the rover

I measured speed on my laptop CPU, since I don't have the rover's NPU:

| input size | ms per image | ~FPS | mAP50 | mallet mAP50 |
|---|---:|---:|---:|---:|
| 640 | 61.9 | 16 | 0.923 | 0.975 |
| 512 | 41.1 | 24 | 0.924 | 0.988 |
| 320 | 21.7 | 46 | 0.884 | 0.970 |

- Running at 512 is just as accurate as 640 and about a third faster. This makes sense because the images are already 512, so 640 just stretches them. I wish I'd checked this before training.
- 320 is almost 3x faster and the mallet score barely drops, so it could be worth it if the NPU is slow. It would probably make the small object problem worse though.
- I didn't get to export the model to run on the NPU (ONNX, then RKNN with INT8). From what I read, INT8 usually loses a little accuracy but runs a lot faster.

### Extra: finding hard examples

`hard_examples.py` runs the model on the training images and ranks them by how many mistakes it makes.

- 750 out of 800 training images had no mistakes, so the problems are mostly in a small group of images.
- In the hardest 80 images, most of the missed objects were small.
- Looking at the worst ones, I found some labeling mistakes (mallets labeled as bottles) and a lot of images with groups of tiny objects far away.
- If I had more time, I would:
  1. Fix the labels in the hardest images first, since that's easy and wrong labels hurt training.
  2. Collect more pictures of the things it struggles with: far away objects on sand and gravel, close-up mallets, and blurry frames. Ideally from the rover's actual camera.
  3. Retrain and repeat.

## 3. Known limitations

- I only ran each augmentation experiment once, and training on a Mac isn't exactly repeatable, so some of the difference could just be randomness.
- The final model was still improving at 60 epochs. I stopped there because my laptop kept crashing on longer runs.
- I didn't test on a real mallet, only on objects I had at home.
- Some scenes show up in both train and validation, so the validation scores are probably a bit too high.
- There are some labeling mistakes in the dataset, which make the scores a bit off in both directions.
- Speed was measured on my laptop, not the actual rover hardware, and I didn't convert the model for the NPU.

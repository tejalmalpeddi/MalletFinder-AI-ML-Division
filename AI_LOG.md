# AI Usage Log

## 1. Planning the project

**What I asked:** Read the README and repo and lay out a basic structured plan for the take-home, without editing anything.

**What I kept vs. rewrote, and why:** I kept the overall order and understanding of look at the data, then train, evaluate, inference, error analysis, video, docs. I asked it to draft specific functions based on my data analysis one at a time, so I could review and approve each step and commit in small pieces.

**What the AI got wrong that I had to catch:** It said near-duplicate leakage between train and valid was a likely problem. After reading the sampling sample-roboflow.py, it turned out the dataset authors already kept only one augmented copy per source photo, so the risk was much smaller than it first stated.

**How I verified it:** Read the sampling script, then measured it with the perceptual-hash check.

## 2. Data exploration script (`explore_data.py`)

**What I asked:** First a basic exploration script (box sizes, brightness, a sample grid, duplicate check). Then I revised it for a second version that focused on specific questions that actually change training settings and approach.

**What I kept vs. rewrote, and why:** The first version used arbitrary thresholds where brightness < 60 counted as "dark", one phash cut-off of 6, and box area as a fraction of the image. The second version (mine) replaced these with:  
- box size in pixels with COCO's size buckets, because the detector's grid is in pixels. 
- brightness percentiles instead of hand-picked thresholds
- a sweep over duplicate cut-offs plus a picture of the closest pairs, so the cut-off is judged by eye
- a within-train duplicate check
- a black-corner count to measure how much rotation is already in the data
- a hue comparison of mallet vs. bottle vs. background pixels

**What the AI got wrong that I had to catch:**
- In the first run both classes came out at exactly "median area 0.013, 42% under 1%", which looked like a filtering bug. I had it recompute independently and the medians are 0.0135 vs. 0.0128 and both happen to be 42.3%, so it was a coincidence, not a bug. Its first cross-check was also broken: it used `cat` on label files without trailing newlines, which glued lines together and gave wrong counts.
- It initially described the objects as "tiny" and called small objects the main difficulty. In pixels, the median object is about 58 px and only about 17% are COCO-small, so that was overstated.
- It initially said the mallet's orange colour would be an easy cue. The hue histogram showed the sand background is also orange, only about 7 hue units away. Hue augmentation needs to stay small, not be used freely.

**How I verified it:**
- Looked at generated plots and image grid, not just the printed numbers.
- Opened the closest duplicate pairs to decide what the phash distances actually mean (same scene vs. same photo).
- Checked what `hsv_h` does in the installed Ultralytics source (`augment.py`: shift of `± hsv_h * 180` hue units) instead of trusting the AI's description.


## 3. Training script (`train.py`)

**What I asked:** A training script skeleton with a baseline and a tuned config.

**What I kept vs. rewrote, and why:** I set the tuned augmentation values from my own exploration results: `hsv_h=0.01` because of the narrow mallet/sand hue gap, `hsv_v=0.5` for the measured brightness range, and `scale=0.3` to protect small objects. I added motion blur, JPEG compression and CLAHE to match rover camera conditions. When my machine kept crashing, I cut training from 50 to 20 epochs and paused the baseline, then brought it back once the tuned run finished. I also fixed the background run so it survived session restarts (`nohup`) and replaced the broken completion check.

**What the AI got wrong that I had to catch:**
- The first background training run died when the session restarted, so it had to be rerun with `nohup`.
- Its "wait until training finishes" loop used `pgrep -f "train.py"`, which matched its own command, so it never stopped even though training was done.

**How I verified it:**
- Ran 1 epoch first as a smoke test.
- Looked at a training batch image to check the boxes still line up after augmentation.
- Checked the training log to confirm the settings and augmentations were actually applied.

## 4. Evaluation script (`evaluate.py`)

**What I asked:** Per-class metrics plus matching at my 0.25 threshold.

**What I kept vs. rewrote, and why:** I kept Ultralytics' val for mAP, and then extended the evaluation with recall by object size and a separate "wrong class" count, so bottle/mallet confusion could be told apart from background mistakes. I used the false positive/negative counts to pull the worst images for my error analysis, and fixed the output path so plots landed in the right folder. Also added a per-class threshold sweep (`threshold_sweep`).


**What the AI got wrong that I had to catch:** Ultralytics nests relative paths, so the val plots ended up in `runs/detect/runs/val`. 

**How I verified it:**
- Ran it on a 2-epoch checkpoint before the real model was done.
- Looked through `errors.jpg` myself. Several "false positives" are real bottles that were never labeled, which changed how I read the precision numbers.

## 5. Inference script (`predict.py`)

**What I asked:** Inference in the README's output format.

**What I kept vs. rewrote, and why:** I reviewed it against the README spec and kept it after confirming it writes one `.txt` per image, including an empty file for images with no detections, so every image has an output. I later changed the default threshold to 0.35 based on the sweep.

**What the AI got wrong that I had to catch:** Nothing much here.

**How I verified it:** Ran it on all 199 valid images, and compared one prediction to its label (predicted center 0.468 / 0.585 vs label 0.471 / 0.585)


## 7. Video test script + video 1 (`video_test.py`, screwdriver, `IMG_4278.MOV`)

**What I asked:** A frame-sampling script that writes predictions with a contact sheet.

**What I kept vs. rewrote, and why:** I designed the real-world test myself: with no mallet available, I filmed a yellow and black screwdriver on a rug as a stand-in. I changed some things and fixed the frame count so it no longer included blank padding tiles.

**What the AI got wrong that I had to catch:** The frame count it printed included the blank padding tiles.

**How I verified it:** Tested it on a fake video made from valid images first. Then I looked through all 20 frames of my video: 0/20 mallet detections, and the whole rug called a "bottle".

## 8. Video 2 (wooden hairbrush, `IMG_4279.MOV`)

**What I asked:** Rerun the test on a second video.

**What I kept vs. rewrote, and why:** I chose a wooden hairbrush over the AI's suggestion of an orange object, because its shape (flat head + handle) is closer to a mallet and it's what I had. I kept the same rug and camera path for comparability, and noted in METHODOLOGY that the two videos change both colour and shape, so the colour question stays untested. I also changed the script to save one contact sheet per video so results weren't overwritten.

**What the AI got wrong that I had to catch:** The contact sheet was always saved to the same file, so video 2 would have overwritten video 1. Changed to one file per video.

**How I verified it:** I checked the highest mallet confidence in any frame (0.09), to make sure it wasn't just the 0.25 threshold.

## 9. Baseline vs tuned comparison

**What I asked:** Train the baseline for 20 epochs and compare.

**What I kept vs. rewrote, and why:** The baseline won (mallet mAP50 0.911 vs 0.833), and I reported that rather than dropping the comparison. I kept the tuned weights as the submitted model because my error analysis was built on them, and stated in METHODOLOGY that I would ship the baseline. I set up both weights to be validated in one script so the comparison was fair.

**What the AI got wrong that I had to catch:** It expected the tuned augmentation to help, and it didn't.

**How I verified it:** Re-ran validation for both weights in one script so they're compared on identical settings. Also ran the baseline on both videos: max mallet confidence 0.17, so it isn't more robust there either.

## 10. Baseline vs tuned comparison

**What I asked:** Train the baseline for 20 epochs and compare it on identical settings.

**What I kept vs. rewrote, and why:** The baseline won (mallet mAP50 0.911 vs 0.833). I reported that instead of dropping it.

**What the AI got wrong that I had to catch:** It expected the tuned augmentation to help.

**How I verified it:** Validated both weights in one script. Ran the baseline on both videos.

## 11. Video 3, orange bottle

**What I asked:** Run an orange bottle on the same rug.

**What I kept vs. rewrote, and why:** This overturned my earlier conclusion. Detected as bottle in 11/20 frames, never as mallet, so colour alone doesn't decide the class; shape does.

**What the AI got wrong that I had to catch:** The colour/background explanation it gave after videos 1-2 was too strong.

**How I verified it:** Counted detections per frame from the label files.

## 12. Stronger model: baseline at 60 epochs + threshold choice

**What I asked:** Ways to get a stronger model. I considered moving training to Colab and decided against it (setup overhead vs. ~35 min locally). Then I trained the baseline for 60 epochs.

**What I kept vs. rewrote, and why:**
- mAP50 rose from 0.873 to 0.923 (mallet 0.975), so I shipped it.
- The sweep showed mallet recall stays at 0.96 from 0.25 to 0.40, so I chose one threshold of 0.35 instead of per-class thresholds.

**What the AI got wrong that I had to catch:** When re-running the videos, its bottle count summed all three videos, because they share an output folder. Recounted per video: 14/20 on the orange bottle.

**How I verified it:**
- Re-ran `evaluate.py`, all three videos and `hard_examples.py` on the new weights, so every reported number matches the shipped model.
- Measured CPU speed and accuracy at 640/512/320: 512 matched 640 and ran 34% faster.

## 13. Hard example mining (`hard_examples.py`)

**What I asked:** Rank training images by misses, false positives and low confidence, for the active-learning stretch goal.

**What I kept vs. rewrote, and why:** The worst images showed orange mallets labeled "bottle" in the training data, so label fixes come first in my data plan.

**What the AI got wrong that I had to catch:** The first version would have crashed with fewer than 12 images (grid not padded).

**How I verified it:** Looked at the worst 12 images myself.

## 14. Code cleanup across scripts

**What I asked:** Clean up the code: remove duplication and leftover formatting.

**What I kept vs. rewrote, and why:**
- The grid code was copy-pasted in four files; it's now one `tile_grid` helper.
- `video_test.py` imports `NAMES`/`COLORS` instead of redefining them.
- Named variables replace tuple indexing.
- I removed most docstrings and comments myself; one-line usage docstrings were added back.

**What the AI got wrong that I had to catch:** Nothing new, but my editor saved an older copy of `explore_data.py` over the change, which broke `evaluate.py`'s import; `tile_grid` was restored.

**How I verified it:** Compiled and imported every module, and tested `tile_grid` on its own.

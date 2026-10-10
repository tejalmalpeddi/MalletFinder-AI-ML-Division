# AI Usage Log

I used Claude Code (in VS Code) as an assistant throughout. I asked it to propose a plan, then draft each file for me to review before it went into the repo.

## 1. Planning the project

**What I asked:** Read the README and repo and lay out a step-by-step plan for the take-home, without editing anything.

**What I kept vs. rewrote, and why:** Kept the overall order: look at the data, then train, evaluate, inference, error analysis, video, docs. I chose to have it draft files that I approve one at a time, so I could review each step and commit in small pieces.

**What the AI got wrong that I had to catch:** It said near-duplicate leakage between train and valid was a likely problem. After reading `sampling/sample-roboflow.py`, it turned out the dataset authors already keep only one augmented copy per source photo. So the risk was much smaller than first stated: only the same photo uploaded twice under different names.

**How I verified it:** Read the sampling script, then measured it with the perceptual-hash check (entry 3): 0 train/valid pairs within distance 6.

## 2. Pinning requirements

**What I asked:** Install the training libraries and add pinned versions to `requirements.txt`.

**What I kept vs. rewrote, and why:** Kept only the packages my code imports directly, rather than the full `pip freeze`, so the file stays readable. I edited the header comment myself.

**What the AI got wrong that I had to catch:** Nothing major. The header originally said Python 3.12, but everything was actually installed and tested on 3.14.7, so I corrected it.

**How I verified it:** Checked that `torch`, `ultralytics` and `cv2` import in the venv, and that `torch.backends.mps.is_available()` is `True` on my M4.

## 3. Data exploration script (`explore_data.py`)

**What I asked:** First a basic exploration script (box sizes, brightness, a sample grid, duplicate check). Then a second version focused on the questions that actually change training settings.

**What I kept vs. rewrote, and why:** The first version used arbitrary thresholds: brightness < 60 counted as "dark", one phash cut-off of 6, box area as a fraction of the image. The second version replaced these with:
- box size in pixels with COCO's size buckets, because the detector's grid is in pixels
- brightness percentiles instead of hand-picked thresholds
- a sweep over duplicate cut-offs plus a picture of the closest pairs, so the cut-off is judged by eye
- a within-train duplicate check
- a black-corner count to measure how much rotation is already in the data
- a hue comparison of mallet vs. bottle vs. background pixels

**What the AI got wrong that I had to catch:**
- In the first run both classes came out at exactly "median area 0.013, 42% under 1%", which looked like a filtering bug. I had it recompute independently: the medians are 0.0135 vs. 0.0128 and both happen to be 42.3%, so it was a coincidence, not a bug. Its first cross-check was also broken: it used `cat` on label files without trailing newlines, which glued lines together and gave wrong counts.
- It initially described the objects as "tiny" and called small objects the main difficulty. In pixels, the median object is about 58 px and only about 17% are COCO-small, so that was overstated.
- It initially said the mallet's orange colour would be an easy cue. The hue histogram showed the sand background is also orange, only about 7 hue units away. This flipped the conclusion: hue augmentation needs to stay small, not be used freely.
- A colour comment in the drawing code was wrong ("orange-ish blue"); fixed.

**How I verified it:**
- Looked at every generated plot and image grid, not just the printed numbers.
- Opened the closest duplicate pairs to decide what the phash distances actually mean (same scene vs. same photo).
- Checked what `hsv_h` does in the installed Ultralytics source (`augment.py`: shift of `± hsv_h * 180` hue units) instead of trusting the docs or the AI's description.

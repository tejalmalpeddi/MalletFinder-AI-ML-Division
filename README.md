# ML/AI Track Project

## Part 0: Intro

### Object detection

Our rover needs to autonomously identify specific objects in its environment, most notably a mallet it has to locate and retrieve as part of our competition tasks. This happens onboard, in real time, using a live camera feed, under real-world lighting and framing conditions that are messier than a clean training set.

### Classification vs. detection

- **Classification:** "is this image a mallet or not" (one label per image).
- **Detection:** "where in this image is the mallet, if it's present at all" (a bounding box plus a label). This is the harder, more useful version, and what we actually run onboard.

Detection is the closer match to our real use case and will be weighted accordingly if you attempt it (see rubric).

Our object detection stack runs YOLO, migrated from a Jetson-based pipeline to run natively on an OrangePi's onboard NPU via RKNN, specifically to cut power draw and free up compute for navigation. Model size, inference speed, and real-world robustness (varying outdoor lighting, camera angle, partial occlusion) all matter as much as raw benchmark accuracy in our actual deployment. A model that scores well on a clean validation set but falls apart outdoors isn't useful.

### Resources

- [Ultralytics YOLO docs](https://docs.ultralytics.com/) (if using YOLO)
- [PyTorch tutorials](https://pytorch.org/tutorials/) (if building a classifier from scratch)
- [Roboflow](https://roboflow.com/) (useful for quick augmentation/preprocessing pipelines, optional)

## Starter Repo / Dataset

- A labeled dataset of 999 images of a mallet and a bottle in varying backgrounds, lighting, and angles, in a standard format (YOLO-style bounding box annotations). Every image contains at least one labeled object, so you can also treat it as classification if you'd rather do that.

### What's in the repo

```
sampling/          the scripts we used to build the dataset (optional, see below)
requirements.txt   Python dependencies (keep it up to date as you add libraries)
AI_LOG.md          template for your AI usage log (Part 3)
METHODOLOGY.md     template for how to run your code and your thought process (Part 4)
```

### Setting up Python

You need Python 3.12 or newer (we tested with 3.13). From the repo folder, make a virtual environment so your libraries stay separate from the rest of your computer:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` only covers the sampling scripts. Install whatever you train with (for example `pip install ultralytics`), then add it to `requirements.txt` with its version pinned (`pip freeze` shows what you have installed).

Training is much faster on a GPU. In our tests, training YOLOv8n on the training images at 512x512 took about 2.5 minutes per epoch on a fast 8-core laptop CPU (roughly 100 minutes for 40 epochs) and about 6 seconds per epoch on an NVIDIA RTX 4070 laptop GPU (roughly 6 minutes for 40 epochs). You don't need a GPU, but on a CPU plan your time accordingly, or use a free GPU on [Google Colab](https://colab.research.google.com) (Runtime > Change runtime type > GPU).

On Windows, plain `pip install ultralytics` installs a CPU-only PyTorch. If you have an NVIDIA GPU, install the CUDA build of PyTorch first (use the selector on [pytorch.org](https://pytorch.org/get-started/locally/)), then check that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

If you train from a `.py` file on Windows, put the training code in a function and call it under `if __name__ == "__main__":`. Without that, the data loader workers crash at startup with an error about starting a new process before the current one has finished bootstrapping.

### Getting the dataset

1. Go to [the dataset's download page](https://universe.roboflow.com/malletbottle2/sampled-yd-object-detection/dataset/2/download) and click **Download Dataset**. Roboflow may ask you to sign in or create a free account.
2. Choose the format and select the option to get a **code snippet or ZIP file**. YOLOv8 matches the YOLO-style labels described above; pick another format if your framework needs it.
3. Choose **Show download code**.
4. Wait for Roboflow to finish preparing (zipping) the files, then copy the code it shows, paste it into a Python file or notebook (`.ipynb`) in your repo, and run it.

The snippet looks like this:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
version = project.version(2)
dataset = version.download("yolov8")
```

In a notebook, Roboflow's version starts with `!pip install roboflow`. In a `.py` file, run `pip install roboflow` in your terminal instead.

Running it downloads the dataset into your repository, in a folder named `Sampled-YD-Object-Detection-2`. That folder is already in `.gitignore`, so the images won't be committed.

```python
import truststore
truststore.inject_into_ssl()
```

Don't turn certificate checking off.

### What you get

```
Sampled-YD-Object-Detection-2/
  data.yaml
  train/images, train/labels     800 images
  valid/images, valid/labels     199 images
```

- There are two classes, listed in `data.yaml`: `bottle` (id 0) and `mallet` (id 1). Report your results for each class.
- Each image has a label file with one line per object: `class x_center y_center width height`, with the four numbers normalized between 0 and 1.
- License: CC BY 4.0 (also in `data.yaml`).

### The `sampling/` folder

This is how we built the dataset from our larger Roboflow project. Its dependencies are in `requirements.txt`, which covers only these scripts, not model training.

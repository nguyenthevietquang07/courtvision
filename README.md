# CourtVision

Computer-vision project for detecting and tracking basketball objects, with a YOLOv8 training notebook, labeled train/validation/test data, sample video assets, and model artifacts managed through Git LFS.

## Demo

![CourtVision annotated basketball detection demo](docs/demo/courtvision-demo.gif)

The original videos are stored with Git LFS and may not preview directly in GitHub:

- [Sample input video](sample.mp4)
- [Full annotated output video](output.mp4)

## Highlights

- YOLOv8 object detection for `ball` and `net` classes
- DeepSORT-based tracking experiments for video analysis
- Made-basket counter using an above-rim to below-rim trajectory check
- Optional near-rim event recorder for shot-attempt analysis
- Included dataset split with YOLO label files
- Notebook workflow for training, validation, and sample video inference
- Git LFS configured for large `.mp4` and `.pt` assets

## Repository Structure

```text
courtvision/
  data.yaml       YOLO dataset configuration
  scripts/        Dataset audit, training, and crossing detection scripts
  notebooks/      Colab-ready GPU workflow
  test.ipynb      Training, validation, and tracking notebook
  train/          Training images and labels
  valid/          Validation images and labels
  test/           Test images and labels
  models/         Trained CourtVision detector weights, stored with Git LFS
  sample.mp4      Sample input video, stored with Git LFS
  output.mp4      Example output video, stored with Git LFS
  yolov8x.pt      Model artifact, stored with Git LFS
```

## Setup

```bash
git lfs install
git clone https://github.com/nguyenthevietquang07/courtvision.git
cd courtvision

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

On macOS/Linux, activate the virtual environment with `source venv/bin/activate`.

## Usage

### 1. Audit the labels

```bash
python scripts/audit_dataset.py
```

Expected class names:

- `0`: ball
- `1`: net

### 2. Train a custom detector

Local CPU training is possible but slow. For the best result, open `notebooks/colab_train_and_crossings.ipynb` in Google Colab with a GPU runtime.

```bash
python scripts/train_ball_net.py --model yolov8s.pt --epochs 80 --batch 16 --imgsz 640
```

The trained model is saved at:

```text
courtvision_runs/courtvision_ball_net/weights/best.pt
```

### 3. Count confirmed baskets

```bash
python scripts/detect_crossings.py \
  --weights models/courtvision_ball_net_best.pt \
  --source sample.mp4 \
  --out-dir runs/crossings \
  --event-mode made-basket \
  --conf 0.10 \
  --ball-conf 0.15 \
  --net-conf 0.25 \
  --save-video
```

Outputs:

- `runs/crossings/crossing_events.csv` - one row per crossing event
- `runs/crossings/crossing_events.json` - structured event output
- `runs/crossings/frame_metrics.csv` - per-frame ball/net IoU and center metrics
- `runs/crossings/annotated_crossings.mp4` - optional annotated video

In the default `made-basket` mode, the counter first observes the ball above and horizontally aligned with the rim, then confirms a point only when a later detection moves below the rim within `--transition-frames`. A cooldown prevents one basket from being counted more than once.

The annotated video always displays the running point total. Ball and net confidence thresholds are configured separately because the small, fast-moving ball commonly has a lower confidence than the larger net.

For exploratory shot-attempt analysis, use `--event-mode near-rim`. That looser mode records overlap, padded-net containment, or center proximity. Near-rim events are not equivalent to made baskets.

## Dataset

The dataset uses YOLO format with two classes:

- `0`: ball
- `1`: net

`data.yaml` uses repo-relative paths so it can run locally or in hosted notebook environments after the repository is mounted.

## Notes for Reviewers

- Large media/model files are intentionally tracked with Git LFS via `.gitattributes`.
- The notebook was originally developed in Google Colab; local runs may need small path changes for notebook-only cells that reference `/content/...`.
- GPU acceleration is recommended for training.

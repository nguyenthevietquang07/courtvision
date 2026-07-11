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
- Included dataset split with YOLO label files
- Notebook workflow for training, validation, and sample video inference
- Git LFS configured for large `.mp4` and `.pt` assets

## Repository Structure

```text
courtvision/
  data.yaml       YOLO dataset configuration
  test.ipynb      Training, validation, and tracking notebook
  train/          Training images and labels
  valid/          Validation images and labels
  test/           Test images and labels
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

1. Open `test.ipynb`.
2. Confirm `data.yaml` points to the local dataset folders.
3. Run the training cells to train or fine-tune a YOLO model.
4. Run the validation cells to inspect mAP metrics.
5. Run the video tracking cells against `sample.mp4` or your own basketball footage.

## Dataset

The dataset uses YOLO format with two classes:

- `0`: ball
- `1`: net

`data.yaml` uses repo-relative paths so it can run locally or in hosted notebook environments after the repository is mounted.

## Notes for Reviewers

- Large media/model files are intentionally tracked with Git LFS via `.gitattributes`.
- The notebook was originally developed in Google Colab; local runs may need small path changes for notebook-only cells that reference `/content/...`.
- GPU acceleration is recommended for training.

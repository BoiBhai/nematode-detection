# nematode-detection

`Plan.md` describes our proposed high level workflow to detect nematodes using ML.

## Baseline: OpenCV candidate box → genus classifier

The dataset contains 1,016 1280×1024 images in 11 `Genus <name>` folders. Those
folder names are usable genus labels, but the source data does **not** include a
magnification label. Record magnification separately (for example, add a
`magnification` column to `artifacts/pseudo_boxes.csv`) if it is important.

Install the dependencies in a Python 3 environment:

```bash
python3 -m pip install opencv-python numpy torch torchvision
```

Generate one OpenCV candidate crop per image and a preview image with its green
box:

```bash
python3 detect_and_crop.py --input dataset --max-boxes 1
```

Inspect `artifacts/previews/` and correct/reject failed boxes before training.
The generated boxes are pseudo-labels, not ground truth; train a detection model
(YOLO, Faster R-CNN) only after manually reviewing them. To train the initial
crop classifier:

```bash
python3 train_crop_classifier.py --data artifacts/crops --epochs 15
```

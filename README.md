# Trainable Image Classifier

Train a small convolutional neural network (CNN) on a labeled image folder, evaluate it on a deterministic holdout, save the best validation checkpoint, and classify an image with that checkpoint.

The model learns from images you provide. There is no fixed class label, fabricated confidence score, or pretrained-weight download.

## Requirements

- Python 3.10 or newer
- PyTorch 2.6 or newer
- Pillow and NumPy

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

PyTorch installation options vary by operating system and accelerator. Use the [official PyTorch install selector](https://pytorch.org/get-started/locally/) if you need a CUDA-specific wheel.

## Prepare a labeled dataset

Use one subfolder per class. Put training images in the corresponding directory; nested folders are also scanned.

```text
data/images/
  class_a/
  class_b/
```

Each class needs at least two supported image files so the command can keep a training image and hold out a validation image. Supported formats include JPEG, PNG, BMP, GIF, TIFF, and WebP.

## Train

```bash
python classifier.py train \
  --data-dir data/images \
  --model-out artifacts/image_classifier.pt \
  --epochs 30 \
  --device auto
```

The command reports training loss and validation loss, accuracy, and macro F1 each epoch. It saves the checkpoint with the lowest validation loss and stops early when the validation loss stops improving.

## Classify an image

```bash
python classifier.py predict \
  --image incoming/photo.jpg \
  --model artifacts/image_classifier.pt \
  --top-k 3
```

The output is JSON with the highest-scoring classes and their softmax confidence. Confidence reflects this model's output; it is not a calibrated probability of correctness.

## Docker

Build the CPU-capable image and pass the training or prediction command to the entry point:

```bash
docker build -t image-classifier .
docker run --rm -v "${PWD}/data:/data" -v "${PWD}/artifacts:/artifacts" image-classifier train --data-dir /data/images --model-out /artifacts/model.pt
```

## Scope

- This is a CNN trained from scratch for the classes in your folder dataset. It does not implement ResNet transfer learning, Grad-CAM, a web UI, or benchmark-specific accuracy claims.
- The validation set is a random, stratified holdout from the same folders. Reported metrics only describe that split and do not establish performance on new domains.
- `--device auto` uses CUDA when PyTorch detects it, otherwise CPU. For reproducible production use, keep training data, class folders, and environment versions under version control.

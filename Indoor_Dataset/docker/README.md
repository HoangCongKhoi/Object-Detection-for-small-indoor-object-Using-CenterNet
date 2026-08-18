# Standard exam environment

From the directory containing the instructor `Dockerfile`, build the image once on the teacher machine:

```bash
docker build -t object-detection-exam:2026 .
```

Verify CUDA on the grading machine:

```bash
docker run --rm --gpus all object-detection-exam:2026 \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Then enter the student submission directory. The dataset mount is read-only; predictions are written to the dedicated output mount:

```bash
cd my_submission
mkdir -p grading_outputs
docker run --rm --gpus all \
  -v "$PWD/public/val/images:/exam/val_images:ro" \
  -v "$PWD:/workspace" \
  -v "$PWD/grading_outputs:/exam/outputs" \
  object-detection-exam:2026 \
  python predict.py \
    --image_dir /exam/val_images \
    --output /exam/outputs/val_predictions.json
```

From the same `my_submission` directory, evaluation intentionally runs on the host, outside the submission container:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions grading_outputs/val_predictions.json \
  --output grading_outputs/val_score.json
```

For hidden grading, replace only the read-only image mount and ground-truth path. Do not mount private annotations into the submission container.

The image contains PyTorch 2.7.1, torchvision 0.22.1, CUDA 12.6, cuDNN 9, and the exact package versions in `requirements-exam.txt`. Student `requirements.txt` files are documentation only during grading; graders do not install them.

The build context is allowlisted by `.dockerignore`. The distributed image contains no dataset pixels, annotations, source manifests, hidden mappings, evaluator, assignment statement, or dataset-builder code.

# Object Detection Final Project 2026 — Student Package

This package contains only the public train/validation dataset, annotations,
evaluation tool, assignment statement, and the standard Docker environment.
It contains no hidden-test images or labels, instructor mappings, dataset
builder, or pretrained model weights.

## Contents

```text
public/
├── classes.json
├── train/images/
├── val/images/
├── annotations/
│   ├── train.json
│   ├── val.json
│   ├── oracle_train_predictions.json
│   └── oracle_val_predictions.json
├── source_manifest.jsonl
└── tools/evaluate_predictions.py
Dockerfile
.dockerignore
docker/
├── requirements-exam.txt
└── README.md
statement.md
```

`source_manifest.jsonl` is included solely for image-license attribution.

## Build the standard environment

```bash
docker build -t object-detection-exam:2026 .
```

Read `statement.md` for the required training and prediction interfaces, and
`docker/README.md` for the exact containerized inference/evaluation workflow.

## Verify an oracle example

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions public/annotations/oracle_val_predictions.json \
  --output val_oracle_score.json
```

The expected `mAP@0.5` is `1.0`.

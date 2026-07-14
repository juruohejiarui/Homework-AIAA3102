# Assignment 2: Image Classification with PyTorch

In this assignment, you will train and compare image-classification models for four apple leaf
conditions: `healthy`, `multiple_diseases`, `rust`, and `scab`.

Your goal is to build a working pipeline, compare a manually implemented ResNet18 baseline with a
pretrained torchvision ResNet18 transfer model, and explain what you learn from your experiments.

## Data

Download `assignment2-course-data.zip` from Canvas and extract it in the project folder:

- `data/train.csv`: training labels;
- `data/validation.csv`: validation labels;
- `data/test.csv`: test image IDs without labels;
- `data/images/`: all images.

Use the provided train and validation sets for every experiment. Do not train on validation images.

## Tasks to Complete

Complete the `TODO` sections in these files:

- `src/plant_pathology/data.py`: read images and labels into a PyTorch dataset;
- `src/plant_pathology/models.py`: manual ResNet18 baseline and transfer-learning model;
- `src/plant_pathology/train.py`: training, validation, checkpoints, and experiment records;
- `src/plant_pathology/evaluate.py`: final metrics, confusion matrix, and error analysis;
- `src/plant_pathology/predict.py`: test predictions and `submission.csv`.

The detailed requirements are written at the top of each Python file. `metrics.py` and
`validate_submission.py` are provided utilities and do not need to be rewritten.

Also complete `REPORT.md`, `chat.md`, `results/experiments.csv`, and
`results/error_analysis.csv`.

Use `configs/baseline.json` as the required baseline run. The baseline must use
`model="manual_resnet18"` and implement the ResNet18 architecture manually from PyTorch layers and
residual blocks. It must train from scratch for exactly 50 configured epochs. Use
`configs/transfer.json` as the required transfer-learning run: it must use torchvision ResNet18,
load pretrained weights, and replace the classification layer.

For required comparisons, keep `image_size=128` for all runs. Do not use a higher input resolution
as a required ablation, because that changes the amount of image detail, memory use, and training
time. Save the baseline training curves as `results/training_curves_baseline.png` and include them
in `REPORT.md`. The figure must include training loss and training accuracy for each completed
epoch, and should also include validation accuracy and validation macro F1.
Run ablation experiments that change one factor at a time, and explain them clearly in the report.
These may include, but are not limited to, batch size and learning rate. Explore as many meaningful
ablation choices as you can within the compute budget. Use no more than 50 epochs for each ablation;
you may stop earlier when appropriate. Describe each choice and result in detail in `REPORT.md`.

## Submission

Submit one ZIP file containing:

```text
README.md
REPORT.md
chat.md
pyproject.toml
uv.lock
configs/baseline.json
configs/transfer.json
src/plant_pathology/__init__.py
src/plant_pathology/data.py
src/plant_pathology/models.py
src/plant_pathology/metrics.py
src/plant_pathology/train.py
src/plant_pathology/evaluate.py
src/plant_pathology/predict.py
src/plant_pathology/validate_submission.py
results/experiments.csv
results/error_analysis.csv
results/training_curves_baseline.png
results/confusion_matrix_baseline.png
results/confusion_matrix_final.png
predictions/submission.csv
tests/test_public.py
```

Submit only one `predictions/submission.csv`, produced by the model you consider your best final
model. The prediction can be uploaded only once, so validate the file and choose your model
carefully. Avoid overfitting to the validation set while tuning for higher validation accuracy.

Do not include the dataset, `.venv`, cache files, or model checkpoints. The required experiments
should fit within about six GPU-hours; start with short test runs before running full experiments.

## Assessment and Bonus

Assessment considers code correctness, model implementation, experiments, evaluation, explanation,
and code quality. Up to 3 bonus points may be awarded for one useful algorithm-level improvement
beyond the required models. To receive bonus credit, include one clearly labeled figure comparing
the method with and without your improvement. The same figure must show training and validation
accuracy across epochs for both versions. Explain the reason for the change and discuss the result
in `REPORT.md`. Bonus work does not replace required work.

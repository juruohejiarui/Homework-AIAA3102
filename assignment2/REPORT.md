# Assignment 2 Report

## Setup

- Name and student ID: Jiarui HE, 50013538
- Environment: `conda base` Python 3.12.9 with a project environment managed by `uv`
- Hardware: NVIDIA GeForce RTX 4090 D (49140 MiB VRAM), 32 CPU cores, 61 GiB system memory
- Seed: 398 for every experiment

Commands needed to run the selected final model and generate predictions/submission.csv:

conda activate base
cd assignment2
python run_all.py
uv run python -m plant_pathology.major_voting --inputs \
	predictions/submission_transfer_seed_2.csv \
	predictions/submission_transfer_scheduler.csv \
	predictions/submission_transfer_diffweight.csv \
	predictions/submission_transfer.csv \
	predictions/submission_transfer_seed_1.csv \
	--output predictions/submission_vote_top5.csv
cp predictions/submission_vote_top5.csv predictions/submission.csv
uv run python -m plant_pathology.validate_submission --test-csv data/test.csv --submission predictions/submission.csv

The experiments were run with the fixed course split: 1274 training images, 273 validation images,
and 274 unlabeled test images. The class distribution was imbalanced, especially for
`multiple_diseases`.

| Split | Healthy | Multiple diseases | Rust | Scab |
|---|---:|---:|---:|---:|
| Training | 361 | 63 | 436 | 414 |
| Validation | 77 | 14 | 93 | 89 |

All experiments below were rerun using `python run_all.py` after finalizing the training pipeline.

## Data Processing

Every image was loaded as RGB, resized to `128 x 128`, converted to a tensor, and normalized with
the ImageNet mean $(0.485, 0.456, 0.406)$ and standard deviation
$(0.229, 0.224, 0.225)$. No data augmentation was used. The required image size remained fixed at
128 for the baseline, transfer-learning model, and both ablations.

For throughput, the final configurations use 16 workers, pinned memory, persistent workers, and a
prefetch factor of 4. A direct end-to-end benchmark on the training data reached 275.0 images/s
with 16 workers, compared with 268.0 images/s using 32 workers; the lower worker count avoids
oversubscribing the CPU. CUDA runs also used non-blocking host-to-device copies, automatic mixed
precision, TF32 matrix multiplication, and cuDNN benchmarking. These execution optimizations do
not alter the model or input-resolution comparison.

In addition to training-curve figures, each run now writes a curve-data CSV file at
`results/training_curves_<run_id>.csv` containing per-epoch training loss, training accuracy,
validation accuracy, and validation macro F1.

## Models

The baseline was a manual ResNet18 implementation built from PyTorch convolution, batch
normalization, ReLU, pooling, and residual blocks. It was trained from scratch for the required 50
configured epochs using AdamW with learning rate 0.001.

The transfer model used `torchvision.models.resnet18` initialized with
`ResNet18_Weights.DEFAULT`. Its final fully connected layer was replaced with a four-class head.
The main transfer experiment fine-tuned all layers using AdamW with learning rate 0.0003 for 12
epochs. Both manual and transfer models used a dropout layer (dropout=0.5) before final
classification in the default setting, and most core experiments in this report are based on this
dropout-enabled setup.

The training code supports four toggles for controlled comparisons:

1. Class-weighted cross-entropy (`label_weights=true`) to upweight minority classes.
2. Learning-rate scheduler (`scheduler=true`) using ReduceLROnPlateau on validation macro F1.
3. Seed variants to estimate sensitivity to stochastic initialization and sampling.
4. Dropout switch (`dropout=0.5` vs `dropout=0.0`) for bonus-style with/without regularization
	comparison.

## Results and Ablations

### Full experiment leaderboard (sorted by validation macro F1)

| Run ID | Model | Batch | LR | Validation accuracy | Validation macro F1 |
|---|---|---:|---:|---:|---:|
| transfer_seed_2 | resnet18 | 128 | 0.0003 | 0.912088 | 0.831825 |
| transfer | resnet18 | 128 | 0.0003 | 0.915751 | 0.818565 |
| transfer_scheduler | resnet18 | 128 | 0.0003 | 0.915751 | 0.818565 |
| transfer_diffweight | resnet18 | 128 | 0.0003 | 0.908425 | 0.810304 |
| transfer_seed_1 | resnet18 | 128 | 0.0003 | 0.904762 | 0.795870 |
| transfer_no_dropout | resnet18 | 128 | 0.0003 | 0.908425 | 0.788928 |
| transfer_lr_0.0001 | resnet18 | 128 | 0.0001 | 0.919414 | 0.764835 |
| baseline_no_dropout | manual_resnet18 | 128 | 0.0010 | 0.871795 | 0.763114 |
| baseline_seed | manual_resnet18 | 128 | 0.0010 | 0.849817 | 0.727923 |
| baseline_batch | manual_resnet18 | 64 | 0.0010 | 0.853480 | 0.727152 |
| baseline_scheduler | manual_resnet18 | 128 | 0.0010 | 0.868132 | 0.717728 |
| baseline_seed_2 | manual_resnet18 | 128 | 0.0010 | 0.860806 | 0.708579 |
| baseline | manual_resnet18 | 128 | 0.0010 | 0.857143 | 0.688385 |
| baseline_diffweight | manual_resnet18 | 128 | 0.0010 | 0.578755 | 0.509335 |
| baseline_lr_0.0001 | manual_resnet18 | 128 | 0.0001 | 0.597070 | 0.499932 |
| transfer_frozen | resnet18 | 128 | 0.0003 | 0.637363 | 0.490575 |

### Main findings

1. Transfer learning clearly dominates manual training from scratch on this dataset.
2. Dropout is a central regularization choice in this project, and its effect is
	architecture-dependent: for transfer, dropout improves macro F1
	(`transfer` 0.818565 vs `transfer_no_dropout` 0.788928), while for the manual baseline,
	removing dropout performs better (`baseline_no_dropout` 0.763114 vs `baseline` 0.688385).
3. Fully fine-tuning the transfer backbone is crucial; freezing it causes a major macro F1 drop.
4. For manual ResNet18, lowering learning rate to 0.0001 underfits badly within 50 epochs.
5. Class weighting is helpful for transfer (`transfer_diffweight`) but harmful for the manual
	baseline (`baseline_diffweight`), indicating architecture-dependent optimization behavior. The combination of the limited dataset size and class weighting causes the manual baseline to over-focus on minority classes, preventing it from learning sufficiently rich patterns. At the same time, it loses attention to the major classes.
6. Scheduler impact is model-dependent. For the manual baseline, enabling
	ReduceLROnPlateau improved macro F1 from 0.688385 (`baseline`) to 0.717728
	(`baseline_scheduler`) and validation accuracy from 0.857143 to 0.868132.
	This suggests the baseline benefits from adaptive step-size reduction when
	training dynamics plateau. In contrast, `transfer_scheduler` matched the
	plain `transfer` run (same accuracy and macro F1), indicating the
	pretrained backbone plus short fine-tuning horizon was already in a stable
	optimization regime where additional LR decay brought little extra gain.
7. Seed variation is non-trivial. The best single run is `transfer_seed_2` with macro F1 0.831825.

Representative training curves:

![Baseline training curves](results/training_curves_baseline.png)
![Transfer training curves](results/training_curves_transfer.png)

## Evaluation and Error Analysis

The baseline checkpoint achieved validation accuracy 0.857143 and macro F1 0.688385. Its recalls
were 0.974026 for `healthy`, 0.071429 for `multiple_diseases`, 0.903226 for `rust`, and 0.820225
for `scab`.

![Baseline confusion matrix](results/confusion_matrix_baseline.png)

The selected final validation strategy (top-5 major voting) achieved validation accuracy 0.919414
and macro F1 0.858810. Its recalls were 0.948052 for `healthy`, 0.571429 for
`multiple_diseases`, 0.978495 for `rust`, and 0.887640 for `scab`.

This confirms that transfer learning, careful optimization, and seed selection materially improve
minority-class behavior relative to the baseline.

![Final-model confusion matrix](results/confusion_matrix_final.png)

`results/error_analysis_final_vote_top5.csv` records eight incorrect validation examples for the
selected final voting model. They form two recurring error types:

1. `multiple_diseases` confusions (5/8): this minority class is still the dominant failure mode,
	with mistakes spread across `rust`, `scab`, and `healthy`.
2. Boundary confusions among `scab`, `healthy`, and `rust` (3/8): subtle lesion severity remains
	hard to separate at 128 resolution, even after ensembling.

## Major Voting Ensemble

To justify ensemble use with directly labeled evidence, I added a validation-set voting evaluation
pipeline (`python -m plant_pathology.validate_voting`) that:

1. Generates validation predictions for each selected checkpoint.
2. Computes single-model metrics on the same validation split.
3. Builds a hard-vote ensemble and reports its validation metrics in the same format.

Two candidate ensembles were tested:

1. Top-3 by validation macro F1:
	`transfer_seed_2`, `transfer_scheduler`, `transfer_diffweight`.
2. Top-5 diverse transfer set:
	`transfer_seed_2`, `transfer_scheduler`, `transfer_diffweight`, `transfer`, `transfer_seed_1`.

The validation-set comparison for the top-5 vote is:

- top-5 vote: accuracy 0.919414, macro F1 0.858810
- best single model (`transfer_seed_2`): accuracy 0.912088, macro F1 0.831825

So the vote improves macro F1 by 0.026985 and especially improves the minority
`multiple_diseases` recall from 0.500000 to 0.571429.

The same validation-only comparison was also run for top-3 members.

- top-3 vote: accuracy 0.915751, macro F1 0.835025
- top-5 vote: accuracy 0.919414, macro F1 0.858810

Therefore, the final submission for this report uses `submission_vote_top5.csv`.

Reasonable generalization argument:

1. The five transfer members are high-performing but make partially different errors due to seed,
	learning dynamics, and weighting variants.
2. Hard voting reduces individual model variance by canceling idiosyncratic mistakes when members
	disagree, while preserving consensus on easy samples.
3. The observed gain appears exactly where expected: minority-class recall and macro F1 improve
	more than overall accuracy, which matches the objective of class-balanced robustness.

## Test Prediction

The final file `predictions/submission.csv` is the top-5 majority-voting output. It contains all
274 test image IDs exactly once and the probability columns `healthy`, `multiple_diseases`, `rust`,
and `scab`. The provided validator completed successfully.

Commands used for the final prediction pipeline:

```bash
conda activate base
cd assignment2
python run_all.py

uv run python -m plant_pathology.major_voting \
  --inputs \
	 predictions/submission_transfer_seed_2.csv \
	 predictions/submission_transfer_scheduler.csv \
	 predictions/submission_transfer_diffweight.csv \
	 predictions/submission_transfer.csv \
	 predictions/submission_transfer_seed_1.csv \
  --output predictions/submission_vote_top5.csv

cp predictions/submission_vote_top5.csv predictions/submission.csv

uv run python -m plant_pathology.validate_submission \
  --test-csv data/test.csv \
  --submission predictions/submission.csv
```

## Bonus Improvement

The bonus improvement is dropout regularization in the transfer-model classification head. The
comparison is between:

1. With improvement: `transfer` (dropout = 0.5).
2. Without improvement: `transfer_no_dropout` (dropout = 0.0).

Motivation: the dataset is relatively small (1274 training images) and imbalanced (only 63
`multiple_diseases` samples), so overfitting is a realistic risk. Dropout is used as an
algorithm-level regularization method to improve generalization.

Bonus comparison figure (the same figure includes training/validation accuracy and validation
macro F1 across epochs for both versions):

![Bonus dropout comparison](results/bonus_dropout_comparison.png)

Observed results from the same with-vs-without comparison:

1. Transfer model: with dropout reaches validation accuracy/macro F1 of 0.915751/0.818565,
   compared with 0.908425/0.788928 without dropout.
2. Manual baseline model: with dropout reaches validation accuracy/macro F1 of 0.857143/0.688385,
   while without dropout reaches 0.871795/0.763114.

The figure shows that the dropout effect is architecture-dependent in this project. For the
transfer model, dropout improves validation behavior while slightly constraining training accuracy,
which is consistent with overfitting control. For the manual baseline, removing dropout yields
stronger validation metrics under the current training budget, suggesting that heavy regularization
can also underfit when representation capacity and optimization stability are limited.

## Conclusion

The complete rerun confirms that transfer learning is essential for this task. The best single
validation run (`transfer_seed_2`) reached macro F1 0.831825, substantially above the manual
baseline. Freezing the transfer backbone or under-tuning the learning rate causes clear degradation.
Dropout is a major regularization lever throughout this work: it helps the transfer model but does
not help the manual baseline under the same training budget, which highlights architecture-dependent
regularization behavior. Class weighting helps transfer models more than manual models, and seed
variance remains non-negligible.

For final submission robustness, majority voting across top-performing transfer variants was used.
The selected top-5 ensemble produced the strongest validation-set macro F1 among tested ensembles
and single models, and was validated as the final `submission.csv`. The remaining technical challenge is the
`multiple_diseases` class; future high-impact work should target class-specific feature
disambiguation and more robust minority-class training strategies.

## References

- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- torchvision ResNet documentation: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- scikit-learn metrics documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Course assignment handout and starter code

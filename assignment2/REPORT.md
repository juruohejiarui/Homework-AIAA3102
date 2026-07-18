# Assignment 2 Report

## Setup

- Name and student ID: [Fill in your name and student ID]
- Environment: `conda base` Python 3.12.9 with a project environment managed by `uv`
- Hardware: NVIDIA GeForce RTX 4090 D (49140 MiB VRAM), 32 CPU cores, 61 GiB system memory
- Seed: 398 for every experiment

The experiments were run with the fixed course split: 1274 training images, 273 validation images,
and 274 unlabeled test images. The class distribution was imbalanced, especially for
`multiple_diseases`.

| Split | Healthy | Multiple diseases | Rust | Scab |
|---|---:|---:|---:|---:|
| Training | 361 | 63 | 436 | 414 |
| Validation | 77 | 14 | 93 | 89 |

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

## Models

The baseline was a manual ResNet18 implementation built from PyTorch convolution, batch
normalization, ReLU, pooling, and residual blocks. It was trained from scratch for the required 50
configured epochs using AdamW with learning rate 0.001.

The transfer model used `torchvision.models.resnet18` initialized with
`ResNet18_Weights.DEFAULT`. Its final fully connected layer was replaced with a four-class head.
The main transfer experiment fine-tuned all layers using AdamW with learning rate 0.0003 for 12
epochs. The best transfer checkpoint occurred at epoch 8, selected by validation macro F1.

## Results and Ablations

| Run ID | Model | Main change | Epochs | Batch size | Learning rate | Validation accuracy | Validation macro F1 | Best epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | manual_resnet18 | trained from scratch | 50 | 128 | 0.0010 | 0.798535 | 0.689379 | 42 |
| transfer_resnet18 | resnet18 | pretrained, all layers fine-tuned | 12 | 128 | 0.0003 | 0.901099 | 0.750064 | 8 |
| transfer_frozen | resnet18 | freeze backbone only | 12 | 128 | 0.0003 | 0.666667 | 0.512983 | 12 |
| transfer_lr_0.0001 | resnet18 | learning rate changed only | 12 | 128 | 0.0001 | 0.871795 | 0.701594 | 9 |

The required transfer model improved validation accuracy by 0.102564 and macro F1 by 0.060685 over
the manual baseline. It therefore provided a better overall class-balanced result while requiring
far fewer epochs.

The frozen-backbone ablation changed only `freeze_backbone` from `false` to `true`. Its macro F1
dropped from 0.750064 to 0.512983, showing that the ImageNet features required adaptation to this
leaf-disease dataset. The lower-learning-rate ablation changed only the learning rate from 0.0003
to 0.0001. It reached macro F1 0.701594, which was better than the scratch baseline but below the
default transfer learning rate. The larger learning rate converged to the stronger validation score
within the 12-epoch budget.

![Baseline training curves](results/training_curves_baseline.png)

## Evaluation and Error Analysis

The baseline checkpoint achieved validation accuracy 0.798535 and macro F1 0.689379. Its recalls
were 0.818182 for `healthy`, 0.214286 for `multiple_diseases`, 0.881720 for `rust`, and 0.786517
for `scab`.

![Baseline confusion matrix](results/confusion_matrix_baseline.png)

The selected final model achieved validation accuracy 0.901099 and macro F1 0.750064. Its recalls
were 0.961039 for `healthy`, 0.142857 for `multiple_diseases`, 0.946237 for `rust`, and 0.921348
for `scab`. The smallest class, `multiple_diseases`, was the lowest-recall class in both models,
which is consistent with its limited training and validation support.

![Final-model confusion matrix](results/confusion_matrix_final.png)

`results/error_analysis.csv` records eight incorrect validation examples from the selected final
model. They form two recurring error types:

1. The three `multiple_diseases` examples were predicted as `rust`, `scab`, or `healthy`. This
	 suggests that mixed symptoms were difficult to distinguish from a dominant single-disease
	 pattern, and the minority class had insufficient examples.
2. Four errors were reciprocal `healthy`/`scab` confusions, with two errors in each direction.
	 This indicates that mild or localized scab symptoms can resemble healthy leaf texture at the
	 fixed 128-pixel input size. One remaining error was `rust` predicted as `scab`, another likely
	 texture-level confusion between visually related lesions.

## Test Prediction

The final file `predictions/submission.csv` was generated with `transfer_resnet18`, using
`configs/transfer.json` and `checkpoints/transfer_resnet18_best.pt`. The file contains all 274 test
image IDs exactly once and the probability columns `healthy`, `multiple_diseases`, `rust`, and
`scab`. The provided validator completed successfully; the observed probability-row sums ranged
from 0.99999988 to 1.00000012, within the checker tolerance.

Commands used for the final prediction:

```bash
conda activate base
cd assignment2
uv run python -m plant_pathology.predict \
	--config configs/transfer.json \
	--checkpoint checkpoints/transfer_resnet18_best.pt \
	--output predictions/submission.csv
uv run python -m plant_pathology.validate_submission \
	--test-csv data/test.csv \
	--submission predictions/submission.csv
```

## Bonus Improvement

Not attempted. The two ablations were used to satisfy the required controlled comparisons and did
not introduce an additional algorithmic method beyond the required models.

## Conclusion

The pretrained, fully fine-tuned ResNet18 was the best model in this study, reaching 0.901099
validation accuracy and 0.750064 macro F1. Transfer learning improved both metrics relative to the
manual ResNet18 trained from scratch. Freezing the backbone substantially reduced validation macro
F1, while reducing the fine-tuning learning rate also produced a weaker result. The key remaining
limitation is `multiple_diseases`: it was the smallest class and remained difficult for every model,
so future work should prioritize a class-focused algorithmic improvement such as weighted loss or
targeted augmentation.

## References

- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- torchvision ResNet documentation: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- scikit-learn metrics documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Course assignment handout and starter code

# Assignment 2 Report

## Setup

- Name and student ID:
- Hardware used:
- Commands needed to run your final model:

## Data Processing

Describe image resizing, normalization, augmentation, and any other preprocessing. Confirm that
required comparisons used `image_size=128`.

## Models

Describe the required baseline as a manually implemented ResNet18 trained from scratch for 50
configured epochs. Describe the torchvision transfer-learning ResNet18 and explain which pretrained
layers were frozen or fine-tuned.

## Results and Ablations

Summarize all experiments in a table. In the written report, describe each ablation, the factor you
changed, why you chose it, and what you learned. Ablations may include, but are not limited to,
batch size and learning rate. Explore as many meaningful choices as you can within the compute
budget and discuss them in detail.

![Baseline training curves](results/training_curves_baseline.png)

## Evaluation and Errors

Report the required validation metrics and confusion matrices. Summarize the main patterns found
in your eight error examples.

## Test Prediction

State which model produced `predictions/submission.csv` and confirm that you ran the submission
checker. Explain in writing why you selected this model. The prediction can be uploaded only once,
so choose carefully and avoid overfitting to the validation set.

## Bonus Improvement (Optional)

Describe the algorithm-level improvement and why you expected it to help. Include one figure with
the original and improved methods together, showing both training and validation accuracy across
epochs for each method. Explain the reason for the change and the observed result in writing.

## Conclusion

Give two conclusions supported by your results and one limitation of your work.

## References

Cite documentation, tutorials, external code, pretrained weights, and other resources you used.

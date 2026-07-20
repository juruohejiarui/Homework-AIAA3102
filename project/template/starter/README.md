# Topic A Starter: Text Classification Pipeline Forensics

This package intentionally contains no baseline code. It gives the public data source, fixed split, and
submission interface for Topic A. You are expected to download the data and implement the pipeline yourself.

## Public Data Source

Use the public Kaggle Disaster Tweets files mirrored at:

```text
https://github.com/ucbrise/kaggle-nlp-disasters/tree/master/data
```

The assignment uses the full labeled `train.csv`. The Kaggle `test.csv` and `sample_submission.csv` are source
context only and are not used for the course split.

## Fixed Split

Use `data/split_indices.json` to assign the Kaggle `id` values to train, dev, and held-out splits. Do not
regenerate the split.

- `train_ids`: fit final models.
- `dev_ids`: choose preprocessing, thresholds, and model variants.
- `heldout_ids`: report finished-ticket results after decisions are frozen.

## Reference Contract

`configs/project_contract.json` gives the reference held-out metric for the course baseline. The project asks
you to diagnose whether your independently implemented baseline reproduces that contract, and if not, why.

## Expected Student Code

Create your own scripts or notebooks for:

- downloading or placing the public CSV files;
- loading the fixed split;
- implementing the floor model and TF-IDF + Logistic Regression baseline;
- running ticket experiments;
- exporting prediction and result artifacts.

Recommended CPU packages include pandas and scikit-learn, but the starter does not require a particular file
layout or command structure.

## Artifact Interface

Your final submission should include machine-checkable artifacts with stable ids. Use these column conventions
unless your report clearly documents an equivalent format:

```text
heldout_predictions.csv:
id,y_true,y_pred,score,model_name,ticket

results/summary.csv:
ticket,model_name,dev_f1_target_1,heldout_f1_target_1,heldout_accuracy,fixed_fp,fixed_fn,new_fp,new_fn,decision,decision_reason

results/threshold_sweep.csv:
ticket,threshold,precision_target_1,recall_target_1,f1_target_1

results/data_quality_audit.csv:
id,issue_type,evidence,disposition,confidence
```

Valid data-quality dispositions are `fix`, `keep_but_flag`, `ambiguous`, and `reject_false_positive`.

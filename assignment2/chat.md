# AI and Tool Usage

List the AI tools, tutorials, repositories, or substantial external help used in this assignment.

| Tool or source | What I used it for | What I checked or changed |
|---|---|---|
| GitHub Copilot | Audited assignment requirements, implemented the training pipeline, and drafted the report. | Implemented the dataset, manual ResNet18, transfer model, training, evaluation, prediction, and validation workflow. It also generated configurations for controlled ablations, ran the required experiments, and summarized measured results in the report. |
| PyTorch documentation | Confirmed expected APIs for datasets, training loops, optimizers, and checkpoint loading. | Used as a reference for implementation planning and for the report references section. |
| torchvision ResNet18 documentation | Confirmed pretrained weight loading and final-layer replacement for transfer learning. | Used to verify the required transfer-learning model design. |

Final scope of AI assistance:

- used `uv` with the conda base Python environment to run `python run_all.py` over all 14 config
  files
- updated the training pipeline to export curve data CSV files in addition to curve PNG files
- generated per-run checkpoints, per-run confusion matrices, per-run error-analysis CSV files, and
  per-run prediction files
- compared all experiment results from `results/experiments.csv` and selected top transfer variants
  for ensemble testing
- ran majority-voting ensembles (top-3 and top-5) and selected the better one as final
  `predictions/submission.csv`
- updated the report with full rerun metrics, ablation findings, error-pattern analysis, and final
  submission rationale

One useful suggestion: compare single-model and ensemble behavior on the validation split before
finalizing `submission.csv`, instead of assuming the validation-best single model always gives the
strongest final behavior.

One suggestion that I rejected or corrected: increasing worker count or batch size indefinitely to
raise GPU utilization. Measured throughput showed over-allocation can reduce end-to-end speed, so
the final settings were chosen from real benchmarks.

# AI and Tool Usage

List the AI tools, tutorials, repositories, or substantial external help used in this assignment.

| Tool or source | What I used it for | What I checked or changed |
|---|---|---|
| GitHub Copilot | Audited assignment requirements, implemented the training pipeline, and drafted the report. | Implemented the dataset, manual ResNet18, transfer model, training, evaluation, prediction, and validation workflow. It also generated configurations for controlled ablations, ran the required experiments, and summarized measured results in the report. |
| PyTorch documentation | Confirmed expected APIs for datasets, training loops, optimizers, and checkpoint loading. | Used as a reference for implementation planning and for the report references section. |
| torchvision ResNet18 documentation | Confirmed pretrained weight loading and final-layer replacement for transfer learning. | Used to verify the required transfer-learning model design. |

Additional scope of AI assistance in this draft:

- verified that the work was run in the `aiaa3102` conda environment
- checked that `pytest -q` passes but does not prove assignment completion
- checked that the required pipeline files still contain unimplemented `TODO` sections
- checked that required artifacts such as confusion matrices and `predictions/submission.csv` have
	not yet been generated
- drafted a step-by-step completion order in `REPORT.md` without modifying `README.md`

Final verification and reporting assistance:

- used `uv` with the conda base Python environment to run public tests, training, evaluation, and
	prediction commands
- added data-loading and CUDA throughput settings, including pinned memory, worker prefetching,
	AMP, and TF32, then checked the public tests again
- ran the required manual baseline, pretrained transfer model, frozen-backbone ablation, and
	lower-learning-rate ablation
- generated the baseline and final confusion matrices, eight-row error-analysis CSV, and validated
	test submission
- replaced report placeholders with measured experiment results and documented the selected model

One useful suggestion: the AI assistant suggested turning the starter README into a step-by-step
submission checklist, which is useful because it makes the missing deliverables easy to verify
before zipping the final submission.

One suggestion that I rejected or corrected: passing `pytest` alone is not enough to claim the
assignment is complete. The public tests only cover a small subset of the project, so I corrected
that assumption and separately checked the repository for unfinished TODO sections and missing
required output files.

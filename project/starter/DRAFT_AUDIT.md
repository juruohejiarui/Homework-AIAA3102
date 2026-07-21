# Topic A Final Draft Audit

## 审核范围与结论

本复审以题目 handout、starter artifact interface 和当前提交内容为准，检查五个 ticket、实验流程、CLI、冻结记录、CSV artifacts、正式报告、PDF、AI work record 和测试。

结论：**当前草稿满足题目要求的主要提交与证据链要求，可以进入提交前的最终检查。** 先前审计发现的工程问题均已修复：单 ticket 的 held-out 隔离、Ticket 5 缺少 `reject_false_positive` disposition，以及 Ticket 1 探针的使用边界。Ticket 1 现在采用固定的冻结后 forensic matrix：probes 在 baseline freeze 后各评估一次 held-out，只用于检查 dev/held-out F1 delta 的一致性，不用于选模。报告仍应保留单一 dev split、多重选择和未知 reference configuration 的谨慎表述；这些是研究局限，不是实现错误。

## 已执行验证

以下命令在 `aiaa3102` conda 环境中运行成功：

```bash
conda run -n aiaa3102 python -m pytest -q
conda run -n aiaa3102 python -m pipeline.cli validate-data
conda run -n aiaa3102 python -m pipeline.cli run-all
conda run -n aiaa3102 python -m pipeline.cli validate-artifacts
```

结果：

- **60 tests passed**。
- 数据校验确认固定 split 为 train/dev/held-out = 4,567/1,523/1,523，正例数为 1,962/655/654。
- artifact validator 从 7,615 条 held-out prediction rows 重建 5 行 `summary.csv` 的 F1、accuracy 与相对于 Ticket 1 baseline 的四类 transition counts。
- [report.pdf](report.pdf) 由 [REPORT.md](REPORT.md) 重建，为 12 页、未加密 PDF。
- `run-ticket --ticket 1 --split dev` 不会重写汇总 [predictions/heldout_predictions.csv](predictions/heldout_predictions.csv)；pending decision 经 `freeze-ticket` 后才允许 held-out 命令。

## 要求覆盖

| 题目要求 | 状态 | 当前证据 |
| --- | --- | --- |
| 五个 ticket 有 hypothesis、lever、controlled setup、dev/held-out evidence、stable-ID examples 与 limitation | 通过 | [tickets/](tickets/) 与 [REPORT.md](REPORT.md) |
| 固定 split，train-only 拟合 | 通过 | [pipeline/data.py](pipeline/data.py)、[pipeline/features.py](pipeline/features.py)、测试 |
| 在 dev 选择、freeze 后评估 held-out | 通过 | [pipeline/cli.py](pipeline/cli.py)、[pipeline/experiments.py](pipeline/experiments.py) |
| 机器可检查的 prediction、summary、threshold 与 audit artifacts | 通过 | [pipeline/artifacts.py](pipeline/artifacts.py)、[results/](results/) |
| transition 统一相对 Ticket 1 frozen baseline | 通过 | [results/summary.csv](results/summary.csv)、artifact validator |
| Ticket 1 discrepancy diagnosis | 通过，但结论有边界 | [results/discrepancy_comparison.csv](results/discrepancy_comparison.csv)、[results/discrepancy_association.json](results/discrepancy_association.json) |
| Ticket 2 normalization mechanism 与 stress evidence | 通过 | [results/perturbation_stress.csv](results/perturbation_stress.csv) |
| Ticket 3 shortcut audit 在 metadata-consuming models 上干预 | 通过 | [results/perturbation_stress.csv](results/perturbation_stress.csv) |
| Ticket 4 precision/recall tradeoff 与 ablation | 通过 | [results/decision_ablation.csv](results/decision_ablation.csv) |
| Ticket 5 保留 held-out labels 并区分 audit dispositions | 通过 | [results/data_quality_audit.csv](results/data_quality_audit.csv) |
| 自包含报告、Difficulties and Solutions、AI declaration 和 work record | 通过 | [REPORT.md](REPORT.md)、[report.pdf](report.pdf)、[logs/chat.md](logs/chat.md) |

## Ticket 1 冻结后诊断

Ticket 1 需要诊断与 course reference 的差距；题目允许 held-out 作为诊断 artifact，但不允许它成为 preprocessing、threshold 或 model-variant 的选择器。当前流程如下：

1. assignment-driven TF-IDF + Logistic Regression baseline 在 dev 上确定并冻结；
2. 26 个单因素 probes 作为固定 forensic matrix，在 baseline freeze 后各评估一次 held-out；
3. probe 结果只用于比较 dev/held-out delta 是否一致，不能替换 baseline、选择后续 ticket 配置或按 held-out F1 排名。

该 policy 写入 [experiments/decisions.json](experiments/decisions.json)。[results/discrepancy_association.json](results/discrepancy_association.json) 记录 26 个 probes 的 F1 delta 关联：Pearson $r=0.996813$、Spearman $\rho=0.874915$。这支持主要 dev movement 在 held-out 上通常同向出现；但 probes 共享数据、baseline 与模型族，并非独立观测，故关联与 p-value 均不作为因果或 confirmatory inference。两个 dev 后、held-out 前固定的代表 probes 还导出完整 stable-ID transition rows 与 counts，见 [results/discrepancy_error_transitions.csv](results/discrepancy_error_transitions.csv) 和 [results/discrepancy_transition_summary.csv](results/discrepancy_transition_summary.csv)。

这比“只看 held-out 最佳 F1”更严谨，也比丢弃全部 held-out diagnosis 更符合 Ticket 1 的取证目的。baseline held-out F1 仍为 0.731984，未匹配 contract reference 0.757422；没有 probe 被认定为 reference reproduction 或替换 baseline。

## 已修复的工程与实验问题

### Dev/freeze/held-out 协议

`run-ticket --split dev` 仅构造 train/dev 分区、写入 dev prediction 和 pending decision。`freeze-ticket` 将该 record 写入 frozen decisions。`run-ticket --split heldout` 要求 frozen decision，并只写对应 ticket 的 held-out prediction。回归测试覆盖 dev CLI 不分派到 held-out runner、缺少 pending decision 时不可 freeze、缺少 frozen decision 时不可 held-out。

### Ticket 5 audit disposition

ID 198 的高置信模型误报现在标记为 `reject_false_positive`：文本讨论的是终生飞机事故风险而非正在发生的事件，人工语义复核支持原始标签 0。该审计发现被拒绝，但标签不被修改。duplicate 的 `keep_but_flag`、冲突证据的 `ambiguous` 和 rejected finding 均在报告中有 stable-ID 示例。

## 实验严谨性复审

- **Ticket 1**：完整 matrix 保留了正、负对照和 held-out evidence，并以 freeze policy 防止其成为选模。它支持实现因素与分数变化有关，但不能识别未公开 reference configuration。
- **Ticket 2**：raw baseline 和 selected URL model 都接受 surface perturbation；URL token 的有限机制结论及 punctuation 反例均有证据。
- **Ticket 3**：metadata interventions 施加于实际消费 keyword/location 的模型，证明确实存在 field reliance，而非在 text-only model 上进行无信息量测试。
- **Ticket 4**：模型族、$C$、class weight、threshold 均在 dev 上选择；ablation 将组合增益与单一 lever 区分开，结论不归因于单项改动。
- **Ticket 5**：未编辑 held-out labels；`fix=0` 是合理的保守结果，因为没有独立人工标注支持自动 correction。

## 仍需诚实保留的限制

- 单一 dev split 同时承担多个 feature、model 和 threshold 的选择，微小 F1 差异不应视为稳定提升。
- Ticket 1 probes 并非独立实验样本；dev/held-out delta 关联不构成因果证明。
- Ticket 2/3 的 perturbation 是 synthetic intervention，不等同于真实部署分布。
- Ticket 5 的语义复核不是独立双人标注，near-duplicate threshold 也是启发式定义。
- 未公开的 contract reference configuration 使 Ticket 1 只能给出有边界的诊断，不能做因果复现声明。

## 提交前最终检查

- [x] `report.pdf` 存在且由当前 `REPORT.md` 生成。
- [x] `logs/chat.md` 是真实摘要，而不是伪造逐字对话。
- [x] README 的单 ticket 命令说明与 CLI 一致。
- [x] CSV 数字、ticket narrative 和 report 的结论相互一致。
- [x] held-out labels 未被编辑，Ticket 5 保留 audit rationale。
- [x] 最终运行已通过 tests、data validation 与 artifact validation。

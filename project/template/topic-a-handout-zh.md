# Final Project - Topic A：文本分类管线取证分析

## Text Classification Pipeline Forensics

*诊断一个带噪声的灾害推文分类器，复现参考 baseline，并解释哪些管线改动真的可信。*

### I. 背景

文本分类模型常常看起来比实际更“可解释”。一个模型的分数可能提高，是因为预处理去掉了噪声，也可能是因为阈值刚好适配了指标，还可能是因为数据集中存在无法泛化的 shortcut。单纯的分数变化不是完整答案。你需要知道哪些样本的预测发生了变化，哪些错误被修复，哪些新错误被引入，以及这些变化是否符合你的原始假设。

在本项目中，你将构建并调查一个 CPU-only 的 Kaggle Disaster Tweets 文本分类管线。任务是二分类：`target=1` 表示推文描述真实灾害，`target=0` 表示不是。数据源是公开的；starter package 提供固定 split 和 reference contract，但你需要自己下载数据、实现 baseline 和实验。

### II. 项目概览

本项目由五个 investigation tickets 组成。每个 ticket 都要求你测试一种管线或数据质量假设，并用证据解释结果。可见 held-out 分数只是诊断证据之一，不是唯一答案。

starter package 包含：

- Disaster Tweets 完整 labeled `train.csv` 的公开来源；
- 包含固定 `train_ids`、`dev_ids`、`heldout_ids` 的 `split_indices.json`；
- 包含参考 baseline 指标和 tolerance 的 `configs/project_contract.json`。

held-out split 只能在某个 ticket 的决策已经冻结后使用。预处理设置、阈值、模型变体等选择应基于 dev split。不要移动任何 id 到其他 split。

### III. Starter 包结构

starter package 提供固定 split、reference contract 和 artifact interface。请使用其中的 `README.md` 查看 data-source links、split 使用方式和 output-format expectations。下载、读取、建模、评估和审计代码都由你自己实现。

### IV. 项目内容

你必须完成下面五个 tickets。每个 ticket 都应包含 hypothesis、一个明确的 intended lever、controlled experiment、dev evidence、最终 held-out evidence、具体样本，以及一个 limitation。

| # | Ticket | 核心问题 |
|---|---|---|
| 1 | Baseline discrepancy diagnosis | 你的 TF-IDF + Logistic Regression baseline 是否匹配参考结果？如果不匹配，是 split、seed、版本还是 preprocessing 差异导致 gap？如果匹配，哪两个 discrepancy probes 能证明参考结果为什么可复现？ |
| 2 | Text normalization lever | URL、mention、hashtag、标点、大小写或 emoji 处理会帮助还是伤害模型？哪些 false positives 和 false negatives 发生了变化？这些变化是否符合你的假设？ |
| 3 | Feature and shortcut audit | `keyword`、长度或其他浅层 artifacts 提供了多少信号？这些信号是合理任务信息、数据集 artifact，还是两者混合？ |
| 4 | Decision-rule and model ticket | threshold tuning、class weighting、regularization 或第二个 CPU classifier 是否改善 operating point？请解释 precision-recall tradeoff，而不是只报告最佳 F1。 |
| 5 | Data-quality and error ticket | 哪些 duplicates、疑似 mislabels、ambiguous tweets 或 hard negatives 限制了分数？请区分应修复、应保留但标记、属于 ambiguous policy case，以及应拒绝的 false-positive audit findings。 |

对于 Ticket 5，不要修改 held-out labels，也不要删除 held-out examples。如果你测试 training-label correction，请在 audit table 中同时保留原始 label 和 proposed label。

### V. 预期证据

你需要设计能支持结论的实验，而不是只产生一个不同的分数。你的证据应让读者理解：什么发生了变化，为什么变化，以及这个变化是否可信。

你应保留足够的 artifacts，使结果能够被重新生成和检查。starter README 会说明 required prediction files、summary tables 和任何 machine-checkable schemas。

Hidden grading 可能会运行 stress variants：它们保持 label 不变，但扰动 URLs、大小写、hashtags、mentions、metadata-like shortcuts 等表层文本信号。强提交应解释每个信号到底是合理任务信息、artifact，还是混合证据，而不是只针对可见 held-out split 调参。

### VI. 报告要求

最终的 `report.pdf` 应总结你的 methodology、analysis、key findings 和 conclusions。它应该是一份自洽的项目报告，而不是命令输出的集合。

一份优秀报告应清楚说明你研究了什么问题，为什么选择这些实验，哪些证据支持你的结论，以及哪些地方仍然不确定或有限制。对于本 topic，报告应让读者相信：你的实验来自明确假设，实验设置是 controlled 的，结论是通过具体 prediction changes 解释的，而不仅仅是通过分数变化解释的。

报告中也应设置单独的 **Difficulties and Solutions** section，说明至少三个具体、可验证的项目困难，以及你如何解决它们。请包含 AI usage declaration，说明你如何使用 AI 工具，以及如何验证 AI 输出。

如果提交只列出命令和 F1 分数，即使代码能运行，也不能获得高分。

### VII. 提交内容

```
repo/
|-- pipeline/
|-- tickets/
|   |-- ticket-1-baseline.md
|   |-- ticket-2-normalization.md
|   |-- ticket-3-shortcuts.md
|   |-- ticket-4-decision-rule.md
|   `-- ticket-5-data-quality.md
|-- experiments/
|-- predictions/
|-- results/
|-- logs/chat.md
|-- report.pdf
`-- README.md
```

`README.md` 必须包含能重新生成主要结果的命令。

### VIII. 学术诚信与 AI 使用

允许并鼓励使用 AI 工具。你必须引用 datasets、notebooks、libraries 和 tools。复现 reference baseline 是允许的；但不能把未验证的 public code 或 AI output 当作自己的分析结论提交。

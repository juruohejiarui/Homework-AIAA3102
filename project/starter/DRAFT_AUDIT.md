# Topic A 草稿审核与高分补强清单

审核对象是本目录中的草稿实现和产物；原始要求见[题目 handout](../template/topic-a-handout-zh.md)与[英文 handout](../template/topic-a-handout.md)。本文不替代最终的 `report.pdf`，而是用于完成提交前逐项修订。

## 审核范围与结论

已检查的内容包括五个 ticket、`pipeline/`、实验冻结记录、预测和结果 CSV、测试，以及 README 中的再生成命令。此前在项目根目录执行的检查结果为：`50 passed`，且 `python -m pipeline.cli validate-artifacts` 能重建五行 `summary.csv` 的 F1 与 accuracy。

当前草稿的工程基础是可靠的：固定 split、train-only 拟合、dev 上选择、冻结 decision、held-out 评估、稳定 ID 和主要 schema 均已实现。尤其是题目澄清所要求的 error transition 口径已经正确实现：`summary.csv` 的 `fixed_fp`、`fixed_fn`、`new_fp`、`new_fn` 都相对于同一个 Ticket 1 冻结 baseline，而不是相对于上一个 ticket。

不过，当前版本不应直接作为高分最终提交。最重要的风险是缺少两个明确 deliverable，且 Ticket 1、Ticket 2/3 和 Ticket 4 的证据或表述仍不足以支撑其结论。

## 要求覆盖总表

| 原题要求 | 当前状态 | 审核结论 | 优先级 |
| --- | --- | --- | --- |
| 五个 ticket，各有 hypothesis、lever、controlled setup、dev/held-out evidence、例子与 limitation | 基本具备 | Ticket 5 正文例子不足；Ticket 4 的 lever 混合，见后文 | P1 |
| 固定 train/dev/held-out split，held-out 仅在决定冻结后使用 | 已实现 | `decisions.json` 和 CLI freeze 检查支持此流程 | 通过 |
| 机器可检查的 predictions、summary、threshold、audit 产物 | 已实现 | schema 与结果重建检查均通过 | 通过 |
| transition 指标固定参照同一 baseline | 已实现 | Ticket 2--5 均与 Ticket 1 baseline 比较 | 通过 |
| baseline discrepancy diagnosis | 部分完成 | 未复现 reference，也未确定差距原因 | P1 |
| 对表面信号/metadata shortcut 的稳健性分析 | 部分完成 | 当前 Ticket 3 metadata stress 实验无信息量 | P1 |
| data-quality audit，不修改 held-out 标签 | 已实现 | 保留原始标签且使用合法 disposition | 通过 |
| `report.pdf` | 缺失 | 明确的必交文件 | P0 |
| `logs/chat.md` | 缺失 | 明确的必交文件；必须是真实记录，不应编造 | P0 |
| README 中给出再生成命令 | 已实现 | 命令和流程说明清楚；最终提交前需用指定 conda 环境重测 | P1 |
| AI 使用声明、引用、至少三项 Difficulties and Solutions | 缺失 | 这些应写入正式报告 | P0 |

## 已验证正确的部分

### 冻结 baseline 的 transition 语义

`pipeline/experiments.py` 在 Ticket 1 后保存 `base_held_pred`。Ticket 2、3、4、5 计算 held-out transition 时都将该数组作为 `old_pred`，而不是复用上一个 ticket 的 prediction。因此：

- Ticket 2 的 `32/8/7/15` 是 URL replacement 相对 baseline 的变化；
- Ticket 3 保留 text-only，因而四项 transition 都是 0；
- Ticket 4 的 `17/37/25/9` 是新模型相对 baseline 的变化；
- Ticket 5 继承 Ticket 4 的模型，所以也应当得到 `17/37/25/9`，这不是重复计算错误。

`metrics.error_transitions()` 对 FP、FN、正确和错误四种转换做了互斥分类，输出与 `summary.csv` 当前数值一致。这一点符合课程 Q&A 的额外澄清。

### 可复现性与产物

当前草稿有下列正面证据：

- `data/split_indices.json` 固定 4,567/1,523/1,523 的 train/dev/held-out 划分；
- FeatureBuilder 在 train IDs 上 fit，测试覆盖 dev/held-out leakage；
- `experiments/decisions.json` 为五个 ticket 记录了 frozen 决定和 dev 选择结果；
- `predictions/heldout_predictions.csv` 有 7,615 行，即 5 个 ticket 各 1,523 条稳定 ID 预测；
- `results/summary.csv`、`threshold_sweep.csv`、`error_transitions.csv`、`data_quality_audit.csv` 和环境版本产物都存在；
- 测试包含 split 完整性、normalization、阈值、transition 覆盖、audit disposition、训练集拟合以及 artifact 重建。

这些证据应进入报告的方法和可复现性部分，而不应仅放在 README 或 walkthrough 中。

## P0：提交前必须完成

### 1. 生成真正的 `report.pdf`

原题明确要求一份自包含、连贯的项目报告；`PROJECT_WALKTHROUGH.md` 不能替代它。目前 README 还写着“`report.pdf` was not generated”，最终提交前必须删除或改写这类声明。

报告建议采用下列结构：

1. **Problem and protocol**：任务定义、数据来源、固定 split 数量、CPU-only 条件、目标指标和预注册式决策流程。
2. **Reference baseline diagnosis**：reference 数值、复现结果、差异、已测因素、不能确认的因素。不要把“最可能”写成“已证明”。
3. **Five ticket investigations**：每个 ticket 用相同模板说明 hypothesis、仅改变的变量、dev 选择、冻结后的 held-out 结果、precision/recall、具体 stable IDs、局限。
4. **Cross-ticket comparison**：五行 summary 表，明确所有 `fixed_*` / `new_*` 是相对 Ticket 1 baseline；可配混淆矩阵和 Ticket 4 threshold/PR 图。
5. **Data-quality findings**：展示少量具有代表性的 duplicate、冲突标签、near duplicate 和高置信错误，不要只报告 1,687 条的总数。
6. **Limitations and conclusion**：dev selection 不确定性、reference 配置不完整、公共 benchmark 的 shortcut 风险、缺乏独立人工标注等。
7. **Difficulties and Solutions**：至少三个真实且可验证的困难及解决办法。
8. **AI usage declaration and references**：说明 AI 用在何处、哪些结论由人工复核、如何用测试/CSV/具体 ID 验证；引用 Disaster Tweets 数据、镜像来源、scikit-learn、pandas、NumPy、SciPy 和使用的工具。

“Difficulties and Solutions” 可以基于实际工作写成：reference contract 未公开完整参数而无法精确归因；保证 vectorizer/模型只在 train 上 fit；强制 held-out 前冻结；让 artifact 能从 prediction 重建。不要写没有发生过的困难。

### 2. 提供真实的 `logs/chat.md`

该文件是原题 deliverable。应记录实际与 AI 工具的关键对话或整理后的真实摘要，包括：提出的问题、建议、采纳/拒绝原因、最终人工验证命令和结果。不得补造对话内容，也不要把 AI 输出直接当作实验结论。

### 3. 更正 README 的提交状态

在最终生成 `report.pdf` 和 `logs/chat.md` 后，修改 README 中“未生成”的说明。README 应保留可执行再生成命令、数据获取方式和输出位置，但不应公开声明必交文件缺失。

## P1：会影响结论可信度的修复

### 1. Ticket 1：baseline discrepancy 尚未被解释

现状：

- contract 指定 reference held-out F1 为 `0.757422`，容差为 `0.001`；
- 当前 baseline held-out F1 为 `0.731984`，差距为 `0.025438`；
- unigrams-only 达到 `0.751220`，但仍差 `0.006203`；
- 仅测试了 unigram 和 `lbfgs` solver 两个诊断 probe。

因此，“n-gram configuration is the most plausible tested source”只能写成有限证据结论，不能说它解释了 reference gap。题目要求是：若不匹配 reference，应说明哪种 split、seed、版本或 preprocessing 差异导致差距。当前实验只证明 n-gram 是已测试因素中影响最大的一个，尚未完成归因。

建议：

- 固定数据和 split，不使用 Ticket 1 held-out 结果反复挑选模型；
- 在 dev 上预定义诊断矩阵，至少隔离 `ngram_range`、`sublinear_tf`、`max_features`、`min_df`、token pattern、lowercasing、normalization、`C`、solver 和 scikit-learn 版本；
- 对每个 probe 写明“单因素变化”和默认参数；
- 若完整配置仍不可知，报告应明确说 reference 配置未公开、无法作因果归因，而不是暗示已经复现；
- 若能找到独立、可引用的参考实现，使用其公开参数重跑并把来源写入报告。

### 2. Ticket 2：robustness probe 没有验证选择后的模型

Ticket 2 选择的是 `url_replace`，但 `perturbation_stress.csv` 中 URL、mention、hashtag 等 stress 是对原始 baseline `bld, mdl` 评分，而非对选中的 URL-replacement 模型评分。它证明原始模型对输入表面格式变化敏感，却不能证明选定的 normalization 更稳健。

建议将每种 stress 设计为成对比较：

| 训练模型 | 测试输入 | 要回答的问题 |
| --- | --- | --- |
| raw baseline | raw / 扰动文本 | 原始模型对该表面信号有多敏感？ |
| URL-replace model | 规范化后的等价扰动输入 | URL replacement 是否降低了该依赖？ |
| 其他候选 normalization 模型 | 与其训练规范一致的输入 | 结论是否仅由 train/test 格式失配造成？ |

报告中应明确区分“测试时单独改变输入造成 distribution shift”和“训练、推理都使用同一 normalization”的效果。当前 URL stress 大幅下降并不能直接证明 URL replacement 成功，只能说明 raw baseline 依赖 URL 表面片段。

### 3. Ticket 3：metadata stress 当前是零信息实验

`keyword_blank`、`keyword_shuffle`、`location_blank`、`location_shuffle` 都是 0 flips。原因是代码把 perturbation 应用于 dev-selected 的 text-only 模型；text-only FeatureBuilder 不读取 `keyword` 或 `location`，因此任何 metadata 改动都不可能改变预测。

这不是 shortcut robustness 的证据。应至少对下列实际使用 metadata 的模型做 perturbation：

- `keyword`；
- `location`；
- `keyword_plus_location`；
- `text_plus_keyword`；
- 可选的 `text_plus_numeric` 对应 numeric feature 扰动。

每个模型报告原始 F1、blank F1、shuffle F1、prediction flips 以及 FP/FN 转换。这样才可判断 keyword 是合理任务信息、benchmark artifact，还是混合证据。最后仍可选择 text-only，但理由应是“它在 dev 上最佳且避免 metadata 依赖”，而不是把 zero-flip 当成 metadata 稳健性。

### 4. Ticket 4：混合多个 lever，且正则化措辞错误

最终模型同时改变了 `$C$`、`class_weight` 和 threshold；LinearSVC 也在候选集中。这种网格搜索可以选择 operating point，却不足以将 F1 改善因果归于其中某一项。题目强调 intended lever 与 controlled experiment，因此建议补充消融结果：

| 对照 | 仅变化 | 建议报告 |
| --- | --- | --- |
| Ticket 1 baseline | 无 | 起点 |
| baseline + threshold | threshold | PR/F1 tradeoff |
| baseline + class weight | class weight | 正例召回与 FP 变化 |
| baseline + `$C$` | `$C$` | 容量/正则化影响 |
| 组合模型 | 明确列出组合 | 最终选择的增益 |
| LinearSVC | classifier family | 模型族比较 |

此外，ticket 中“stronger regularization settings”是不正确表述。在 sklearn 的 Logistic Regression 中，$C$ 越大表示 L2 正则化越弱；从 `$C=1` 到 `$C=10` 是减弱正则化，而不是增强。应将描述改为“weaker regularization together with class balancing and a higher threshold”，且不要声称仅由其中某个因素导致正例分数变化。

### 5. Ticket 5：正文应给出真正的案例分析

CSV 有丰富证据，但 ticket 正文目前只说证据“recorded per row”，没有将具体 case 讲清楚。高分报告和 ticket 至少应各选一个以下类别，并包含 stable ID、原文简述、标签、预测/相似度、disposition 及理由：

- 跨 split exact duplicate；
- normalized duplicate；
- conflicting-label duplicate；
- 高相似 near duplicate；
- 高置信 FP 或 FN。

尤其要解释：为什么冲突 duplicate 记为 `ambiguous` 而不是 `fix`；为什么模型的高置信分数不能证明 label 错；哪些 audit finding 被 `reject_false_positive`。如果没有可证实应修改的标签，`fix=0` 是可接受的，但报告必须说明采用这一保守决定的证据门槛。

## P2：可验证性和呈现质量增强

### 1. 让 artifact validator 验证 transition 计数

当前 `validate_artifacts()` 仅从 prediction 重建每个 ticket 的 held-out F1 与 accuracy；它不检查 `summary.csv` 的 `fixed_fp`、`fixed_fn`、`new_fp`、`new_fn` 是否来自 Ticket 1 baseline。实现当前是正确的，但 validator 没有保护这条课程 Q&A 的关键契约。

建议在 validator 或新增测试中：

1. 从 held-out predictions 找到 Ticket 1 的 `y_pred`；
2. 对 Ticket 2--5 逐行重建 transition category；
3. 将四项 count 与 summary 对应列比较；
4. 断言 Ticket 1 的四项均为 0；
5. 断言每个 ticket 的 transition 行数为 1,523，且分类互斥且覆盖全部行。

### 2. 为报告补充图表和不确定性

建议至少增加：

- Ticket 4 的 precision/recall/F1 对 threshold 曲线，并标出 `$0.56$`；
- 五个 ticket 的 F1、precision、recall 对比表；
- baseline 到 Ticket 4 的 FP/FN transition 图或表；
- Ticket 3 实际 metadata 模型的 blank/shuffle stress 对比；
- Ticket 5 的小型案例表。

不要把微小的 dev 提升写成确定改善。Ticket 2 dev F1 的提升很小，Ticket 4 从 dev 到 held-out 的差异也只能提供有限的稳健性证据；报告需要明确单一 dev split 和多重选择带来的不确定性。

### 3. 清晰记录环境与命令

README 现有命令基本充分。最终提交前应在 `aiaa3102` conda 环境中重跑全部产物并记录环境版本。不要只引用旧 CSV；应确保报告数字、`environment.json`、summary 和预测文件来自同一轮运行。

## 推荐修订顺序

1. 补真实 `logs/chat.md`，建立 report 的来源和 AI 使用声明。
2. 修复 Ticket 3 metadata stress，补 Ticket 2 的 selected-model robustness 比较。
3. 补 Ticket 4 消融表并改正 `$C$` 的正则化文字。
4. 扩展或收紧 Ticket 1 discrepancy 结论；不要在证据不足时声称找到原因。
5. 给 Ticket 5 加入可读的 stable-ID 案例。
6. 加强 transition validator/test，重新生成所有 artifacts。
7. 依据最终 artifacts 撰写正式报告，生成 `report.pdf`，再修改 README 的提交状态。

## 最终提交前验证清单

在本目录运行下列命令；所有 Python 操作均使用指定 conda 环境：

```bash
conda run -n aiaa3102 python -m pytest -q
conda run -n aiaa3102 python -m pipeline.cli validate-data
conda run -n aiaa3102 python -m pipeline.cli run-all
conda run -n aiaa3102 python -m pipeline.cli validate-artifacts
```

提交前人工检查：

- `report.pdf` 存在、可打开，并包含 Difficulties and Solutions、AI declaration 和 references；
- `logs/chat.md` 存在，且内容为真实对话或真实整理摘要；
- README 不再声称必交文件未生成；
- report 中每一个数字可追溯到最终 CSV；
- Ticket 2--5 的 transition 描述都明确写为“相对 Ticket 1 frozen baseline”；
- 所有 correction/audit 结论均保留原始标签和证据，不编辑 held-out 标签；
- `run-all` 后再运行 validator，避免报告引用过期产物。
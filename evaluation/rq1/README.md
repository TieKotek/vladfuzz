# RQ1：自然语言驾驶指令质量人工评测

## 1. 实验目的

RQ1 评估 VLAD-Fuzz 的测试用例生成模块能否生成正确且高质量的自然语言驾驶指令。对相同驾驶种子比较两种受控条件：

- 场景图像与结构化路线信息共同生成；
- 仅根据场景图像生成。

人工评测只判断指令本身是否正确、清楚并贴合测试用例，不评价被测驾驶模型能否执行成功。

## 2. 盲评设计

正式评测采用两名评测者独立盲评、第三人裁决的流程。系统只生成一份匿名 case 包。实验负责人将该包的独立副本分别发给两名评测者；两人评价完全相同且顺序一致的 case，并分别返回填写结果。评测包不得包含方法名称、实验假设、私有映射、生成参数或源文件路径。

每个 case 包含：

```text
case_xxxxxx/
  image.png
  instruction.txt
  reference.txt
  annotation.txt
```

`reference.txt` 只包含模拟器根据测试用例规划路线导出的结构化路线描述，用于确定路线语义 ground truth。它不包含基础自然语言指令，也不是措辞模板；评测者应判断待评指令的动作、方向和顺序是否与规划路线一致，而不是进行字面相似度比较。

## 3. 评分与错误分类

评分采用 ordinal 三级标签：

- `0`：错误、不可执行或与路线/场景冲突；
- `1`：语义正确且可执行，但自然性、清晰度或场景贴合质量一般；
- `2`：语义正确，并且自然、清楚、简洁且贴合场景。

评分为 `0` 时必须选择至少一个错误类型；多个适用标签以英文逗号分隔，并将主要原因置于首位：

- `wrong_maneuver`
- `missing_maneuver`
- `hallucinated_maneuver`
- `scene_mismatch`
- `target_marker_leakage`
- `ambiguous_or_non_navigation`
- `format_or_language_issue`
- `other`

任一标签为 `other` 时必须填写备注。评分为 `1` 或 `2` 时错误类型必须留空。对外分发的完整评分规则由 `evaluator_readme.md` 维护，并在生成 round 时复制为该评测包的 `README.md`。

## 4. 正式标注包生成

当前正式配置使用每个静态场景最多 5 个种子，并从每个种子的每种生成条件抽取 1 条指令。当前 14 个静态场景因此形成 70 个配对、140 个匿名 case：

```bash
python3 -m evaluation.rq1 prepare \
  --seeds-root test_cases \
  --output-root results/rq1/formal_round \
  --max-seeds-per-static-scenario 5 \
  --instructions-per-seed 1 \
  --random-seed 0 \
  --overwrite
```

输出结构：

```text
results/rq1/formal_round/
  private/
    mapping.csv
    sampling_config.json
    manifest.json
  package/
    README.md
    cases/
  submissions/
  adjudication/
  reports/
```

只分发 `package/`，不要分发 round 根目录。两名评测者应各自收到该目录的独立副本，并分别返回填写后的目录；`private/` 必须由实验负责人保留。

## 5. 回收与校验

建议将返回目录原样保存到：

```text
results/rq1/formal_round/submissions/annotator_a/
results/rq1/formal_round/submissions/annotator_b/
```

分别校验：

```bash
python3 -m evaluation.rq1 validate \
  --round-root results/rq1/formal_round \
  --submission results/rq1/formal_round/submissions/annotator_a
```

校验会拒绝缺失或额外 case、空评分、非法分数、不合法错误类型，以及评分与错误类型之间不一致的记录。

### 5.1 优先检查正确性分歧

查找一名评测者评分为 `0`、另一名评分为 `1` 或 `2` 的 case：

```bash
python3 -m evaluation.rq1 check-correctness \
  --round-root results/rq1/formal_round \
  --annotator annotator_a=results/rq1/formal_round/submissions/annotator_a \
  --annotator annotator_b=results/rq1/formal_round/submissions/annotator_b
```

先校验两份返回包，再生成 `reports/correctness_conflicts.csv`，记录匿名 case ID、双方分数、错误标签、备注及标注文件路径。终端输出正确性一致率、双方零分数量和全部冲突 ID。`1` 与 `2` 的分歧以及双方均为 `0` 但错误标签不同，不属于本检查的筛选范围，仍由后续 merge 处理。

此命令不修改原始评分、不创建裁决包；重复执行只更新该检查报告。原始独立评分可以存在分歧，应由第三人基于证据裁决，而不是为提高一致率回改 A/B 数据。原始一致性统计仍使用未裁决的两份评分。

## 6. 合并与第三人裁决

```bash
python3 -m evaluation.rq1 merge \
  --round-root results/rq1/formal_round \
  --annotator annotator_a=results/rq1/formal_round/submissions/annotator_a \
  --annotator annotator_b=results/rq1/formal_round/submissions/annotator_b
```

合并后会生成：

- `private/merged_annotations.csv`：两名评测者的原始评分；
- `private/merge_summary.json`：一致和分歧数量；
- `adjudication/private_disagreements.csv`：仅供负责人查看的分歧明细；
- `adjudication/package/`：不含方法身份和 A/B 评分的第三人裁决包。

第三名评测者填写 `adjudication/package/cases/*/annotation.txt` 后运行：

```bash
python3 -m evaluation.rq1 adjudicate \
  --round-root results/rq1/formal_round
```

## 7. 统计分析

### 7.1 保留主观评分差异的描述性统计

若正确性和错误标签已经裁决，但保留 `1` 与 `2` 的主观差异，使用：

```bash
python3 -m evaluation.rq1 analyze-descriptive \
  --round-root results/rq1/formal_round \
  --annotator annotator_a=results/rq1/formal_round/submissions/annotator_a \
  --annotator annotator_b=results/rq1/formal_round/submissions/annotator_b
```

此入口直接读取已处理的两份返回包，不要求生成合并或裁决目录，也不修改评分。
存在 `0` 与 `1/2` 的分歧或错误标签分歧时拒绝统计。输出位于 `reports/descriptive/`：

- `summary.md`、`summary.json`、`summary.tex`：描述性报告与可用于论文的表格片段；
- `method_summary.csv`：分别展示两位评测者的评分分布、正确率和高质量率；
- `scenario_summary.csv`：逐静态场景、方法和评测者的统计；
- `ratings.csv`：结合私有映射还原的逐用例评分明细；
- `error_types.csv`：裁决后的错误类型计数，每个用例在每个类型中只计一次。

正确率为 `(score_1 + score_2) / N`；高质量率为 `score_2 / N`，分别保留两位评测者的结果。
错误类型允许多标签，因此各类型数量之和可能大于错误用例数量。
不计算原始评分一致性、评分均值、配对检验或置信区间；已处理评分不能作为原始独立评分的一致性证据。

### 7.2 完整裁决后的分析入口

以下为保留的完整裁决分析流程，会计算一致性及推断统计，不用于本轮描述性分析。

```bash
python3 -m evaluation.rq1 analyze \
  --round-root results/rq1/formal_round \
  --bootstrap-iterations 10000 \
  --random-seed 0
```

分析只使用一致标签或第三人裁决后的最终标签，输出：

- 原始评分一致率；
- 线性加权 Cohen's kappa；
- 两种条件的三级评分分布；
- 正确率与高质量率；
- 配对 exact McNemar 检验；
- 按静态场景聚类 bootstrap 的配对风险差及 95% 置信区间；
- 错误类型分布；
- JSON、CSV、Markdown 和 LaTeX 报告。

Ordinal score 不作为连续变量报告平均值。主要结果为正确率和高质量率，完整三级分布用于解释质量变化。

## 8. 数据管理要求

- `evaluation/rq1/` 中的代码、协议和模板属于框架，应纳入版本控制；
- `results/rq1/` 包含图像、标注和私有映射，已通过 `.gitignore` 排除；
- 两名评测者可以使用相同的原始压缩包，但回收文件必须分别保存，不能相互覆盖；
- 原始 A/B 标注和第三人裁决结果必须全部保留；
- 正式写入论文前，应冻结代码 commit、round manifest 和报告；
- 不得手工修改聚合数字，论文表格应由冻结 round 自动生成。

重复执行 merge 会拒绝覆盖已有 `adjudication/`。如确需重新合并，应先人工归档原裁决目录，再显式移除该目录。

# PaperGraph-Bench 测评方案：指标 1.3 多维度幻觉评估指标体系

> **已有结果离线统计：`uv run --locked python scripts/summarize_results.py --result-root eval_result`，只读取完成报告与对话轨迹，不调用模型、不需要密钥或最终状态缓存。测试新模型：配置 Judge 和被测模型两组 API 后执行 `.\start.ps1`（macOS/Linux：`bash ./start.sh`），使用已有题包评测并自动生成同一套论文口径统计。**

## 一 测评目标与四个核心维度

本方案仅对应**指标 1.3「多维度幻觉评估指标体系」**，按 2026 年 11 月里程碑要求组织测试：支持核心评估维度不少于 4 个，细粒度幻觉类型标注不少于 3 个层级。

测评对象为 **PaperGraph-Bench 面向科学论文理解的图引导多维度幻觉评测工具**，公开仓库为 [PaperGraphBench](https://github.com/furao0011/PaperGraphBench)。统计与论文分析口径一致，数据源为 `evaluation_report.json` 和 `dialogue_trajectory.json`；不以运行时最终状态、全局 KC 点亮率或事件闭环率替代论文指标。新增论文、重建图题包不属于本次测试步骤。

本次验收的是工具能否提供可计算、可定位、可追溯的四维结果及分层分类，**不是要求被测模型达到某个覆盖率、幻觉率或修复率阈值**。

| 核心维度 | 测试方法 | 主要指标与计算口径 | 核验重点 |
| --- | --- | --- | --- |
| 1. 知识覆盖 | 读取宏阶段主问题的首次回答及 KC 覆盖判决，不并入后续修复回答 | 首答 KC 覆盖率＝首答中覆盖的目标 KC 数 / 首答目标 KC 数；首答宏完成率＝首答覆盖完成的宏问题数 / 已执行宏问题数 | 以 `main / macro_main_question` 首答为准；不使用最终全局覆盖率 |
| 2. 挑战诱发幻觉 | 读取普通 `challenge_question` 的错误前提、过度推断、错误关系等判决，并分文本/图像统计 | 总体、Text、MM 挑战失败率＝相应失败挑战数 / 相应实际执行挑战数 | **不含 `thread_challenge_question`**；`INCOMPLETE` 不算幻觉，未执行题目不计入分母 |
| 3. 幻觉行为 | 汇总各轮 `judge_result.hallucination_events` 的新事件及四类分布 | 主指标为每篇平均幻觉事件数；补充含事件轮次比例、事件密度 | 有事件的轮次才计入幻觉轮次分子，不按 Judge 状态名代替；不读取最终事件字典 |
| 4. 错误修复 | 从真实跟进轮的 `repair_context` 识别 detail 与 hallucination 修复任务，按根任务分组 | 两类任务各报告成功率及平均尝试轮数；尝试计至首次成功，未成功任务计其实际尝试数 | 不用事件 resolved 比例或缺口闭环率代替任务成功率；输出 `repair_tasks.csv` 供复算 |

各维度至少保留一个实际执行实例：问题、模型回答、Judge 判决、关联 KC 或事件；修复维度分别提供幻觉修复和遗漏补全实例。未成功任务可验证过程记录功能，但不计入修复成功数。同一维度下的多个指标不重复计作新维度。

挑战题的“失败”指模型接受误导、作出与论文证据不符的回答，属于挑战诱发幻觉，不是接口失败。`INCOMPLETE` 计入已执行普通挑战数，但不计入失败数；接口错误另列执行异常。线程挑战可产生幻觉事件并进入四类归并，但不进入论文普通挑战失败率的分子、分母。

## 二 数据、环境与运行前配置

### 2.1 测试数据与实验设置

已有结果可直接对 `eval_result/` 或任意测试批次离线统计，不读取题包或运行时缓存。新模型使用完整公开仓库 `data/` 中的 12 篇论文题包及其引用的 `rawPaper/` 图片；不同模型保持相同数据、Judge、提示词与修复预算。默认每模型一轮，可用 `--repeats 3` 独立重复三轮。

每批记录论文与模型清单、Git 版本、输入签名、轮数和参数。动态调度、幻觉数量及修复分支是否触发以实际轨迹为准，不假定各模型轨迹完全相同。

### 2.2 环境准备

安装 [uv](https://docs.astral.sh/uv/)，使用 Python 3.11+，按 `uv.lock` 准备依赖。已有结果统计只需本地文件，无模型服务或 GPU 要求；只有新模型实测需要可用模型 API 和调用额度。

```powershell
git clone https://github.com/furao0011/PaperGraphBench.git PaperGraphBench-test
Set-Location PaperGraphBench-test
git rev-parse HEAD
```

使用完整克隆，不使用仅检出部分论文的 sparse checkout。记录操作系统、CPU、内存、磁盘、Python 与 uv 版本。

### 2.3 新模型实测的必填 API 配置

> **仅测试新模型时需要下表配置；已有结果统计不加载私有配置、不需要 API。** 仓库不提供可用密钥、模型服务或免费额度。首次启动会创建私有 `config.env` 并提示待填项；填写两组配置后再执行启动命令，已有配置不会被覆盖。

| 服务 | 必填 Key | 必填 Base URL | 必填模型 ID | 能力要求 |
| --- | --- | --- | --- | --- |
| 评测 Judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` | OpenAI-compatible chat，支持结构化 JSON 输出 |
| 被测模型 | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` | OpenAI-compatible chat，支持文本与图片输入 |

Base URL 填服务商 API 根地址，不填网页或 `/chat/completions` 完整端点；模型 ID 必须在账号下可调用。同一服务可兼任两种角色，但两组字段仍须分别填写。环境变量优先于配置文件，模型比较期间固定 Judge。

其余项保持公开默认值即可。正式测试保持 `USE_ONLINE_EVAL=true`、`ALLOW_MOCK_EVAL=false`、`ALLOW_OFFLINE_FALLBACK=false`，不得用模拟或离线输出替代实测。默认 `CLAIM_VERIFY_ENABLED=false`，无需配置视觉生成、Embedding 或 OCR；开启额外 claim verification 才另需 Embedding。

`EVAL_MAX_TURNS=0` 不设置全局提前截断，但保留框架自身的有界跟进规则。不要为缩短测试而随意更改修复预算。配置、密钥和 Authorization 头不进入交付包。

## 三 测试步骤与运行命令

### 3.1 已有结果离线统计

在仓库根目录执行，路径可换为已有批次或某轮结果目录：

```powershell
uv run --locked python scripts/summarize_results.py --result-root eval_result
```

该命令读取逐任务 `evaluation_report.json` 与 `dialogue_trajectory.json`，自动生成同一套论文口径四维表、四类幻觉表、修复任务和分类追溯文件。**不调用模型，不需要 API 配置、原论文、图题包或 `cache/eval_state_graph.json`，也不重建最终状态。** 默认输出到所选结果根的 `statistics/`。

统计退出码 0 表示发现的任务均完整，2 表示批次不完整，1 表示数据或执行错误。有批次任务清单时核对缺失任务；没有清单的历史目录只描述实际找到的报告，计划总数未知，不宣称整套测试已完成。

如需核对仓库发布的论文固定实验，可在离线统计后执行参考值核验：

```powershell
uv run --locked python scripts/summarize_results.py --result-root eval_result --output-dir outputs/paper_verification/statistics
uv run --locked python scripts/verify_paper_results.py --statistics-dir outputs/paper_verification/statistics
```

核验使用仓库内冻结参考值，分别列出主指标、修复、四类数量及回答长度的差异，不调用模型，也不修改实测结果。退出码 2 表示核心结果不符或出现新的成本偏差，1 表示输入错误。原始结果中已确认的 GLM 回答长度差异（轨迹实算 1737.1，论文 1736.1）单列为 `known_cost_discrepancies` 警告，不修改真实值；任何其他偏差均不放行。该检查只针对论文原始固定实验，不要求新模型或重新采样结果与参考值相同。

### 3.2 新模型默认评测与自动统计

填好第 2.3 节的两组 API 后运行：

```powershell
# 默认：全部已有题包，每模型一轮，结束后按论文口径自动统计
.\start.ps1

# 可选：两篇论文并发
.\start.ps1 --workers 2

# 可选：同一题包独立评测三轮
.\start.ps1 --workers 2 --repeats 3
```

macOS/Linux 使用：

```bash
bash ./start.sh
# 并发或独立重复
bash ./start.sh --workers 2 --repeats 3
```

| 其他测试目的 | Windows 命令 |
| --- | --- |
| 显式指定测试模式，与默认相同 | `.\start.ps1 test --workers 2` |
| 使用其他私有配置 | `.\start.ps1 --config path/to/config.env` |
| 比较多个实际可调用的模型 | `.\start.ps1 --models model-a model-b --workers 2` |
| 测试另一套已有题包 | `.\start.ps1 --data-root path/to/dataset` |

`--workers` 为论文级并发。多模型 ID 须替换为实际可调用的 ID，并配置相应服务。Windows 若拦截脚本，可用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`。不要使用 `run --force` 做独立重复，以免重新建图出题。

可先用 `.\start.ps1 --dry-run` 预览，或 `.\start.ps1 --paper-id ConVis` 做独立单篇检查，再新开全量批次。运行中查看终端、批次 `main.log` 和 `logs/`；认证、限流、超时、格式及附件错误须保留。预览成功不代表 API 已连通，接口调用成功也不代表回答无幻觉或分类层级有效。

### 3.3 查看自动输出

新模型启动会打印 `outputs/testing/<session>/`；离线统计打印所选结果根下的 `statistics/`。**先查看 `four_dimensions.csv` 和 `failure_category_by_model.csv`**，需要核对修复时查看 `repair_tasks.csv`，再检查原始事件和轨迹。分类文件自动导出，无需额外 PowerShell 提取步骤。

`four_dimensions.csv` 每模型十一行：首答覆盖与宏完成 2 行；总体、Text、MM 挑战失败 3 行；平均事件数与含事件轮次比例 2 行；detail/hallucination 两类修复的成功率与平均轮数 4 行。列为 `model`、`status`、`dimension_id`、`dimension`、`metric`、`source_metric`、`value`、`aggregation`、`paper_std`、`valid_papers`、`valid_runs`、`micro`、`numerator_sum`、`denominator_sum`；以 `value` 和 `aggregation` 确认论文主值，不能将所有指标一律取论文均值。

失败、缺失或中断任务保留为不完整记录，不填零混入能力指标；完成报告缺少对应轨迹也记为不完整。身份不一致、轮次或事件 ID 重复等数据错误明确报错；缺少可选运行时缓存不构成统计错误。

## 四 细粒度幻觉分类专项测试

### 4.1 论文中的四类幻觉与归并规则

本方案采用论文实验中的**四类幻觉**：KC 证据冲突、错误前提接受、过度推断接受、错误关系接受。线程错误按相同错误性质归并，不作为额外主类别；底层事件族和原始标签也不另计为论文主类别。

| 论文类别 | 表中标签 | 规范统计列 | 定义与判定依据 |
| --- | --- | --- | --- |
| KC 证据冲突 | `Contr.KC` | `contradicted_kc_claim` | 回答中的论文事实或主张与关联 KC 的证据冲突；按论文统计口径合并回顾阶段的退化错误 |
| 错误前提接受 | `False prem.` | `challenge_false_premise_acceptance` | 把与论文证据不符的前提或变异后的线程前提当真，并据此作答，而非纠正前提 |
| 过度推断接受 | `Overclaim` | `challenge_overclaim_acceptance` | 接受超出实验范围、适用条件或结论强度的说法，包括对已建立推理桥的过度推广 |
| 错误关系接受 | `Wrong rel.` | `challenge_wrong_relation_acceptance` | 接受不受支持、错置或颠倒的 KC 关系，或采用错误的线程推理桥 |

程序根据轨迹中新事件的 `hallucination_type` 精确归并，规则与论文结果分析一致：

| 原始 `hallucination_type` | 归入论文类别 |
| --- | --- |
| `contradicted_kc_claim`、`review_regression` | `Contr.KC` |
| `challenge_false_premise_acceptance`、`thread_premise_mutation_acceptance` | `False prem.` |
| `challenge_overclaim_acceptance`、`thread_overclaim_acceptance` | `Overclaim` |
| `challenge_wrong_relation_acceptance`、`thread_wrong_bridge_acceptance`、`thread_reasoning_hallucination` | `Wrong rel.` |

仅按上表映射，不根据事件族、细分标签或文字印象猜测类别。`fabricated_claim`、未细分的 `challenge_failure` 及其他未映射标签保留原值，计入 `unmapped_event_count`，不强行归入四类。四类计数之和为 `categorized_event_count`，再加 `unmapped_event_count` 应等于全部 `event_count`。只有实际事件计入类别，挑战题数量不等于模型发生的幻觉数量。

`hallucination_family / hallucination_type / subtype` 是原始记录字段，用于技术溯源和分类质量检查，不代表另一套论文主类别。逐事件导出额外提供 `paper_category` 和 `paper_category_label`，记录上述规范类别及展示标签；原始字段保持不变。来源为 `turns[].judge_result.hallucination_events`，可沿 `trajectory_path` 和来源轮次复核。

**四类是分类数量，不等于四个层级，也不自动满足三级标注要求。** 层级核验使用实际原始路径及其语义：标签缺失、重复、仅含状态或同义重述均单列，不能补写标签凑足三级。文本与图像是交叉证据模态，不另算新的幻觉类别；通过源轮次的 `question_metadata.modality_pool`、资产引用及多模态输入记录定位证据。

统计源为每轮 `judge_result.hallucination_events` 中的**新事件数组**，不是最终状态字典。同一任务内 `event_id` 必须唯一，重复时报数据不一致，不做首出现去重或静默删除；不同论文、模型或独立轮次可使用相同局部 ID，按任务分别计数。已被后续修复的事件仍保留在发生数中。

### 4.2 默认自动分类统计

默认启动与独立离线统计采用同一统计入口：只读取已完成任务的报告和轨迹，核对身份、轮次与事件 ID，再从轨迹计算分类、修复任务和论文指标。`statistics/` 中自动生成以下分类文件，不需要额外 API 或提取步骤：

- `failure_category_by_model.csv`：按模型列出论文四类计数、`categorized_event_count`、`unmapped_event_count`、全部 `event_count` 和任务 `status`，优先查看。
- `classification_events.csv`、`classification_events.json`：逐事件论文类别、原始标签、来源轮次、关联 KC 和检测时快照。
- `classification_paths.csv`：按模型及原始 `family/type/subtype` 组合列出 `event_count`，用于排查底层标签，不代替论文四类主表。
- `classification_summary.csv`：按模型汇总事件数、分类字段完整性和缺失、重复等问题，便于直接查看。
- `classification_summary.json`：结构化分类汇总及实际分类路径分布。

逐事件明细保留 `round`、`model`、`paper_id`、`event_id`、`paper_category`、`paper_category_label`、`trajectory_path` 及 `issues`。`classification_summary.json` 的 `paper_categories` 提供按模型的论文四类统计，`classification_counts` 保留原始标签组合及数量；字段完整性和标签问题不能作为额外论文类别。

`status_at_detection`、`followup_turns_at_detection`、`resolved_at_turn_at_detection` 仅是事件出现时的原始快照，**不代表最终状态，不据此计算修复成功率**。后续修复由轨迹中的实际跟进轮和 `repair_context` 单独统计到 `repair_tasks.csv`。

分类数量为各模型已完成任务跨论文、跨独立轮次的原始事件数累加，不是论文等权均值或幻觉率；轮数不同的批次不能只比较事件总数。四维表的 `value` 按 `aggregation` 分别采用论文等权或合并分子分母，不能统一按论文求均值。

统计保留原始标签，自动标记缺失字段、重复或未细分的候选路径，不填造细分标签。`three_fields_present` 和 `non_repeating_path` 仅表示字段完整或不重复，不自动证明其语义上满足三级细分；文件同时保留实际标签和原始事件，供验收复核。

未完成或缺失报告/轨迹的任务单列为不完整，不将其当作“零事件”纳入分类结果。事件结构损坏或 ID 重复时明确报错。合法空新事件数组正常导出，分类完整率为 N/A；运行时最终状态缓存既不读取，也不重建。

### 4.3 分类、证据与修复核验

1. **计数一致：**按轨迹新事件数组复算逐任务事件数与分类分布，并与导出明细核对。跨任务以“轮次、模型、论文、事件 ID”区分，不能仅按局部 ID 全批去重；不强行用报告的最终事件库统计覆盖轨迹口径。
2. **类别与层级有效：**按第 4.1 节的四类及精确归并规则核对主表，并单列未映射事件；四类计数总和加未映射数须等于全部事件数。结合 `target_failure_mode`、KC 证据及 Judge 原始判决说明实例依据。未出现的类别只记为本批未观测到，不以题目数量代替幻觉数量。原始路径中的缺失、重复、同义重述或仅含状态问题另列，独立核验实际层级。
3. **错误可定位：**由 `created_at_turn`、`source_question_id` 找到轨迹中的 `question_text`、`model_answer`、`judge_result`；由 `related_kc_ids` 找到主图 KC 的 `evidence` 及相关正文。事件 `source` 是记录来源，不是原文证据。图像事件还需核对资产引用与实际图片。
4. **修复可追踪：**按 `(repair_type, root_turn_id, root_question_id)` 将真实跟进轮归为修复任务。`repair_tasks.csv` 保留任务来源、尝试数、成功与首次成功轮；不得用事件检测快照中的状态、resolved 时间或最终状态缓存代替任务判定。

无事件时分类完整率记 N/A，该分支为待补测，不记为 100%。若缺少分类或修复实例，可增加独立重复轮次或补充被测模型，保留独立批次和模型信息；不得编造错误、补写标签或混入原模型正式均值。

## 五 结果文件与指标口径

```text
outputs/testing/<session>/
  testing.json                       计划任务、参数、版本、输入签名和状态
  main.log、logs/                     主进程与逐论文日志
  round-01/evaluation/<model>/<paper>/
    evaluation_report.json           运行时汇总、完成状态及事件计数
    dialogue_trajectory.json         问题、回答、Judge 判决及轮次
    cache/                          可选运行时缓存；离线统计不读取、不重建
  round-02/ ...                      仅在独立重复时生成
  statistics/
    four_dimensions.csv              自动：论文四维专用表，每模型十一行，优先查看
    failure_category_by_model.csv    自动：论文四类幻觉及未映射计数，优先查看
    summary.csv                      自动：完整模型指标与任务状态
    per_paper.csv                    自动：每轮每篇状态、指标和分子分母
    summary.json                     自动：结构化汇总
    repair_tasks.csv                 自动：实际修复任务、尝试轮数与成功判定
    classification_events.csv        自动：新事件类别、原始标签及检测快照
    classification_events.json       自动：相同明细的结构化版本
    classification_paths.csv         自动：原始标签路径数量，用于排查
    classification_summary.csv       自动：按模型的分类计数、完整性与问题统计
    classification_summary.json      自动：分类汇总及路径分布
```

`<model>` 为安全化后的目录名，完整模型 ID 以清单和报告为准。离线统计只需逐任务报告与轨迹；需要查看论文原证据时，可由 `related_kc_ids` 查对应主图、正文与引用图片，但这些并非统计输入。上述十一个统计文件默认自动生成。

`summary.csv` 的论文主值直接使用下列列名，**没有 `_mean` 后缀**；`four_dimensions.csv` 用 `source_metric` 对应这些列，以 `value` 展示主值。挑战维度的 `metric` 使用“挑战诱发幻觉”展示名称，`source_metric` 保留论文 `challenge_failure_rate` 等统计名：

| 维度 | 论文指标列 | 聚合口径 |
| --- | --- | --- |
| 知识覆盖 | `one_turn_kc_coverage`、`one_turn_macro_complete_rate` | 每篇首答覆盖比例、首答宏完成比例；同篇先跨独立轮次平均，再论文等权 |
| 挑战诱发幻觉 | `challenge_failure_rate`、`text_challenge_failure_rate`、`multimodal_challenge_failure_rate` | 汇总全部有效普通挑战的失败数 / 执行数；Text/MM 以该组挑战作分母，不含线程挑战 |
| 幻觉行为 | `mean_hallucination_events`、`hallucination_turn_rate` | 主事件数按论文等权；含新事件轮数 / 全部轮数按模型合并计数 |
| 错误修复 | `detail_repair_success_rate`、`hallucination_repair_success_rate` | 各类成功实际修复任务数 / 各类实际修复任务总数，按任务合并 |
| 错误修复 | `detail_repair_avg_rounds`、`hallucination_repair_avg_rounds` | 各类任务尝试数之和 / 各类任务数；成功任务计至首次成功，未成功任务计所有实际尝试 |

首答 KC 按每个宏主问题的目标集合与已覆盖集合交集计数，再累计该篇的分子分母，不用全篇最终点亮的全局集合替代。普通挑战以 `requires_multimodal_input=true` 或非空 `asset_references` 判为 MM，其余为 Text；不把线程挑战混入任一组。

补充列包括 `hallucination_event_density`、`mean_total_turns`、`mean_answer_chars_per_turn`。平均对话轮数按论文汇总，回答字符数按全部有效回答字符数 / 有效轮数计算。事件密度按模型合并事件数 / 总轮数，可大于 1，不是幻觉轮次比例。

修复任务从真实跟进轮按 `(repair_type, root_turn_id, root_question_id)` 分组，不把每次跟进当成一个独立任务。幻觉修复成功为 `SELF_CORRECTED`，或 `MAIN_PROGRESS / CHALLENGE_RESISTED / THREAD_PROGRESS` 且该轮无新事件；遗漏修复以覆盖完成，或非空 `remaining_kc_ids` 与该轮遗漏集合不相交判定。平均尝试轮数包含未成功任务，并非只对成功任务求均值，也不是根轮到结束轮的时间跨度。

用 `per_paper.csv` 的计数、轨迹及 `repair_tasks.csv` 复算；模型级挑战、修复成功率/轮数和回答长度按合并分子分母计算，不能把单篇比例简单平均替代。`paper_std` 描述跨论文波动，`valid_papers/valid_runs` 标记有效数量，`micro` 与分子分母同时保留以便核验。通用运行时报告中的最终全局覆盖、含线程挑战率、事件闭环和覆盖缺口闭环不作为本方案论文主值。

以下情况必须区分：

| 实际情况 | 记录与判定 |
| --- | --- |
| 有有效轮次但无新幻觉事件 | 平均事件数、含事件轮次比例及事件密度为真实 0；分类完整率为 N/A |
| 有实际修复任务但没有成功任务 | 该类修复成功率为真实 0，平均尝试数仍可计算 |
| 没有某类实际修复任务或没有执行普通挑战 | 相应成功率/平均轮数或挑战失败率为 N/A，该分支未覆盖 |
| 缺少报告、轨迹或必要判决内容 | 输出不完整或数据错误，不得解释为“没有事件” |
| 缺少运行时最终状态缓存 | 不影响报告＋轨迹的离线统计，不重建缓存 |

零分母在统计表记为空/N/A，不照抄运行时报告中的占位 0.0。`INCOMPLETE` 不计挑战失败；Text/MM 只对实际普通挑战分组。修复任务成功率是轨迹判决下的论文操作性指标，不宣称为独立金标验证的纠错准确率。

## 六 验收方式与交付材料

### 6.1 执行完整性

新模型批次以 `testing.json` 的“论文数 × 模型数 × 独立轮数”为计划任务数，要求正常退出、批次 `status=completed`，且统计 `task_counts.completed == task_counts.planned`、`incomplete=0`。历史结果没有计划清单时，计划总数未知，单列实际发现与完成任务数。每个完成任务须满足：

- 报告与轨迹的论文 ID、模型 ID 与计划一致。
- 报告 `summary.evaluation_status == "completed"` 且 `summary.failed == false`。
- 轨迹非空、轮次 ID 非空且唯一，报告总轮数等于轨迹实际轮数。
- 报告与轨迹可独立加载，轮内新事件结构有效、同任务事件 ID 唯一，修复来源和计数可复算；不要求最终状态缓存。
- 新模型实测输入签名未变化，正文与图片可读取，不用 mock 结果代替真实模型结果；离线回归与本机模拟测试仅验证程序和口径，不作为新模型性能实测。

执行完成只说明任务跑完，不代替下面的专项验收。

### 6.2 指标 1.3 专项验收

| 验收项 | 通过条件 | 证据 |
| --- | --- | --- |
| 四个核心维度 | 第一节四维均按论文口径输出主值、分子分母和执行实例；不得以最终状态类指标替代 | 自动 `four_dimensions.csv`、逐论文计数、轨迹与 `repair_tasks.csv` |
| 论文四类幻觉 | 按精确映射归并线程错误，四类及未映射计数与事件总数对账；不增加或猜测主类别 | `failure_category_by_model.csv`、逐事件 `paper_category` 及原始标签 |
| 不少于三级的细粒度分类 | 实际事件中有可解释、逐层细分的三级路径；同时报告完整性、缺失和退化情况，仅有三个字段不视为通过 | 分类明细、分类统计、原始判决和层级说明 |
| 分类与证据追溯 | 按轨迹新事件核对数量与四类归并，快照字段不冒充最终状态；身份、ID 或数据错误明确报错 | 自动分类 CSV/JSON 与来源轨迹；原证据可另查主图、正文或图片 |
| 错误修复过程 | 两类实际任务的分组、首次成功与尝试数可复算；无实际任务时 N/A，不用事件/缺口闭环替代 | `repair_tasks.csv`、根问题、实际跟进轮及其判决 |
| 文本与图像链路 | 有两类挑战执行记录，图像实际输入目标模型，并有接口调用成功的 `answer_mode=online_target_multimodal` 轮次；不要求回答正确 | 轨迹、附件引用和回答记录 |
| 结果可复核 | 清单、版本、输入签名、分子分母、日志与原始结果齐全，可重新统计 | `testing.json`、报告、CSV/JSON 和日志 |

未触发的事件、分类或修复分支标注“待补测”；文件损坏、计数不一致或错误归类标注“不通过”，不得兜底填零或补写标签。执行完整性与专项覆盖分别给出结论，二者均满足后判定本次通过。

### 6.3 交付材料

交付代码版本、环境、报告与轨迹、论文口径四维表、四类统计、修复任务和逐事件明细、运行或离线统计日志及补测说明。新模型批次另附非敏感配置说明、任务清单与输入签名。每个结论注明文件、事件或轮次，不只提供截图；无需补造未发布状态缓存。排除 `config.env`、密钥及其他私有配置。

实现入口：[默认启动](../scripts/start.py)、[已有数据集评测](../scripts/evaluate_dataset.py)、[离线统计](../scripts/summarize_results.py)、[论文统计指标](../scripts/statistics_metrics.py)、[四维呈现](../scripts/statistics_dimensions.py)、[分类归一化](../papergraph_demo/src/judge_result_normalizer.py)。配置见[中文 README](../README.zh-CN.md)、[English README](../README.md)、[API 配置](configuration.md)。实际版本固定记录在批次清单或离线核验记录中。

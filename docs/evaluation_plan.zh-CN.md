# PaperGraph-Bench 测评方案：指标 1.3 多维度幻觉评估指标体系

> **默认入口：Windows 执行 `.\start.ps1`，macOS/Linux 执行 `bash ./start.sh`。使用已有 `data/` 题包评测，自动导出四维结果及细粒度幻觉分类统计，无需另行执行分类脚本，不重新建图出题。运行前必须自行填写 Judge 和被测模型两组 API Key、Base URL、模型 ID。**

## 一 测评目标与四个核心维度

本方案仅对应**指标 1.3「多维度幻觉评估指标体系」**，按 2026 年 11 月里程碑要求组织测试：支持核心评估维度不少于 4 个，细粒度幻觉类型标注不少于 3 个层级。

测评对象为 **PaperGraph-Bench 面向科学论文理解的图引导多维度幻觉评测工具**，公开仓库为 [PaperGraphBench](https://github.com/furao0011/PaperGraphBench)。使用已有论文图、题包、正文和图片，固定 Judge 与参数，对被测模型开展多轮评测。新增论文、重建图题包不属于本次测试步骤。

本次验收的是工具能否提供可计算、可定位、可追溯的四维结果及分层分类，**不是要求被测模型达到某个覆盖率、幻觉率或修复率阈值**。

| 核心维度 | 测试方法 | 主要指标与计算口径 | 核验重点 |
| --- | --- | --- | --- |
| 1. 知识覆盖 | 执行覆盖问题，记录目标、已覆盖与遗漏的 KC（知识组件），检查多轮后的最终状态 | 全局 KC 覆盖率＝最终 lit 或 corrected 的 KC 数 / 全部 KC 数；宏阶段完成率为辅助指标 | 目标 KC 引用有效；判决能关联具体回答；最终覆盖状态与计数一致 |
| 2. 挑战诱发幻觉 | 执行错误前提、过度推断、错误关系等文本/图像挑战，将接受错误主张或错误关系的回答记录为挑战诱发幻觉 | 挑战诱发幻觉率＝出现错误或幻觉判决的挑战轮数 / 实际执行的宏与线程挑战总轮数；兼容原 `challenge_failure` 字段 | 挑战元数据、错误回答、判决及事件可追溯；`INCOMPLETE` 不算幻觉，未执行题目不计入分母 |
| 3. 幻觉行为 | 记录幻觉轮次、具体错误主张及分类，保存关联 KC 与来源轮次 | 幻觉轮次比例＝幻觉判决轮数 / 总轮数；事件数＝最终事件字典长度；事件密度＝事件数 / 总轮数 | 轮次与事件分别统计；按事件 ID 去重，已修复事件仍计入历史发生数 |
| 4. 错误修复 | 对幻觉事件和信息遗漏分别跟进，记录修复、未修复及预算耗尽结果 | 事件闭环率＝最终 resolved 事件数 / 全部事件数；遗漏补全率＝resolved 覆盖缺口数 / 全部覆盖缺口数 | 原始错误/缺口、跟进回答和最终状态能串联；执行了跟进不等于修复成功 |

各维度至少保留一个实际执行实例：问题、模型回答、Judge 判决、关联 KC 或事件；修复维度分别提供幻觉修复和遗漏补全实例，成功或预算耗尽均可验证记录功能。同一维度下的多个指标或别名不重复计作新维度。

挑战题的“失败”在本方案中指模型接受误导、作出与论文证据不符的回答，属于幻觉事件，并非泛指任务未完成或接口失败。`INCOMPLETE` 计入已执行挑战数，但不计入挑战诱发幻觉数；接口错误另列执行异常。旧报告和 CSV 保留 `challenge_failure_rate`、`challenge_failure` 等字段名，以兼容既有结果。

## 二 数据、环境与运行前配置

### 2.1 测试数据与实验设置

使用完整公开仓库 `data/` 中的 12 篇论文题包及其引用的 `rawPaper/` 图片。先以 ConVis 检查启动链路，再对全部论文正式测试。不同模型保持相同数据、Judge、提示词与修复预算；默认每模型一轮，可用 `--repeats 3` 独立重复三轮。

每批记录论文与模型清单、Git 版本、输入签名、轮数和参数。动态调度、幻觉数量及修复分支是否触发以实际轨迹为准，不假定各模型轨迹完全相同。

### 2.2 环境准备

安装 [uv](https://docs.astral.sh/uv/)，使用 Python 3.11+；启动脚本按 `uv.lock` 准备依赖。通过 API 推理不要求本地 GPU，测试机需能访问依赖源和所配置的模型服务，并有足够 API 额度。

```powershell
git clone https://github.com/furao0011/PaperGraphBench.git PaperGraphBench-test
Set-Location PaperGraphBench-test
git rev-parse HEAD
```

使用完整克隆，不使用仅检出部分论文的 sparse checkout。记录操作系统、CPU、内存、磁盘、Python 与 uv 版本。

### 2.3 必填 API 配置

> **仓库不提供可用密钥、模型服务或免费额度。首次启动会创建私有 `config.env` 并提示待填项；必须填写以下两组配置，再执行同一启动命令。已有配置不会被覆盖，配置缺失时不调用模型。**

| 服务 | 必填 Key | 必填 Base URL | 必填模型 ID | 能力要求 |
| --- | --- | --- | --- | --- |
| 评测 Judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` | OpenAI-compatible chat，支持结构化 JSON 输出 |
| 被测模型 | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` | OpenAI-compatible chat，支持文本与图片输入 |

Base URL 填服务商 API 根地址，不填网页或 `/chat/completions` 完整端点；模型 ID 必须在账号下可调用。同一服务可兼任两种角色，但两组字段仍须分别填写。环境变量优先于配置文件，模型比较期间固定 Judge。

其余项保持公开默认值即可。正式测试保持 `USE_ONLINE_EVAL=true`、`ALLOW_MOCK_EVAL=false`、`ALLOW_OFFLINE_FALLBACK=false`，不得用模拟或离线输出替代实测。默认 `CLAIM_VERIFY_ENABLED=false`，无需配置视觉生成、Embedding 或 OCR；开启额外 claim verification 才另需 Embedding。

`EVAL_MAX_TURNS=0` 不设置全局提前截断，但保留框架自身的有界跟进规则。不要为缩短测试而随意更改修复预算。配置、密钥和 Authorization 头不进入交付包。

## 三 测试步骤与运行命令

### 3.1 预览与单篇检查

在仓库根目录执行：

```powershell
# 预览论文、模型和输出计划，不调用 API、不创建配置或结果
.\start.ps1 --dry-run

# 填好 API 配置后，先测试一篇已有题包并统计
.\start.ps1 --paper-id ConVis
```

预览成功不代表 API 已连通；接口调用成功也不代表回答无幻觉或分类层级有效。单篇检查需确认报告与轨迹完整、模型接口正常；全量测试另开新批次，不把此轮混入全量结果。

### 3.2 正式评测并自动统计

```powershell
# 默认：全部已有题包，每模型一轮，结束后自动统计
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

运行中查看终端、批次 `main.log` 及 `logs/`。API 认证、限流、超时、格式及附件错误须保留并处理；失败任务不填零混入均值。

### 3.3 查看结果与独立重统计

启动会打印实际批次目录 `outputs/testing/<session>/`。四维结果和分类统计在同一次运行中自动生成：**先查看 `statistics/four_dimensions.csv`、`classification_paths.csv` 和 `classification_summary.csv`**，分别核对四维指标、实际分类数量与分类完整性。完整模型汇总和逐事件 CSV/JSON 也已导出，无需额外运行 PowerShell 提取步骤。

`four_dimensions.csv` 每模型六行，覆盖四个维度：知识覆盖率、挑战诱发幻觉率、幻觉轮次比例与事件密度、事件闭环率与遗漏补全率。每行保留 `dimension`、`metric`、`source_metric`、`mean`、`std`、`valid_papers`、`valid_runs`、`micro`、`numerator_sum`、`denominator_sum` 和任务 `status`，不用从完整指标表手工拼出四维结果。

已有结果可用同一统计入口重新导出四维结果和分类文件，**不调用模型 API**：

```powershell
# 将路径替换为终端打印的实际批次目录
uv run --locked python scripts/summarize_results.py --result-root outputs/testing/<session>
```

统计命令退出码 0 表示发现的任务均完整，2 表示批次不完整，1 表示数据或执行错误。有 `testing.json` 时会核对计划任务；无清单的旧目录只描述找到的结果，不能证明整套数据均已测完。声明完成的任务必须有对应最终状态文件；缺少该文件、身份或事件计数不一致均报错，不退回仅统计报告或轮内快照。

## 四 细粒度幻觉分类专项测试

### 4.1 分层定义

读取每个任务最终 `cache/eval_state_graph.json` 的顶层 `hallucination_events`，结构为 `{事件ID: 事件对象}`。轨迹中每轮的事件数组用于回溯，不重复累计为新事件。

| 层级 | 字段 | 核验内容 |
| --- | --- | --- |
| 第一级：类别族 | `hallucination_family` | 挑战诱发幻觉、推理线程幻觉、回顾退化等场景类别 |
| 第二级：具体类型 | `hallucination_type` | 结合失败模式区分错误前提接受、过度推断接受、错误关系接受等 |
| 第三级：细分标签 | `subtype` | Judge 实际提供的进一步细分标签及含义，而非仅检查字段存在 |

代码允许的字段路径示例为 `challenge_failure → challenge_false_premise_acceptance → false_premise`。这仅用于说明字段，**不是实测结果或三级分类已达标的证明**；验收使用实际事件及其判决，检查是否具有逐层细分的语义。

部分事件可能缺少 `subtype`，或类别族与类型重复；`subtype` 也可能只记录状态。需分别标注缺失、重复、同义重述或未实质细分，不能改写标签凑足三级。三个字段均非空或字符串不同，均不自动证明三个有效分类层级。

### 4.2 默认自动分类统计

默认启动会在评测后自动读取已完成任务的最终事件库，并与报告、轨迹核对身份、事件 ID、计数和最终状态。随后在 `statistics/` 中生成以下文件，不需要新增 API 配置或另行提取：

- `classification_events.csv`、`classification_events.json`：逐事件类别族、类型、细分标签、来源轮次、关联 KC、最终状态和追溯信息。
- `classification_paths.csv`：按模型及原始 `family/type/subtype` 组合列出 `event_count`，可直接在电子表格中查看各类事件数量。
- `classification_summary.csv`：按模型汇总事件数、分类字段完整性和缺失、重复等问题，便于直接查看。
- `classification_summary.json`：结构化分类汇总及实际分类路径分布。

逐事件明细保留 `round`、`model`、`paper_id`、`event_id` 及 `issues`；模型汇总中的 `event_count`、`three_fields_present_count`、`three_fields_present_rate`、`non_repeating_path_count` 和 `warning_counts` 分别记录事件数、三级字段完整性、不重复候选路径及问题计数。JSON 的 `classification_counts` 给出实际 `family/type/subtype` 组合及数量。

分类数量为各模型已完成任务跨论文、跨独立轮次的原始事件数累加，不是论文等权均值或幻觉率；轮数不同的批次不能只比较事件总数。四维表的 `mean` 仍按同一论文先跨轮平均、再按论文等权汇总。

统计保留原始标签，自动标记缺失字段、重复或未细分的候选路径，不填造细分标签。`three_fields_present` 和 `non_repeating_path` 仅表示字段完整或不重复，不自动证明其语义上满足三级细分；文件同时保留实际标签和原始事件，供验收复核。

未完成或缺失任务在任务统计中单列，不将其当作“零事件”纳入已完成任务的分类结果。已完成任务缺少最终状态文件或事件库损坏时，统计报错，不以空字典、报告计数或轮内事件数组代替。合法空事件库仍正常导出文件，分类完整率为 N/A。

### 4.3 分类、证据与修复核验

1. **计数一致：**自动统计核对逐任务事件字典长度与报告事件数，以及按类型、最终状态的分布与 `hallucination_by_type`、`hallucination_by_status` 是否一致。跨任务以“轮次、模型、论文、事件 ID”区分，不能仅按局部 ID 全批去重。
2. **层级有效：**列出实际三级路径，结合挑战 `target_failure_mode` 和 Judge 原始标签解释各层含义；将缺失、重复、同义重述或仅含状态的路径单列。分类专项报告包含总事件数、三级字段非空数及比例、有效三级路径实例和未细分记录。
3. **错误可定位：**由 `created_at_turn`、`source_question_id` 找到轨迹中的 `question_text`、`model_answer`、`judge_result`；由 `related_kc_ids` 找到主图 KC 的 `evidence` 及相关正文。事件 `source` 是记录来源，不是原文证据。图像事件还需核对资产引用与实际图片。
4. **修复可追踪：**由 `followup_turns`、`resolved_at_turn` 和最终 `status` 连接原始错误与后续处置，不用轮内事件快照代替最终状态。

无事件时分类完整率记 N/A，该分支为待补测，不记为 100%。若缺少分类或修复实例，可增加独立重复轮次或补充被测模型，保留独立批次和模型信息；不得编造错误、补写标签或混入原模型正式均值。

## 五 结果文件与指标口径

```text
outputs/testing/<session>/
  testing.json                       计划任务、参数、版本、输入签名和状态
  main.log、logs/                     主进程与逐论文日志
  round-01/evaluation/<model>/<paper>/
    evaluation_report.json           四维指标及计数
    dialogue_trajectory.json         问题、回答、Judge 判决及轮次
    cache/eval_state_graph.json      最终 KC、幻觉事件、缺口与修复状态
  round-02/ ...                      仅在独立重复时生成
  statistics/
    four_dimensions.csv              自动：四维专用表，每模型六行，优先查看
    summary.csv                      自动：完整模型指标与任务状态
    per_paper.csv                    自动：每轮每篇状态、指标和分子分母
    summary.json                     自动：结构化汇总
    classification_events.csv        自动：逐事件分类、状态与追溯明细
    classification_events.json       自动：相同明细的结构化版本
    classification_paths.csv         自动：按模型及原始分类路径的事件数量
    classification_summary.csv       自动：按模型的分类计数、完整性与问题统计
    classification_summary.json      自动：分类汇总及路径分布
```

`<model>` 为安全化后的目录名，完整模型 ID 以清单和报告为准。证据源为 `data/<paper_id>/master_graph.json`、`question_templates.json`、正文及引用图片；主图包含完整 KC bank，不依赖未发布建图缓存。

| 维度 | `evaluation_report.json` 主字段 | `four_dimensions.csv` 的 `metric` / 完整表列前缀 |
| --- | --- | --- |
| 知识覆盖 | `coverage_metrics.global_kc_coverage_rate` | `global_kc_coverage`，辅助 `macro_completion` |
| 挑战诱发幻觉 | `challenge_metrics.challenge_failure_rate`（兼容原字段名） | `challenge_induced_hallucination`；`source_metric` 及完整表沿用 `challenge_failure` |
| 幻觉行为 | `hallucination_metrics.hallucination_turn_rate`、`hallucination_event_rate` | `hallucination_turn_rate`、`hallucination_event_density` |
| 错误修复 | `hallucination_metrics.hallucination_repair_success_rate`、`detail_completion_metrics.detail_completion_rate` | `hallucination_event_closure`、`detail_completion` |

四维表的 `mean` 先对同一论文独立轮次取均值，再按论文等权汇总；`std` 为跨论文样本标准差，`valid_papers`、`valid_runs` 表示有效数量；`micro` 另列分子分母总和之比。完整 `summary.csv` 保留相应 `<指标>_mean` 等兼容列。比例用 0 到 1 表示，事件密度可大于 1。通用报告中的其他指标不增加本次验收范围。

逐项用 `per_paper.csv` 的 `<指标>_numerator`、`<指标>_denominator` 复算比率，并与报告及最终状态库对账；按报告四位小数精度核对。不能只确认 CSV 列存在而不检查分子、分母和执行实例。

以下情况必须区分：

| 实际情况 | 记录与判定 |
| --- | --- |
| 有有效轮次但无幻觉 | 幻觉轮次比例和事件密度为真实 0；事件闭环率、分类完整率为 N/A |
| 有幻觉事件但无已修复事件 | 事件闭环率为真实 0，不是 N/A |
| 无覆盖缺口或无执行挑战 | 相应补全率或挑战诱发幻觉率为 N/A，该分支未覆盖 |
| 缺少报告字段、最终状态文件或分类证据 | 输出不完整，不得解释为“没有事件”或合法 N/A |

源码报告可能在零分母时写 0.0，统计表将其转为空/N/A。挑战诱发幻觉率与抵抗率不保证互补，`INCOMPLETE` 不属于任一分子；分模态失败计数不等于失败率。事件闭环率按最终 resolved / 全部事件计算，不是独立金标验证的纠错准确率。

## 六 验收方式与交付材料

### 6.1 执行完整性

以 `testing.json` 的“论文数 × 模型数 × 独立轮数”为计划任务数，要求启动正常退出、批次 `status=completed`，且统计 `task_counts.completed == task_counts.planned`、`incomplete=0`。每个任务还须满足：

- 报告与轨迹的论文 ID、模型 ID 与计划一致。
- 报告 `summary.evaluation_status == "completed"` 且 `summary.failed == false`。
- 轨迹非空、轮次 ID 非空且唯一，报告总轮数等于轨迹实际轮数。
- 最终状态文件存在，身份、事件字典和报告统计一致，分类文件已自动生成；未完成任务不混入已完成任务统计。
- 输入签名未变化，正文和所需图片可读取，实测不含 mock 或 offline 输出。

执行完成只说明任务跑完，不代替下面的专项验收。

### 6.2 指标 1.3 专项验收

| 验收项 | 通过条件 | 证据 |
| --- | --- | --- |
| 四个核心维度 | 第一节四维均有有效输出、分母和执行实例；别名与同维多个指标不重复计数 | 自动 `four_dimensions.csv`、逐论文报告、轨迹与最终状态 |
| 不少于三级的细粒度分类 | 实际事件中有可解释、逐层细分的三级路径；同时报告完整性、缺失和退化情况，仅有三个字段不视为通过 | 分类明细、分类统计、原始判决和层级说明 |
| 分类与证据追溯 | 自动核对事件计数及类型/状态分布，明细能连接回答、KC 与论文证据；缺失状态或损坏数据明确报错 | 自动分类 CSV/JSON、最终事件库、轨迹、主图和原文/图片 |
| 错误修复过程 | 幻觉修复与遗漏补全均有实际跟进实例，能连到最终状态；成功或耗尽均可验证工具功能 | 跟进轮次、事件/缺口状态、闭环和补全指标 |
| 文本与图像链路 | 有两类挑战执行记录，图像实际输入目标模型，并有接口调用成功的 `answer_mode=online_target_multimodal` 轮次；不要求回答正确 | 轨迹、附件引用和回答记录 |
| 结果可复核 | 清单、版本、输入签名、分子分母、日志与原始结果齐全，可重新统计 | `testing.json`、报告、CSV/JSON 和日志 |

未触发的事件、分类或修复分支标注“待补测”；文件损坏、计数不一致或错误归类标注“不通过”，不得兜底填零或补写标签。执行完整性与专项覆盖分别给出结论，二者均满足后判定本次通过。

### 6.3 交付材料

交付代码版本、环境和非敏感模型配置说明、论文/任务清单、输入签名、四维结果表、分类明细及统计、代表性分类与修复轨迹、原始报告/状态库、日志及补测说明。每个结论注明对应文件、事件或轮次，不只提供汇总截图。排除 `config.env`、密钥及其他私有配置。

实现入口：[默认启动](../scripts/start.py)、[已有数据集评测](../scripts/evaluate_dataset.py)、[结果统计](../scripts/summarize_results.py)、[指标计算](../papergraph_demo/src/reporter.py)、[分类归一化](../papergraph_demo/src/judge_result_normalizer.py)、[事件与修复状态](../papergraph_demo/src/evaluation_hallucination_state.py)。配置及其他使用说明见[中文 README](../README.zh-CN.md)、[English README](../README.md)、[API 配置](configuration.md)。实际运行版本固定记录在批次清单中。

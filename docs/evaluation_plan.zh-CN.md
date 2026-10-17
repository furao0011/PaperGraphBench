# PaperGraph-Bench 已有数据集评测与结果统计方案

> **直接运行 `.\start.ps1`（macOS/Linux：`bash ./start.sh`），默认评测已有数据集并自动统计，不重新建图出题。只需在 `config.env` 填写 Judge、被测模型两组 API Key、Base URL 和模型 ID；默认无需视觉生成、Embedding 或 OCR 服务。**

## 一 测评对象与范围

本方案使用已发布的论文图和题包，测试模型的科学论文理解能力，并自动生成可查看和比较的统计结果：

```text
已有论文图、题包、正文与图片
    → 目标模型多轮作答
    → AI Judge 判决与状态更新
    → 逐论文报告与完整轨迹
    → 模型级统计表与逐轮逐论文明细
```

评测不修改已有图题包。`review_followup` 表示模型在对话中的回顾轮次。新增论文和重新构建基准属于可选构建流程，不是本方案的测试步骤。

| 项目 | 内容 |
| --- | --- |
| 成果名称 | PaperGraph-Bench 科学论文多模态基准构建与多轮诊断工具 |
| 公开仓库 | [PaperGraphBench](https://github.com/furao0011/PaperGraphBench) |
| 测评基线 | 使用包含 `test` 启动入口与 `scripts/summarize_results.py` 的公开版本；每批记录实际 Git 提交号 |
| 输入 | `data/<paper_id>/` 的图、题包、正文及其引用的 `rawPaper/` 图片 |
| 主要输出 | 逐论文报告、对话轨迹、按模型汇总表、任务清单和日志 |
| 模型角色 | 评测 Judge、被测模型 |
| 测试方式 | 固定已有数据包、Judge 与参数，按论文和模型评测后自动统计 |

## 二 与考核指标的对应关系

**建议主要对接指标 1.3「多维度幻觉评估指标体系」**，成果可表述为「面向科学论文理解的图引导多维度幻觉评测工具」。知识覆盖、挑战稳健性、幻觉行为、错误修复构成四个核心评估维度，结合事件分层记录提供细粒度诊断。指标 1.1 仅作为数据构建方面的辅助对应，当前发布规模不直接满足其 2 万条样本要求。

本次运行验证上述评测能力，并统计已有数据规模、模态和任务类型，不重新测试建图出题。软件完成条件与项目性能阈值分别记录。

| 图片考核项 | 本次测试内容及证据 | 判定边界 |
| --- | --- | --- |
| 1.1 多种类多模态幻觉定义与数据构建，里程碑要求样本 ≥2 万条、模态 ≥2 种、任务 ≥2 类 | 统计已有图题包的论文、KC、问题、推理线程及实际文本/图像任务 | 不将不同对象合并计为同一种独立样本；2 万条规模要求按项目统一样本定义另行核验 |
| 1.3 多维度幻觉评估指标体系，里程碑要求核心维度 ≥4 个、标注层级 ≥3 级 | 检查知识覆盖、挑战稳健性、幻觉行为、错误修复四个核心维度，补充推理线程表现；检查事件的 `hallucination_family / hallucination_type / subtype` 分层记录 | 验证多维输出与三级字段记录能力，并报告实际非空比例；字段层级与统一分类体系的完整性分别评估 |
| 1.2 幻觉检测准确率 | 可输出 AI 判决、事件及其关联证据，验证检测记录功能 | 本方案没有独立可信标签，不将 AI 自评比例换算成“检测准确率 ≥80%” |
| 1.7 输出幻觉率 | 报告本任务的幻觉轮次比例，观察不同模型的差异 | 以实测值为准，不预设 ≤9%；该口径不是 CHAIR，也不单独证明某种抑制算法达标 |

不确定性信号协同建模、校准、风险提前预警、外部检索 Recall@10、强化学习、SIUO 和平台集成不纳入本次独立软件测试。论文、专利、软著成果另以正式证明材料考核。

## 三 数据与测试设计

使用完整公开仓库中的 12 篇论文，单篇以 ConVis 做启动检查，全量以 12 篇做正式测试。

基线发布包的静态对象统计如下。测试直接使用这些已有数据；如果通过 `--data-root` 改用其他题包，应单独记录版本、论文清单和对象数量。

| 对象 | 发布包数量 |
| --- | ---: |
| 论文 | 12 |
| 宏阶段节点 | 99 |
| KC 节点 | 1,727 |
| 宏覆盖主问题 | 97 |
| 宏挑战问题 | 238 |
| 线程挑战问题 | 75 |
| 线程问题种子 | 183 |

线程种子不等于最终执行问题；跟进轮数由实际对话决定。正文是文本模态，曲线图、示意图和表格截图均属于图像模态，不另算成多种模态。

正式实验采用同一套已有题包。默认每个模型评测全部 12 篇、运行一轮；可用 `--repeats 3` 做三轮独立评测。Judge、提示词、修复预算及其他参数保持一致，记录模型完整 ID、服务商、测试日期、温度和并发数。动态对话轨迹及最终轮数可因模型表现而不同。

## 四 环境与 API 配置

### 4.1 运行环境

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows PowerShell 为本方案命令环境；Python 程序也可在相应 Linux 环境运行 |
| Python | 3.11 或更高，使用仓库声明的环境版本 |
| 依赖管理 | uv，按 `uv.lock` 安装 |
| 本地依赖 | Python 标准库及 `rich`，由 `uv sync --locked` 安装 |
| 推理资源 | Judge 与被测模型的服务 API；通过 API 推理时不要求本地 GPU |
| 网络 | 能连接 GitHub、依赖源及配置的模型 API，并具备足够调用额度 |
| 测试机记录 | 按实际配置记录操作系统、CPU、内存、可用磁盘、Python 和 uv 版本 |

在新的完整克隆中执行，不使用仅检出部分论文的 sparse checkout：

```powershell
git clone https://github.com/furao0011/PaperGraphBench.git PaperGraphBench-test
Set-Location PaperGraphBench-test
git rev-parse HEAD
```

依赖准备、配置初始化及预检由启动脚本完成，无需逐项执行。首次运行后按下表填写生成的 `config.env`，已有文件不会被覆盖。所有 API Key、Base URL 和模型 ID 均需使用实际服务配置。

### 4.2 模型角色与配置项

| 角色 | 配置项 | 接口及使用阶段 |
| --- | --- | --- |
| 评测 Judge | `API_KEY`、`BASE_URL`、`LLM_MODEL` | OpenAI-compatible chat，须支持所需结构化 JSON 输出 |
| 被测模型 | `EVAL_TARGET_API_KEY`、`EVAL_TARGET_BASE_URL`、`EVAL_TARGET_MODEL` | OpenAI-compatible chat；正式多模态测试须支持图片输入 |

模型 ID 使用实际获授权的完整标识。固定 Judge，不在模型比较期间更换；被测模型与 Judge 的角色分别记录。

`BASE_URL` 填 API 根地址，不填网页地址或 `/chat/completions` 完整端点。每组 Key、URL 和模型 ID 均由运行者从服务商获取，账号需有调用权限及额度。环境变量优先于 `config.env`。默认评测无需填写 `VISION_*`、`EMBED_*` 和 `PADDLE_OCR_*`；只有主动开启额外 claim verification 才需要 Embedding。

除上述两组 API 外，其余项目可保留默认值。与测试有关的设置如下：

```dotenv
USE_ONLINE_EVAL=true
PAPERGRAPH_PROGRESS=true
CLAIM_VERIFY_ENABLED=false
ALLOW_OFFLINE_FALLBACK=false
ALLOW_MOCK_EVAL=false
EVAL_MAX_TURNS=0
PAPERGRAPH_DATA_ROOT=data
RAW_PAPER_ROOT=rawPaper
EVAL_RESULT_ROOT=outputs/evaluation
```

`EVAL_MAX_TURNS=0` 表示不设置全局提前截断，不取消框架自身的有界跟进规则；默认跟进预算为 3 次，各模型保持一致。`CLAIM_VERIFY_ENABLED=false` 与公开默认一致。`test` 默认明确使用仓库 `data/`，不会因旧 `PAPERGRAPH_DATA_ROOT` 环境设置而改用其他数据；选择其他已有数据时使用 `--data-root`。输出位置由每批独立目录管理。

配置文件、密钥、Authorization 头均不进入测试报告、归档包或 Git 提交。报告只记录服务商、模型 ID、参数和非敏感端点信息。

## 五 测试步骤与运行命令

### 5.1 一键评测已有数据并统计

在仓库根目录执行下面一个命令：

```powershell
.\start.ps1
```

macOS/Linux 使用：

```bash
bash ./start.sh
```

首次执行会自动准备依赖、创建 `config.env` 并列出 Judge、被测模型两组必填信息；未配置时停止。**填写自己的 Key、Base URL 和模型 ID 后，再执行同一命令即可评测全部已有题包并导出统计表。** 若 Windows 执行策略拦截，使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`，仅对此次启动生效。

脚本依次检查配置和已有图题包、创建独立批次目录、记录输入签名、按模型和论文评测、验证输入未变化，最后生成逐论文与按模型统计。**不会重新建图、生成问题或修改已有数据文件。**

完整克隆默认选中 12 篇论文；实际论文、模型、轮数、Git 版本、预期任务和输入签名保存在 `testing.json`。不带子命令等同于 `test`，默认一轮、单任务并发；使用 `--repeats 3` 会额外消耗对应多轮 API 额度。只有显式使用 `run`、`build`、`questions` 或 `measure` 才会进入构建流程，例如 `.\start.ps1 run --paper-id ConVis`。

### 5.2 常用启动方式

| 测试目的 | Windows 命令 |
| --- | --- |
| 默认：全部已有题包，每模型一轮并统计 | `.\start.ps1` |
| 先测试一篇已有题包 | `.\start.ps1 test --paper-id ConVis` |
| 全部已有题包，每模型一轮并统计 | `.\start.ps1 test --workers 2` |
| 三轮独立测试并统计 | `.\start.ps1 test --repeats 3 --workers 2` |
| 自定义配置文件 | `.\start.ps1 test --config path/to/config.env` |
| 多模型使用同一已有题包 | `.\start.ps1 test --models model-a model-b --workers 2` |
| 指定另一套已有数据 | `.\start.ps1 test --data-root path/to/dataset` |
| 只预览测试计划，不调用模型 | `.\start.ps1 test --dry-run` |

多模型命令中的 `model-a`、`model-b` 必须替换为账号实际可调用的 ID，并配置对应服务；macOS/Linux 把 `.\start.ps1` 换为 `bash ./start.sh` 即可。

`--workers` 是论文级并发，单篇内部按顺序执行。所有轮次保持同一 Judge、题包和参数；重复轮次使用独立输出目录，不因缓存复用而误计为独立实验。不要用 `run --force` 做重复评测，以免重新生成图和问题。

### 5.3 过程检查与结果目录

启动时会打印本批目录 `outputs/testing/<session>/`，主要文件如下：

```text
<session>/
  testing.json              计划任务、输入签名和执行状态
  main.log                  主进程日志
  round-01/evaluation/      第一轮逐模型逐论文结果
  round-02/evaluation/      仅当 repeats >=2 时存在
  logs/                     各轮逐论文评测日志
  statistics/summary.csv    优先查看的按模型汇总表
  statistics/per_paper.csv  每轮每篇明细及完成状态
  statistics/summary.json   程序可读取的统计结果
```

检查每篇有非空宏阶段、KC 和有效主问题，正文和所需图片可读取，批次覆盖文本与图像及至少两类任务；至少存在一轮成功的 `answer_mode=online_target_multimodal`，才能确认图片输入链路已执行。未触发的模态或修复分支按实际情况记录，不推断其已经通过。

正常完成后优先查看 `statistics/summary.csv`，再按需查看逐论文明细和原始轨迹。API 认证、限流、超时、格式错误及附件缺失保留在日志中；失败、中断或缺失任务单列，不填零混入均值。再次执行 `test` 会创建新批次，不把上一批结果复用为独立实验。

需要复核旧结果时直接使用第七节的独立统计命令，不必再次调用模型。新建图题包的操作参见 README 的可选构建部分。

## 六 结果文件与指标口径

### 6.1 产物位置

| 产物 | 路径 |
| --- | --- |
| 论文图 | `<数据根>/<paper_id>/master_graph.json` |
| 题包 | `<数据根>/<paper_id>/question_templates.json` |
| 正文与评测上下文 | 同目录 `paper_clean_text.md`、`paper_eval_context.md` |
| 多模态资产及解释 | 同目录 `multimodal_assets.json`、`multimodal_asset_explanations.json` |
| 逐论文评测报告 | `<结果根>/<model>/<paper_id>/evaluation_report.json` |
| 完整对话轨迹 | 同目录 `dialogue_trajectory.json` |
| 最终事件和状态库 | 同目录 `cache/eval_state_graph.json` |
| 评测断点 | 同目录 `cache/evaluation_checkpoint.json` |
| 状态可视化 | 同目录 `cache/final_state_graph.mmd`、`cache/final_thread_state_graph.mmd` |
| 自动统计 | `<session>/statistics/summary.csv`、`per_paper.csv`、`summary.json` |
| 批次日志 | `<session>/main.log`、`<session>/logs/` |

默认“数据根”为仓库 `data/`，“结果根”为 `outputs/testing/<session>/round-XX/evaluation`。发布主图内含完整 KC bank，评测不依赖未发布的建图缓存；本批评测状态缓存用于追溯。`eval_result/` 是历史结果，可单独重统计，不作为本次新测试输出。

### 6.2 核心指标

下表字段均来自 `evaluation_report.json`；比例原始值采用 0 到 1 表示，事件密度除外。展示百分比时乘 100。方向仅表示同条件比较偏好，不是预先承诺的达标结果。

| 维度与指标 | JSON 字段 | 定义 | 方向 |
| --- | --- | --- | --- |
| 全局 KC 覆盖率 | `coverage_metrics.global_kc_coverage_rate` | 最终状态为 lit 或 corrected 的 KC 数 / 全部 KC 数 | ↑ |
| 目标 KC 覆盖率 | `coverage_metrics.target_kc_coverage_rate` | 已覆盖的目标 KC 数 / 目标 KC 总数 | ↑ |
| 关键 KC 覆盖率 | `coverage_metrics.critical_kc_coverage_rate` | 已覆盖 critical KC 数 / 全部 critical KC 数 | ↑ |
| 宏阶段完成率 | `coverage_metrics.macro_completion_rate` | 非空目标 KC 集全部覆盖的宏阶段数 / 全部宏阶段数 | ↑ |
| 挑战失败率 | `challenge_metrics.challenge_failure_rate` | AI 判失败的已执行挑战轮数 / 已执行宏与线程挑战总轮数 | ↓ |
| 挑战抵抗率 | `challenge_metrics.challenge_resistance_rate` | 明确判为 resisted 的挑战轮数 / 已执行挑战总轮数 | ↑ |
| 幻觉轮次比例 | `hallucination_metrics.hallucination_turn_rate` | 判为 HALLUCINATION、MISLED、GLOBAL_OVERCLAIM 或 REFUSE_TO_CORRECT 的轮数 / 总轮数 | ↓ |
| 幻觉事件数 | `hallucination_metrics.hallucination_event_count` | 最终事件库中的去重事件数，包含后来已修复的事件 | 记录并结合轮数 |
| 幻觉事件密度 | `hallucination_metrics.hallucination_event_rate` | 幻觉事件数 / 总轮数；同一轮可有多个事件，数值可大于 1 | ↓ |
| 幻觉事件闭环率 | `hallucination_metrics.hallucination_repair_success_rate` | 最终标为 resolved 的事件数 / 全部幻觉事件数 | ↑ |
| 遗漏补全率 | `detail_completion_metrics.detail_completion_rate` | resolved 覆盖缺口数 / 全部覆盖缺口数 | ↑ |
| 桥接推理成功率 | `thread_metrics.bridge_reasoning_success_rate` | 成功桥接轮数 / 实测桥接轮数 | ↑ |
| 推理线程成功率 | `thread_metrics.thread_success_rate` | 成功或复查一致等成功状态的线程数 / 图中全部线程数 | ↑ |
| 总对话轮数 | `summary.total_turns` | 轨迹中的轮数 | 记录 |

必须遵循以下口径：

- 挑战失败率与抵抗率不保证互补，INCOMPLETE 等状态可能不属于两者之一。
- 事件闭环率是当前调度器与 AI Judge 标记的事件级结果，不是独立金标验证的纠错准确率；一个旧事件闭环后仍可能出现新事件。优先使用最终事件库的整体闭环率，不把轮内事件快照当作最终修复状态。
- 线程执行完成率不等于推理成功率；未被调度的线程也可能在图中总数内，应同时报告已测数量。
- 源码在空分母时通常输出 0.0。汇总表中应标为 N/A，并保留原始文件及分母；无挑战、无事件或无桥接测试不能解释为“零错误”或“修复能力为零”。
- `graph_coverage_rate` 是全局覆盖率别名，不重复计作独立维度；`hallucination_count` 是幻觉轮数别名，不当作事件数。

### 6.3 模态与细粒度诊断

分文本与图像挑战统计时，从 `dialogue_trajectory.json` 筛选 `challenge_question` 和 `thread_challenge_question`。分组顺序与 reporter 一致：优先使用 `question_metadata.modality_pool`，其次附件资产类型，再其次多模态标记，最后为 text。对每组分别计算“失败轮数 / 该组执行轮数”。

报告中的 `challenge_failure_by_modality` 只有失败计数，不能直接当失败率，也不能把所有挑战总数当成每个模态的分母。

幻觉类型与状态分布读取 `hallucination_by_type`、`hallucination_by_status`。分层记录从最终事件库取 `hallucination_family`、`hallucination_type`、`subtype`，统计三字段均非空的事件数及其占比；没有事件时记 N/A。保存事件 ID、来源轮次、关联 KC 和证据引用，支持定位到具体回答。

## 七 功能验收与结果汇总

### 7.1 功能通过条件

| 测试项 | 通过条件 |
| --- | --- |
| 环境与接口 | 本地回归和配置检查通过；实际 API 请求成功，无 mock 或 offline 结果 |
| 已有数据加载 | 12 篇主图均可解析，宏阶段和 KC 非空，原文及所需图片可读取 |
| 题包加载 | 题包可解析，主问题 ID 有效，目标 KC 引用有效；实际任务类型及模态覆盖有记录 |
| 多模态链路 | 有实际图片附件进入目标模型，存在成功的 online_target_multimodal 轮次 |
| 多轮评测 | 每个计划中的论文和模型组合均有完整报告及轨迹，满足下述一致性条件 |
| 指标输出 | 覆盖、挑战、幻觉、修复四个维度均有对应输出；未触发的事件指标注明 N/A |
| 可追溯性 | 版本、输入哈希、非敏感参数、日志、图题包及原始轨迹齐全 |

一份有效完成报告必须同时满足：

1. 报告与轨迹的 `paper_id`、`target_model` 与计划一致。
2. `summary.evaluation_status == "completed"` 且 `summary.failed == false`。
3. 轨迹非空，`turn_id` 非空且唯一，`summary.total_turns` 等于轨迹实际轮数。
4. 每个模型每轮的计划论文清单完整；停止、失败、中断或缺失结果单列，不能填零混入均值。

计划任务数为 `12 × 被测模型数 × 独立轮数`，任务完成率为“符合上述条件的任务数 / 计划任务数”。功能测试目标为计划任务全部完成；这是软件执行目标，不是模型覆盖率或正确率阈值。

### 7.2 自动统计及旧结果重统计

`test` 正常完成后自动导出三份统计文件，无需手工提取 JSON：

- `summary.csv`：每个模型的主要指标均值、标准差、有效论文数及任务完整性。
- `per_paper.csv`：逐轮逐论文的状态、分子分母、指标值。
- `summary.json`：相同统计信息的结构化版本。

CSV 使用带 BOM 的 UTF-8，便于在 Windows Excel 中打开含中文的论文 ID；比例值以 0 到 1 表示，事件密度可大于 1。可能被电子表格识别为公式的文本会加保护前缀，JSON 保留原始模型 ID。

在 `summary.csv` 中先查看 `model`、`status`、`completed_tasks`、`planned_tasks`；指标的 `_mean` 是论文等权均值，`_std` 是跨论文样本标准差，`_valid_papers` 和 `_valid_runs` 是该指标的有效数量，`_micro` 是另列的分子分母总和之比。

| 汇总列前缀 | 指标 |
| --- | --- |
| `global_kc_coverage` | 全局 KC 覆盖率 |
| `macro_completion` | 宏阶段完成率 |
| `challenge_failure` | 挑战失败率 |
| `hallucination_turn_rate` | 幻觉轮次比例 |
| `hallucination_event_density` | 幻觉事件密度 |
| `hallucination_event_closure` | 幻觉事件闭环率 |
| `detail_completion` | 遗漏补全率 |

例如 `global_kc_coverage_mean`、`global_kc_coverage_std`、`global_kc_coverage_valid_papers` 分别是 KC 覆盖率均值、标准差和有效论文数。`hallucination_events_sum` 为已完成任务观察到的事件总数，不是平均每篇事件数。

已有报告可单独重统计，不需要任何模型 API 配置：

```powershell
uv run --locked python scripts/summarize_results.py --result-root outputs/evaluation
```

`--result-root` 可以是某批 `outputs/testing/<session>`、某轮 `evaluation` 目录或 `eval_result` 等历史根目录。统计程序读取报告与轨迹的一致性，并在有任务清单时核对缺失任务；不完整批次会标注并返回非零退出码，不能视为已全部完成。

没有任务清单的旧目录只统计实际发现的结果，计划总数及完成率为未知。发现已完成报告内容损坏、身份不一致或重复任务会明确报错，不静默略过。

### 7.3 汇总与交付格式

主结果按论文等权宏平均，报告有效论文数及跨论文标准差；另给分子、分母总和计算的微平均。零分母论文不进入该项有效样本数。幻觉事件数汇总求和，同时可报告每篇平均数，但二者不得混用。

独立重复时先对同一“论文 × 模型”的各轮取均值，再按论文等权汇总，并另报跨轮波动。模型比较使用相同论文集合；若某模型缺少论文结果，先补跑，不将不完整集合的平均值直接并列。

| 模型 | 完成任务数/计划数 | 有效论文数 | KC 覆盖率 | 宏完成率 | 挑战失败率 | 幻觉轮次比例 | 事件密度 | 事件闭环率 | 遗漏补全率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 按实际模型填写 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

需要文本/图像挑战分表或幻觉分类细分时，按第六节从原始报告和轨迹另行统计；自动导出的三份文件侧重上述模型级核心指标和任务完整性。运行时间记录测试起止时间；API token 与费用采用服务商账单或调用记录，当前报告没有统一 token 成本字段，不从轮数推算。

交付材料包括：代码版本、环境与模型清单、非敏感参数、论文与任务清单、冻结图题包及输入哈希、逐任务报告与轨迹、状态库、CSV 汇总、阶段日志、异常及补跑记录。归档前排除 `config.env`、其他私有配置和凭据。

## 八 公开实现索引

测试入口见 `start.ps1 test`、`start.sh test`、`scripts/evaluate_dataset.py` 和 `scripts/summarize_results.py`。以下链接固定到核心评测实现版本，用于核对沿用的参数及指标口径；实际运行版本由批次清单记录：

- [中英文运行说明](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/README.zh-CN.md)
- [配置要求](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/docs/configuration.md)
- [统一命令入口](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_cli.py)
- [计划与数据选择](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_plans.py)
- [进程调度与日志](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_runner.py)
- [指标计算](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/reporter.py)
- [幻觉事件字段](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/judge_result_normalizer.py)
- [完成报告校验](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_artifacts.py)

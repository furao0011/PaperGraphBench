# PaperGraph-Bench 建图与评测方案

> **运行前必须提供自己的 API Key、Base URL 和模型 ID，并填写根目录 `config.env`：文本/Judge、视觉、Embedding、被测模型共四组。本仓库不提供模型服务或密钥。首次启动脚本会创建空配置并列出待填项，填好后再次执行同一命令即可。**

## 一 测评对象与范围

本方案用于测试科学论文多模态基准的自动构建及多轮评测能力，执行链路为：

```text
已解析论文正文与图片
    → 论文图构建与自动验证
    → 图关联问题生成与求解器筛选
    → 目标模型多轮作答
    → AI Judge 判决与状态更新
    → 覆盖、挑战、幻觉、修复及推理指标
```

测评从已解析的论文正文和图片开始，覆盖建图、出题与多轮评测。`review_followup` 表示模型在对话中的回顾轮次。

| 项目 | 内容 |
| --- | --- |
| 成果名称 | PaperGraph-Bench 科学论文多模态基准构建与多轮诊断工具 |
| 公开仓库 | [PaperGraphBench](https://github.com/furao0011/PaperGraphBench) |
| 测评基线 | 使用包含 `start.ps1`、`start.sh` 和 `scripts/run_measurement.py` 的公开版本；每批记录实际 Git 提交号 |
| 输入 | `rawPaper/<paper_id>/` 中的 Markdown 正文和图片 |
| 主要输出 | 论文图、问题包、对话轨迹、AI 判决、评测报告和运行日志 |
| 模型角色 | 文本构建模型兼 Judge、视觉模型、Embedding 模型、被测模型 |
| 测试方式 | 固定软件版本与非敏感参数，自动建图和出题，冻结题包后进行逐论文、逐模型评测 |

## 二 与考核指标的对应关系

本工作主要支撑图片中指标 1.1 的多模态数据构建能力，以及指标 1.3 的多维度幻觉评估能力。软件功能通过条件与项目数量、性能阈值分别记录，不相互替代。

| 图片考核项 | 本次测试内容及证据 | 判定边界 |
| --- | --- | --- |
| 1.1 多种类多模态幻觉定义与数据构建，里程碑要求样本 ≥2 万条、模态 ≥2 种、任务 ≥2 类 | 自动生成论文图、证据关联 KC、覆盖问题、挑战问题及推理线程；记录文本与图像两类输入 | 分别报告论文数、KC 数、各类问题数和实际执行轮数，不合并计为同一种独立样本。2 万条规模要求按项目统一样本定义另行核验 |
| 1.3 多维度幻觉评估指标体系，里程碑要求核心维度 ≥4 个、标注层级 ≥3 级 | 检查知识覆盖、挑战稳健性、幻觉行为、错误修复四个核心维度，补充推理线程表现；检查事件的 `hallucination_family / hallucination_type / subtype` 分层记录 | 验证多维输出与三级字段记录能力，并报告实际非空比例；字段层级与统一分类体系的完整性分别评估 |
| 1.2 幻觉检测准确率 | 可输出 AI 判决、事件及其关联证据，验证检测记录功能 | 本方案没有独立可信标签，不将 AI 自评比例换算成“检测准确率 ≥80%” |
| 1.7 输出幻觉率 | 报告本任务的幻觉轮次比例，观察不同模型的差异 | 以实测值为准，不预设 ≤9%；该口径不是 CHAIR，也不单独证明某种抑制算法达标 |

不确定性信号协同建模、校准、风险提前预警、外部检索 Recall@10、强化学习、SIUO 和平台集成不纳入本次独立软件测试。论文、专利、软著成果另以正式证明材料考核。

## 三 数据与测试设计

使用完整公开仓库中的 12 篇论文，单篇以 ConVis 做启动检查，全量以 12 篇做正式测试。

基线发布包的静态对象统计如下。重新建图与出题后可能因模型生成而变化，应重新统计并冻结本次产物，不要求与发布包数量逐项相等。

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

正式实验采用同一冻结题包。每个被测模型覆盖全部 12 篇；建议独立运行 3 轮。Judge、构建模型、提示词、修复预算及其他参数保持一致，记录模型完整 ID、服务商、测试日期、温度和并发数。不同目标模型可形成不同的动态对话轨迹，不要求最终轮数相同。

## 四 环境与 API 配置

### 4.1 运行环境

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows PowerShell 为本方案命令环境；Python 程序也可在相应 Linux 环境运行 |
| Python | 3.11 或更高，使用仓库声明的环境版本 |
| 依赖管理 | uv，按 `uv.lock` 安装 |
| 本地依赖 | Python 标准库及 `rich`，由 `uv sync --locked` 安装 |
| 推理资源 | 对应文本、视觉、Embedding 和被测模型的服务 API；通过 API 推理时不要求本地 GPU |
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
| 文本构建模型兼 Judge | `API_KEY`、`BASE_URL`、`LLM_MODEL` | OpenAI-compatible chat；建图、问题生成、AI 判决，须支持所需结构化 JSON 输出 |
| 视觉模型 | `VISION_API_KEY`、`VISION_BASE_URL`、`VISION_MODEL` | 图片理解与图像相关问题生成，须支持图片输入 |
| Embedding 模型 | `EMBED_API_KEY`、`EMBED_BASE_URL`、`EMBED_MODEL` | 建图所需向量服务 |
| 被测模型 | `EVAL_TARGET_API_KEY`、`EVAL_TARGET_BASE_URL`、`EVAL_TARGET_MODEL` | OpenAI-compatible chat；正式多模态测试须支持图片输入 |

模型 ID 使用实际获授权的完整标识。固定 Judge，不在模型比较期间更换；被测模型与 Judge 的角色分别记录。

`BASE_URL` 填 API 根地址，不填 `/chat/completions` 或 `/embeddings` 完整端点。环境变量优先于 `config.env`，测试前应排除无关旧变量的干扰。Embedding 的维度等额外参数应与所用服务匹配，并记录于非敏感参数清单。

除 API 配置外，其余项目可保留模板默认值。完整测评脚本会自动为每批创建独立目录，不改写发布数据包：

```dotenv
USE_ONLINE_KC_EXTRACT=true
MULTIMODAL_ENABLED=true
MULTIMODAL_KC_ENABLED=true
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

`EVAL_MAX_TURNS=0` 表示不设置全局提前截断，不取消框架自身的有界跟进规则；当前任务的默认跟进预算为 3 次，各模型保持一致。`CLAIM_VERIFY_ENABLED=false` 与公开默认一致；开启额外 claim verification 应单列实验条件，不能和默认条件混报。本方案不调用 OCR，故无需填写 `PADDLE_OCR_*`。

配置文件、密钥、Authorization 头均不进入测试报告、归档包或 Git 提交。报告只记录服务商、模型 ID、参数和非敏感端点信息。

## 五 测试步骤与运行命令

### 5.1 一键启动完整测评

在仓库根目录执行下面一个命令：

```powershell
.\start.ps1 measure --workers 2
```

macOS/Linux 使用：

```bash
bash ./start.sh measure --workers 2
```

首次执行会自动准备依赖、创建 `config.env` 并列出四组必填 API 信息；未配置时会停止。**填写自己的 Key、Base URL 和模型 ID 后，再执行同一命令即可完成整批测评。** 不需要分别运行建图、出题和评测命令。若 Windows 执行策略拦截，使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 measure --workers 2`，仅对此次启动生效。

启动脚本依次完成配置预检、创建本批独立目录、保存源论文快照、建图、自动出题、冻结输入并计算 SHA256、独立评测三轮。所有模型和轮次使用同一图题包；任何阶段失败立即停止，不以模拟结果替代。

完整克隆应选中 12 篇论文；实际选中清单、Git 版本、模型 ID、轮数、并发数和执行状态保存在 `measurement.json`。运行会消耗 API 额度，默认三轮；如只需一轮，使用 `--repeats 1`。

### 5.2 常用启动方式

| 测试目的 | Windows 命令 |
| --- | --- |
| 先测一篇并只评测一轮 | `.\start.ps1 measure --paper-id ConVis --repeats 1` |
| 完整数据集，建图出题一次并评测三轮 | `.\start.ps1 measure --workers 2` |
| 自定义配置文件 | `.\start.ps1 measure --config path/to/config.env --workers 2` |
| 多模型使用同一冻结题包 | `.\start.ps1 measure --models model-a model-b --workers 2` |
| 只预览计划，不调用模型或创建本批结果 | `.\start.ps1 measure --dry-run` |
| 复测已发布题包，不重新建图出题 | `.\start.ps1 evaluate --all --workers 2` |

多模型命令中的 `model-a`、`model-b` 必须替换为账号实际可调用的 ID，并配置对应服务；macOS/Linux 把 `.\start.ps1` 换为 `bash ./start.sh` 即可。发布题包复测与重新构建测评分表记录。

`--workers` 是论文级并发，单篇内部按顺序执行。所有轮次保持同一 Judge、题包和参数；重复轮次使用独立输出目录，不因缓存复用而误计为独立实验。不要用 `run --force` 做重复评测，以免重新生成图和问题。

### 5.3 过程检查与结果目录

启动时会打印本批目录 `outputs/measurement/<session>/`，主要文件如下：

```text
<session>/
  measurement.json          批次清单和执行状态
  main.log                  主进程日志
  rawPaper/                 本批源论文与图片快照
  dataset/                  一次构建后冻结的图题包
  frozen_inputs.json        输入 SHA256 清单
  round-01/evaluation/      第一轮逐模型逐论文结果
  round-02/evaluation/      第二轮结果
  round-03/evaluation/      第三轮结果
  logs/                     建图 出题及各轮评测日志
```

检查每篇有非空宏阶段、KC 和有效主问题，正文和所需图片可读取，批次覆盖文本与图像及至少两类任务；至少存在一轮成功的 `answer_mode=online_target_multimodal`，才能确认图片输入链路已执行。未触发的模态或修复分支按实际情况记录，不推断其已经通过。

正常完成后按第六、七节读取结果和验收。API 认证、限流、超时、格式错误及附件缺失保留在主日志和对应阶段日志中；失败、中断或缺失任务单列，不填零混入均值。再次启动 `measure` 会创建新批次；需要处理上一批失败时应使用其日志、快照和记录的路径进行定位，不把新批次冒充上一批的断点恢复。

需要分阶段调试时，可使用 README 保留的 `run.py build / questions / evaluate` 命令，并将数据根、源论文根及结果根明确指向相应批次。

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
| 阶段日志 | `outputs/logs/<paper_id>/` |

完整测评中的“数据根”为 `<session>/dataset`，“结果根”为 `<session>/round-XX/evaluation`；各阶段日志位于 `<session>/logs/`。常规 `start.ps1` 或 `run.py` 运行仍使用 `outputs/logs/`，可通过 `PAPERGRAPH_LOG_ROOT` 自定义。发布主图内含完整 KC bank，评测不依赖未发布的建图缓存；本批缓存保留用于追溯和恢复。`eval_result/` 是历史结果，不作为本次输出。

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
| 自动建图 | 12 篇均生成可解析主图，宏阶段和 KC 非空，原文及所需图片可读取 |
| 自动出题 | 12 篇均生成可解析题包，主问题 ID 有效，目标 KC 引用有效；统计实际任务类型与模态覆盖 |
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

### 7.2 提取逐论文结果

以下 PowerShell 示例读取一个正式结果根并生成明细 CSV。空分母导出为空，展示时标为 N/A；状态不完整的行保留记录，不纳入能力指标均值。

```powershell
$ResultRoot = "outputs/measurement/<session>/round-01/evaluation"
function Get-Rate($Numerator, $Denominator) {
    if ([double]$Denominator -le 0) { return $null }
    return [math]::Round([double]$Numerator / [double]$Denominator, 4)
}
$files = @(Get-ChildItem -LiteralPath $ResultRoot -Recurse -File -Filter "evaluation_report.json")
if ($files.Count -eq 0) { throw "No evaluation reports found." }
$rows = foreach ($file in $files) {
    $r = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $c = $r.coverage_metrics
    $h = $r.hallucination_metrics
    $q = $r.challenge_metrics
    [pscustomobject]@{
        Paper = $r.paper_id
        Model = $r.target_model
        Status = $r.summary.evaluation_status
        Failed = $r.summary.failed
        Turns = $r.summary.total_turns
        KCTotal = $c.global_kc_total
        KCCovered = $c.global_kc_lit_count
        KCCoverage = Get-Rate $c.global_kc_lit_count $c.global_kc_total
        TargetCoverage = Get-Rate $c.target_kc_lit_count $c.target_kc_total
        MacroTotal = $r.macro_metrics.macro_total
        MacroCompletion = if ([int]$r.macro_metrics.macro_total -gt 0) { $c.macro_completion_rate } else { $null }
        ChallengeN = $q.challenge_total
        ChallengeFailed = $q.challenge_failure_count
        ChallengeFailure = Get-Rate $q.challenge_failure_count $q.challenge_total
        HallucinationTurns = $h.hallucination_turn_count
        HallucinationTurnRate = Get-Rate $h.hallucination_turn_count $r.summary.total_turns
        Events = $h.hallucination_event_count
        EventDensity = Get-Rate $h.hallucination_event_count $r.summary.total_turns
        EventClosure = Get-Rate ([int]$h.hallucination_by_status.resolved) $h.hallucination_event_count
        DetailCompletion = Get-Rate $r.detail_completion_metrics.coverage_gap_resolved_count $r.detail_completion_metrics.coverage_gap_count
    }
}
$rows | Export-Csv -LiteralPath "$ResultRoot/paper_metrics.csv" -NoTypeInformation -Encoding UTF8
```

该命令不替代前述轨迹一致性和计划完整性检查：报告缺失的任务不会自动出现在 CSV 中，须按计划清单补列。

### 7.3 汇总与交付格式

主结果按论文等权宏平均，报告有效论文数及跨论文标准差；另给分子、分母总和计算的微平均。零分母论文不进入该项有效样本数。幻觉事件数汇总求和，同时可报告每篇平均数，但二者不得混用。

独立重复时先对同一“论文 × 模型”的各轮取均值，再按论文等权汇总，并另报跨轮波动。模型比较使用相同论文集合；若某模型缺少论文结果，先补跑，不将不完整集合的平均值直接并列。

| 模型 | 完成任务数/计划数 | 有效论文数 | KC 覆盖率 | 宏完成率 | 挑战失败率 | 幻觉轮次比例 | 事件密度 | 事件闭环率 | 遗漏补全率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 按实际模型填写 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

另附文本/图像挑战分表、推理结果分表、幻觉分类与状态分布、异常任务清单及各指标分母。运行时间记录每阶段和总起止时间；API token 与费用采用服务商账单或调用记录，当前报告没有统一 token 成本字段，不从轮数推算。

交付材料包括：代码版本、环境与模型清单、非敏感参数、论文与任务清单、冻结图题包及输入哈希、逐任务报告与轨迹、状态库、CSV 汇总、阶段日志、异常及补跑记录。归档前排除 `config.env`、其他私有配置和凭据。

## 八 公开实现索引

新启动入口见 `start.ps1`、`start.sh`、`scripts/start.py` 和 `scripts/run_measurement.py`。以下链接固定到核心评测实现版本，用于核对沿用的参数及指标口径；实际运行版本由批次清单记录：

- [中英文运行说明](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/README.zh-CN.md)
- [配置要求](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/docs/configuration.md)
- [统一命令入口](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_cli.py)
- [计划与数据选择](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_plans.py)
- [进程调度与日志](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_runner.py)
- [指标计算](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/reporter.py)
- [幻觉事件字段](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/judge_result_normalizer.py)
- [完成报告校验](https://github.com/furao0011/PaperGraphBench/blob/5f5f47dfbb6555d3f22959ff861f35da97a0fe40/papergraph_demo/src/pipeline_artifacts.py)

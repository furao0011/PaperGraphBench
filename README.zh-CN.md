# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench 通过图引导的多轮阅读任务评测模型对科学论文的理解能力。论文图将文章的论证结构、有证据支撑的知识组件、推理关系及图表连接起来，据此生成覆盖、误导挑战和推理问题，并在对话过程中持续观察幻觉及其修复情况。

这一过程能够反映模型是否理解论文的整体内容，是否忠实于图文证据，以及能否在后续阅读中纠正错误。

## 快速开始：使用已有数据集测试模型

仓库 `data/` 已提供 **12 篇论文题包**。默认启动脚本使用这些题包评测你的模型，并自动生成统计结果。

### 1. 准备环境与 API 配置

安装 [uv](https://docs.astral.sh/uv/)，使用 Python 3.11+。完整克隆仓库后，在仓库根目录运行：

```powershell
.\start.ps1
```

macOS/Linux 使用：

```bash
bash ./start.sh
```

脚本会准备运行依赖。首次启动时会创建 `config.env`，并提示需要填写的配置。

> **测试前，请在 `config.env` 中填写下面两组服务的 API Key、API 根地址和可调用模型 ID，再次执行启动命令。实际调用会消耗你的服务商额度，并可能产生费用。**

| 服务用途 | API Key | API 根地址 Base URL | 模型 ID |
| --- | --- | --- | --- |
| 评测 Judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` |
| 被测模型 | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` |

两组服务均使用 OpenAI 兼容接口。Judge 需支持结构化 JSON 输出，被测模型需支持图片输入。Base URL 填写服务商提供的 API 根地址，例如 `https://your-provider.example/v1`。默认已有数据集测试使用上述两组服务，更多选项见[配置说明](docs/configuration.md)。

### 2. 运行评测

配置完成后，执行 `.\start.ps1` 即可对全部 12 篇论文评测一轮并生成结果表。以下命令可用于单篇试运行、重复测试和模型对比：

```powershell
# 先测试一篇论文
.\start.ps1 test --paper-id ConVis

# 全部论文独立评测三轮，同时运行两个任务
.\start.ps1 test --workers 2 --repeats 3

# 对比配置的被测服务中可调用的多个模型
.\start.ps1 test --models model-a model-b --workers 2
```

macOS/Linux 将 `.\start.ps1` 替换为 `bash ./start.sh`。使用 `--dry-run` 预览测试计划，`--data-root` 选择其他数据集，`--config` 选择其他配置文件。若 Windows 执行策略拦截，可使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`。

### 3. 查看结果

每次测试会打印结果目录：`outputs/testing/<session>/`。优先查看以下文件：

| 文件 | 内容 |
| --- | --- |
| `statistics/four_dimensions.csv` | 四个评估维度的测试结果 |
| `statistics/failure_category_by_model.csv` | 四类细粒度幻觉的数量统计 |
| `statistics/summary.csv` | 模型级指标汇总 |
| `statistics/per_paper.csv` | 各篇论文、各轮测试结果 |
| `main.log` | 整体进度与运行日志 |

四个评估维度与论文保持一致：

| 评估维度 | 测试内容 |
| --- | --- |
| 知识覆盖 | 首次回答的知识覆盖率与宏观阶段完成率 |
| 挑战诱发幻觉 | 误导挑战下的失败率，以及文本与多模态问题的分组表现 |
| 幻觉发生 | 每篇论文的平均幻觉事件数与出现幻觉的对话轮次比例 |
| 错误修复 | 信息遗漏修复、幻觉修复的成功率与平均尝试轮数 |

细粒度分析区分**与论文证据冲突**、**接受错误前提**、**接受过度推断**、**接受错误推理关系**四类幻觉。这些结果共同反映模型的阅读能力及其主要错误类型。

完整测试步骤和验收方式见[指标 1.3 测评方案](docs/evaluation_plan.zh-CN.md)。

## 离线统计已有结果

使用已保存的报告和对话轨迹，即可生成相同的统计表：

```powershell
uv run --locked python scripts/summarize_results.py --result-root eval_result
```

结果保存至 `eval_result/statistics/`。将 `eval_result` 替换为此前的测试批次目录，即可统计自己的结果。此步骤在本地完成，无需模型服务。

## 可选：构建新基准

对已解析的论文依次建图、出题和评测：

```powershell
.\start.ps1 run --paper-id ConVis
```

该流程使用文本/Judge、视觉、Embedding 和被测模型四组服务，请按[配置说明](docs/configuration.md)填写。使用 `run --all --workers 2` 可批量处理已解析论文；默认启动方式仍为前述已有数据集测试。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `data/` | 已发布的论文图、问题、正文与图像证据 |
| `rawPaper/` | 用于构建基准的已解析论文 |
| `papergraph_demo/`、`scripts/` | 建图、评测与统计代码 |
| `eval_result/` | 论文实验的已有评测结果 |
| `outputs/` | 新运行产生的结果和日志 |

统一入口为 `start.ps1`、`start.sh` 和 `run.py`，代码组织见[架构说明](docs/architecture.md)。

## 文档与许可

- [测评方案与验收方式](docs/evaluation_plan.zh-CN.md)
- [API 配置说明](docs/configuration.md)
- [代码架构](docs/architecture.md)

代码采用 [Apache License 2.0](LICENSE) 许可。论文正文、图表及数据集产物仍须遵守原始来源的版权和再分发条款。

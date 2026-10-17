# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench 从科学论文构建图引导的评测基准，并通过多轮阅读任务评测目标模型。标准流水线按需解析 PDF、构建论文图、生成图关联问题，并记录对话轨迹中的知识覆盖、挑战题表现、幻觉及修复结果。

所有标准阶段均自动执行，包括图验证、问题验证、求解器筛选和对话评判，无需人工审核。

测试人员请参阅[纯 AI 建图与评测方案](docs/evaluation_plan.zh-CN.md)，其中包含环境与 API 配置、逐步运行命令、批量与重复实验、指标定义及验收条件。

## 快速启动

使用 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。项目使用 Python 标准库，并通过 `rich` 展示进度。

```powershell
uv sync
uv run python run.py init
```

`init` 将 `config.example.env` 复制为 `config.env`，不会覆盖已有配置。请在 `config.env` 中填写所需阶段的 API 配置。建图需要文本模型、Embedding 模型，以及默认多模态配置下的视觉模型；评测还需要评判模型和被测模型。只有从 PDF 开始解析时才需要 OCR 配置。各阶段的配置要求见[配置说明](docs/configuration.md)。

调用 API 前，先检查配置：

```powershell
uv run python run.py check --stage all
```

此命令仅执行本地预检，不会发起付费 API 请求。只有明确需要测试服务连通性时，才添加 `--probe-apis`。本地检查通过不代表外部 API 一定接受后续请求。

对于已完成 OCR、位于 `rawPaper/ConVis/` 的论文，可直接重新建图并评测：

```powershell
uv run python run.py run --paper-id ConVis
```

`run` 依次执行建图、问题生成和评测。评测默认只使用 `config.env` 中的 `EVAL_TARGET_MODEL`。从新 PDF 开始执行完整流程（含 OCR）：

```powershell
uv run python run.py run --pdf pdfInput/your-paper.pdf
```

以两个论文级工作进程并发处理所有符合条件的论文：

```powershell
uv run python run.py run --all --workers 2
```

如果只需评测 `data/ConVis/` 中已发布的数据包，而不重新构建图和问题：

```powershell
uv run python run.py evaluate --paper-id ConVis
```

添加 `--dry-run` 可查看计划执行的阶段和路径，不实际运行；使用 `--config path/to/config.env` 可指定其他配置文件。例如：

```powershell
uv run python run.py run --paper-id ConVis --dry-run --config path/to/config.env
```

## 输入与输出

| 路径 | 用途 |
| --- | --- |
| `pdfInput/` | 可选的待 OCR 解析 PDF。 |
| `rawPaper/<paper_id>/` | OCR 页面、提取的图片和完成清单。 |
| `data/<paper_id>/` | 论文图、多模态资产、清洗后的正文和图关联问题包。 |
| `outputs/evaluation/<model>/<paper_id>/` | 新生成的对话轨迹、报告和评测检查点。 |
| `outputs/logs/` | 新生成的流水线日志。 |
| `eval_result/` | 论文中的历史评测结果，不存放新运行的结果。 |

原始论文文本及图片仍受其来源许可约束。

已发布数据包使用正斜杠格式的仓库相对路径。主图中包含完整 KC bank，评测已有数据包无需未发布的建图缓存。

## 更多运行方式

统一入口也支持单独执行构建阶段和多模型评测：

```powershell
uv run python run.py build --paper-id ConVis
uv run python run.py questions --paper-id ConVis
uv run python run.py evaluate --paper-id ConVis --models gpt-5-mini gpt-5
uv run python run.py run --all --workers 2 --models gpt-5-mini gpt-5
uv run python run.py check --stage build
```

`check --stage` 支持 `parse`、`build`、`questions`、`evaluate` 和 `all`。执行阶段缺少必要输入或配置时会明确报错，不会静默切换为离线或模拟结果。`papergraph_demo/` 下的底层脚本仍可用于开发和消融实验；端到端运行请使用 `run.py`。

代码目录、阶段模块和产物约定见[架构说明](docs/architecture.md)。

## 验证

```powershell
uv run python -m unittest discover -s tests -v
```

测试使用已发布的 ConVis 数据包、隔离的临时配置和本机 HTTP 服务模拟器，不访问外部模型 API，也不需要真实 API 密钥。

## 许可

代码采用 Apache License 2.0 许可，详见 [LICENSE](LICENSE)。数据集产物和评测输出用于研究复现，仍须遵守原始论文及图片的版权和再分发条款。

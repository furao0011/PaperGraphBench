# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench 从科学论文构建图引导的评测基准，并通过多轮阅读任务评测目标模型。标准流水线按需解析 PDF、构建论文图、生成图关联问题，并记录对话轨迹中的知识覆盖、挑战题表现、幻觉及修复结果。

所有标准阶段均自动执行，包括图验证、问题验证、求解器筛选和对话评判。

测试人员请参阅[建图与评测方案](docs/evaluation_plan.zh-CN.md)，了解测试流程、指标定义及验收条件。

## 运行前必须填写 API 配置

> **必须在根目录 `config.env` 中提供自己的 API Key、服务商 Base URL 和模型 ID。本仓库不提供模型服务或密钥，实际调用会消耗你的服务额度，并可能产生费用。**

完整的“建图 → 出题 → 评测”需要填写下面四组配置：

| 服务用途 | API Key | API 根地址 Base URL | 模型 ID |
| --- | --- | --- | --- |
| 文本模型及评测 Judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` |
| 视觉模型 | `VISION_API_KEY` | `VISION_BASE_URL` | `VISION_MODEL` |
| Embedding 模型 | `EMBED_API_KEY` | `EMBED_BASE_URL` | `EMBED_MODEL` |
| 被测模型 | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` |

这三类信息均需从你的服务商处获取。Base URL 填 API 根地址，不能填网页地址或带 `/chat/completions`、`/embeddings` 的完整接口地址；模型 ID 必须在你的账号下可调用。被测模型需支持图片输入。同一服务的 Key 和 URL 可复用，但每组仍需显式填写。仅评测已有数据包时，默认只需文本/Judge 与被测模型两组。详见[配置说明](docs/configuration.md)。

## 快速启动

安装 [uv](https://docs.astral.sh/uv/)，使用 Python 3.11+，在仓库根目录执行：

```powershell
.\start.ps1
```

macOS/Linux 使用：

```bash
bash ./start.sh
```

脚本自动准备依赖，并在配置不存在时创建 `config.env`。**首次运行后，按上表填写 API 配置，再执行同一命令即可启动。** 已有配置不会被覆盖；配置缺失时会列出待填字段并停止，不会调用模型。

配置完成后，上述一个命令会依次完成 ConVis 的建图、出题和评测。首次完整构建会发起较多付费 API 请求。其他常用启动方式：

```powershell
# 完整测评：建图出题一次、冻结输入、独立评测三轮
.\start.ps1 measure --workers 2

# 批量处理所有已解析论文，各评测一轮
.\start.ps1 --all --workers 2

# 只评测已发布题包，不重新建图出题
.\start.ps1 evaluate --all --workers 2

# 只查看计划，不调用模型
.\start.ps1 --all --dry-run
```

macOS/Linux 将 `.\start.ps1` 替换为 `bash ./start.sh` 即可。完整测评会在 `outputs/measurement/<session>/` 下保存独立的数据包、各轮结果、输入哈希和主日志；`measure --repeats 1` 可改为只评测一轮，`measure --paper-id ConVis` 可只测一篇。

可用 `--config path/to/config.env` 指定其他配置。若 Windows 执行策略拦截脚本，可仅对此次启动使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`。下方 `uv run python run.py ...` 命令保留用于开发及分阶段操作。

## 输入与输出

| 路径 | 用途 |
| --- | --- |
| `pdfInput/` | 可选的待 OCR 解析 PDF。 |
| `rawPaper/<paper_id>/` | OCR 页面、提取的图片和完成清单。 |
| `data/<paper_id>/` | 论文图、多模态资产、清洗后的正文和图关联问题包。 |
| `outputs/evaluation/<model>/<paper_id>/` | 新生成的对话轨迹、报告和评测检查点。 |
| `outputs/logs/` | 新生成的流水线日志。 |
| `outputs/measurement/<session>/` | 独立测评数据包、各轮结果、清单和日志。 |
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

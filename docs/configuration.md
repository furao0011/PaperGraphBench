# API configuration / API 配置

> **Before evaluating a new model, provide your own API key, base URL and model ID for both the judge and target in `config.env`. Model calls use your provider's quota.**
>
> **运行新模型评测前，请在 `config.env` 中填写评判模型和被测模型两组 API 密钥、服务地址与模型 ID。调用费用由所使用的服务计费。**

## Evaluate the released dataset / 评测已有数据集

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11+. Run the launcher from the repository root:

```powershell
# Windows
.\start.ps1
```

```bash
# macOS / Linux
bash ./start.sh
```

The first launch prepares dependencies and creates `config.env`. Fill these six settings, then run the same command again:

首次启动会准备依赖并生成 `config.env`。填写以下六项后，再次执行启动命令即可评测全部 12 篇论文并自动统计结果。

| Role / 角色 | API key | Base URL | Model ID | Capability / 能力 |
| --- | --- | --- | --- | --- |
| Judge / 评判模型 | `API_KEY` | `BASE_URL` | `LLM_MODEL` | OpenAI-compatible chat with JSON output / 支持 JSON 输出 |
| Target / 被测模型 | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` | OpenAI-compatible chat with text and image input / 支持文本和图片输入 |

Use your provider's API root, for example `https://api.example.com/v1`; the client adds the request endpoint. Enter a model ID available to your account. A provider can serve both roles; fill each role's settings explicitly.

Base URL 填写服务商的 API 根地址，程序自动补全请求端点；模型名称填写账号下实际可用的 ID。同一服务可兼任两种角色，两组配置项分别填写。

Keep the remaining options at their defaults for the released-dataset test. Configuration files and credentials are private. Existing environment variables take priority over `config.env`; check those settings if a file edit appears to have no effect. Use `--config path/to/config.env` to select another private file.

评测已有数据集时，其余选项保持默认即可。配置文件与密钥自行保管。环境变量优先于文件设置；如需指定其他配置文件，使用 `--config path/to/config.env`。

## Common commands / 常用命令

```powershell
# Check one paper first / 先测单篇
.\start.ps1 --paper-id ConVis

# Two concurrent papers, three independent rounds / 两篇并发、独立重复三轮
.\start.ps1 --workers 2 --repeats 3

# Compare models available through your configured service / 比较服务下可用的模型
.\start.ps1 --models model-a model-b --workers 2
```

On macOS/Linux, replace `.\start.ps1` with `bash ./start.sh`. Results go to `outputs/testing/<session>/`; start with `statistics/four_dimensions.csv` and `statistics/failure_category_by_model.csv`. See the [evaluation plan / 测评方案](evaluation_plan.zh-CN.md) for result interpretation and acceptance steps.

### Multiple providers / 多服务商配置

Selected models share the target key and URL by default. For a model using another service, add per-model overrides. Convert punctuation in the model ID to underscores and uppercase it for the suffix. For example, `gpt-5-mini` uses:

多个模型默认共用被测模型的密钥与地址。模型来自其他服务时，可按模型 ID 添加专用配置：将标点替换为下划线并转成大写。例如 `gpt-5-mini` 对应：

```dotenv
EVAL_TARGET_API_KEY_GPT_5_MINI=your-provider-key
EVAL_TARGET_BASE_URL_GPT_5_MINI=https://api.example.com/v1
```

The judge remains fixed during model comparisons. / 比较期间固定评判模型。

## Optional: build a new benchmark / 可选：构建新基准

The complete graph → questions → evaluation workflow also uses vision and embedding services. Fill these additional settings in `config.env`:

完整的建图、出题与评测流程还需要视觉和向量服务，补充以下配置：

| Role / 角色 | API key | Base URL | Model ID | Capability / 能力 |
| --- | --- | --- | --- | --- |
| Vision / 视觉模型 | `VISION_API_KEY` | `VISION_BASE_URL` | `VISION_MODEL` | Chat with image input / 支持图片输入 |
| Embedding / 向量模型 | `EMBED_API_KEY` | `EMBED_BASE_URL` | `EMBED_MODEL` | Embeddings API / 向量接口 |

```powershell
.\start.ps1 run --paper-id ConVis
```

This builds the graph and questions from the released parsed paper, then evaluates the target. To start from a new PDF, use `run --pdf path/to/paper.pdf` and configure `PADDLE_OCR_API_URL` and `PADDLE_OCR_TOKEN` for OCR.

该命令从已有解析论文开始建图、出题并评测。处理新的 PDF 时，使用 `run --pdf path/to/paper.pdf`，并配置 OCR 服务的 `PADDLE_OCR_API_URL` 和 `PADDLE_OCR_TOKEN`。

## Summarize saved results / 统计已有结果

```powershell
uv run --locked python scripts/summarize_results.py --result-root eval_result
```

This reads saved evaluation results locally and writes `eval_result/statistics/`. / 此命令在本地读取已有评测结果，输出至 `eval_result/statistics/`，无需配置模型服务。

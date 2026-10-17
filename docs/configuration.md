# Configuration

> **Before running, supply your own API keys, base URLs, and model IDs in the root `config.env`. No API access, credentials, or free model services are included. Running the pipeline consumes your provider quota.**
>
> **运行前必须在根目录 `config.env` 填写自己的 API 密钥、服务地址和模型 ID。本仓库不提供密钥或免费模型服务，调用费用与额度由运行者承担。**

## Required API settings

**For the recommended `start.ps1 test` / `bash start.sh test` workflow, fill only groups 1 and 4 below.** It uses existing datasets, makes no graph/question-generation requests, and exports statistics automatically. Groups 2 and 3 are for rebuilding benchmarks. Each required group's fields must be filled explicitly even when services share a provider, URL or key.

| Group / purpose | API key | Provider API base URL | Exact model ID | Required capability |
| --- | --- | --- | --- | --- |
| 1. Text model: graph construction, questions, evaluation judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` | OpenAI-compatible chat API with JSON responses |
| 2. Vision model: figure/table understanding and multimodal questions | `VISION_API_KEY` | `VISION_BASE_URL` | `VISION_MODEL` | OpenAI-compatible chat API accepting image input |
| 3. Embedding model: graph construction | `EMBED_API_KEY` | `EMBED_BASE_URL` | `EMBED_MODEL` | OpenAI-compatible embeddings API; a chat model is not sufficient |
| 4. Evaluated target model | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` | Chat API accepting text and images for the released multimodal packages |

Use the base URL documented by your provider, such as its API root ending in `/v1` when applicable. Do **not** use a website URL or append `/chat/completions` or `/embeddings`: the client adds the endpoint itself. Model IDs must be callable by your account; selecting a name does not grant access. The judge and evaluated target have independent settings and need not use the same model.

To **evaluate existing packages only**, fill groups **1 and 4**; the default `CLAIM_VERIFY_ENABLED=false` does not require embeddings. Groups 2 and 3 are needed to construct graphs and questions. OCR credentials are only required when starting from PDF, not for the released parsed papers.

## First run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and ensure Python 3.11+ is available. In the repository root, run the appropriate launcher:

```powershell
# Windows PowerShell
.\start.ps1 test --workers 2
```

```bash
# macOS / Linux
bash ./start.sh test --workers 2
```

The launcher prepares dependencies and creates private `config.env` if absent. Fill the Judge and target API key, base URL, and model ID, then repeat the command to evaluate all released packages and write model-level statistics. Missing settings stop execution before model requests. The template contains blank values deliberately.

Common existing-dataset tests (all automatically export statistics):

```powershell
.\start.ps1 test --workers 2
.\start.ps1 test --paper-id ConVis
.\start.ps1 test --repeats 3 --workers 2
```

```bash
bash ./start.sh test --workers 2
bash ./start.sh test --paper-id ConVis
bash ./start.sh test --repeats 3 --workers 2
```

Keep `config.env` private and out of version control. Use `--config path/to/config.env` for a different file; relative paths resolve from the repository root. **Existing process environment variables take priority over the file**, so remove obsolete environment settings if your edits appear to have no effect. The application does not read a legacy `.env` automatically.

`test` defaults to the released `data/` directory, one independent round and all paper packages. `--data-root` explicitly selects another existing dataset. It does not use an old `PAPERGRAPH_DATA_ROOT` setting implicitly. Results and statistics are isolated under `outputs/testing/<session>/`; look first at `statistics/summary.csv`. `--repeats 3` evaluates the same packages three times. No graph or question file is rewritten.

To summarize reports without any API access, run `uv run --locked python scripts/summarize_results.py --result-root outputs/evaluation`. A test-session root, a round's evaluation root, or a historical result root can be used. Without a task manifest, planned completion is unknown rather than assumed to be 100%.

Optional: `measure` rebuilds a package once, then evaluates it three times. This separate construction workflow writes under `outputs/measurement/` and needs all four API groups. It is not required for testing released packages. See the [evaluation plan](evaluation_plan.zh-CN.md) for the recommended existing-dataset procedure.

For manual control, `uv sync`, `uv run python run.py init`, and `uv run python run.py ...` remain available. If the uv environment is active, `python run.py ...` is equivalent.

## Required services by stage

| Stage | Local input | API configuration |
| --- | --- | --- |
| `parse` | A PDF file or directory of PDFs | `PADDLE_OCR_API_URL`, `PADDLE_OCR_TOKEN` |
| `build` | Completed `rawPaper/<paper_id>/` OCR output | `API_KEY`, `BASE_URL`, `LLM_MODEL`; `EMBED_API_KEY`, `EMBED_BASE_URL`, `EMBED_MODEL`; and, for the default multimodal build, `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL` |
| `questions` | `data/<paper_id>/master_graph.json` and associated paper artifacts | Text-model settings plus vision-model settings for default multimodal challenge generation |
| `evaluate` | Released or generated graph, questions, and clean paper text in `data/<paper_id>/` | Text-model settings for the judge; `EVAL_TARGET_API_KEY`, `EVAL_TARGET_BASE_URL`, `EVAL_TARGET_MODEL` for the target model |

The embedding service is required during graph construction. Direct evaluation of an existing package needs it only if `CLAIM_VERIFY_ENABLED` is enabled. A target model used for multimodal questions must accept the attached images. Supplying a credential does not by itself establish that a model supports image input.

`run --paper-id <id>` executes `build`, `questions`, and `evaluate` from existing OCR output. `run --pdf <path>` adds `parse`. `run --all` selects eligible papers and runs their stages in order, with `--workers` controlling paper-level concurrency. Each stage uses only the services it actually needs; in particular, an OCR key is not required to evaluate a published package.

## Local checks and API probes

```powershell
uv run python run.py check --stage all
uv run python run.py check --stage evaluate --config path/to/config.env
uv run python run.py run --paper-id ConVis --dry-run
```

`check` validates the stage configuration; `--dry-run` lists selected papers, stages and output paths without sending model or OCR requests. `check --stage` accepts `parse`, `build`, `questions`, `evaluate`, and `all`. If you add `--probe-apis`, the command contacts configured providers; a provider may bill or rate-limit those probe requests.

## Evaluation models

Without `--models`, evaluation uses the single `EVAL_TARGET_MODEL` in the configuration. To select several models explicitly:

```powershell
uv run python run.py evaluate --paper-id ConVis --models gpt-5-mini gpt-5
uv run python run.py run --all --workers 2 --models gpt-5-mini gpt-5
```

Configure valid target credentials and endpoints for every selected model. A list of model names alone does not provision access. The judge uses `API_KEY`, `BASE_URL`, and `LLM_MODEL` independently of the evaluated target.

Models share `EVAL_TARGET_API_KEY` and `EVAL_TARGET_BASE_URL` unless per-model overrides are supplied. For example, `gpt-5-mini` can use `EVAL_TARGET_API_KEY_GPT_5_MINI` and `EVAL_TARGET_BASE_URL_GPT_5_MINI`. Replace punctuation with underscores and uppercase the model ID for the suffix. The selection order is `--models`, optional `EVAL_BATCH_MODELS`, then `EVAL_TARGET_MODEL`.

## Paths and resumes

OCR output goes to `rawPaper/<paper_id>/`; generated graph and question artifacts go to `data/<paper_id>/`. New evaluation results are isolated by model and paper under `outputs/evaluation/<model>/<paper_id>/`, and logs go to `outputs/logs/`. The tracked `eval_result/` directory holds historical results and is not the new output root.

Keep each paper's OCR manifest, graph, and question package together under the same `paper_id`. The unified runner should surface missing prerequisites as errors; do not use `ALLOW_OFFLINE_FALLBACK` or `ALLOW_MOCK_EVAL` for real benchmark runs.

The runner records input/configuration signatures under each output's `cache/pipeline/` directory. Unchanged completed stages are skipped; an interrupted stage resumes only when its signature is unchanged. Changed inputs or model settings restart that stage instead of reusing incompatible checkpoints. Outputs made before the unified runner have no signature and are regenerated when their stage is selected. Use `evaluate` to consume a released graph/question package without rebuilding it. `--force` explicitly reruns every selected stage.

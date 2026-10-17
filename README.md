# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench constructs a graph-guided benchmark from scientific papers and evaluates a target model through multi-turn reading tasks. The standard pipeline parses a PDF when needed, builds a paper graph, generates graph-linked questions, and records dialogue-level coverage, challenge, hallucination, and repair results.

All standard stages run automatically, including graph verification, question validation, solver filtering, and dialogue judging.

For test operators, see the [Evaluation plan (Chinese)](docs/evaluation_plan.zh-CN.md) for the test procedure, metric definitions, and acceptance criteria.

## Required before running: your API configuration

> **You must provide your own API keys, provider base URLs, and model IDs in the root `config.env`. No model services or credentials are bundled. Calls consume your provider quota and may incur charges.**

The full graph → questions → evaluation workflow needs all four groups:

| Service | API key | API base URL | Model ID |
| --- | --- | --- | --- |
| Text model and evaluation judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` |
| Vision model | `VISION_API_KEY` | `VISION_BASE_URL` | `VISION_MODEL` |
| Embedding model | `EMBED_API_KEY` | `EMBED_BASE_URL` | `EMBED_MODEL` |
| Evaluated target model | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` |

Obtain the key, API root URL, and callable model ID from your provider. Do not use a website address or a full `/chat/completions` or `/embeddings` endpoint. The target must support image input for multimodal evaluation. If groups share a service, explicitly fill each group even when the key and URL are the same. Evaluation of existing packages needs only the text/judge and target groups by default. See [Configuration](docs/configuration.md) for details.

## Quick start

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11+. From the repository root, run:

```powershell
.\start.ps1
```

On macOS/Linux:

```bash
bash ./start.sh
```

The launcher prepares dependencies and creates `config.env` if absent. **On the first run, fill the API settings listed above, then run the same command again.** Existing configuration is never overwritten. Missing configuration stops execution before model requests and lists the fields to fill.

Once configured, that single command builds the ConVis graph, generates questions, and evaluates the configured target model. The first full build can make many paid API requests. For other runs:

```powershell
# Full measurement: build once, freeze inputs, evaluate three independent rounds
.\start.ps1 measure --workers 2

# Process all parsed papers once
.\start.ps1 --all --workers 2

# Evaluate released packages without rebuilding graphs or questions
.\start.ps1 evaluate --all --workers 2

# Preview the plan without model requests
.\start.ps1 --all --dry-run
```

On macOS/Linux, replace `.\start.ps1` with `bash ./start.sh`. Measurement sessions are isolated under `outputs/measurement/<session>/`, including the dataset, round results, input hashes, and main log. Use `measure --repeats 1` for one evaluation round or `measure --paper-id ConVis` for one paper.

Use `--config path/to/config.env` to select a different configuration. If Windows blocks script execution, use `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1` for that launch only. Individual `uv run python run.py ...` commands remain available below for development and stage-level control.

## Inputs and outputs

| Path | Purpose |
| --- | --- |
| `pdfInput/` | Optional input PDFs for OCR. |
| `rawPaper/<paper_id>/` | OCR pages, extracted images, and completion manifest. |
| `data/<paper_id>/` | Paper graph, multimodal assets, clean text, and graph-linked question package. |
| `outputs/evaluation/<model>/<paper_id>/` | New dialogue trajectories, reports, and evaluation checkpoints. |
| `outputs/logs/` | New pipeline logs. |
| `outputs/measurement/<session>/` | Isolated measurement dataset, round results, manifests, and logs. |
| `eval_result/` | Historical paper evaluation results; not the destination for new runs. |

Original paper text and figures remain subject to their source licenses.

Released packages use repository-relative paths with forward slashes. Their master graphs contain the full KC bank, so evaluating a released package does not require unpublished graph-building caches.

## More ways to run

The unified entry point also supports individual construction stages and multi-model evaluation:

```powershell
uv run python run.py build --paper-id ConVis
uv run python run.py questions --paper-id ConVis
uv run python run.py evaluate --paper-id ConVis --models gpt-5-mini gpt-5
uv run python run.py run --all --workers 2 --models gpt-5-mini gpt-5
uv run python run.py check --stage build
```

`check --stage` accepts `parse`, `build`, `questions`, `evaluate`, or `all`. A requested stage fails when a required input or credential is missing; it does not silently switch to an offline or mock result. The lower-level scripts in `papergraph_demo/` remain available for development and ablations, while `run.py` is the supported end-to-end interface.

For the source tree, stage modules, and artifact contracts, see [Architecture](docs/architecture.md).

## Verification

```powershell
uv run python -m unittest discover -s tests -v
```

Tests use the released ConVis package, isolated temporary configurations, and a local HTTP provider simulator. They do not contact external model APIs or need real API keys.

## License

The code is released under the Apache License 2.0 license; see [LICENSE](LICENSE). Dataset artifacts and evaluation outputs are for research reproduction, subject to the copyright and redistribution terms of the original papers and figures.

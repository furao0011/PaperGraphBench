# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench constructs a graph-guided benchmark from scientific papers and evaluates a target model through multi-turn reading tasks. The standard pipeline parses a PDF when needed, builds a paper graph, generates graph-linked questions, and records dialogue-level coverage, challenge, hallucination, and repair results.

All standard stages run automatically, including graph verification, question validation, solver filtering, and dialogue judging.

For test operators, see the [Pure-AI evaluation plan (Chinese)](docs/evaluation_plan.zh-CN.md) for environment and API setup, step-by-step commands, batch and repeat runs, metric definitions, and acceptance criteria.

## Quick start

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). The project uses the Python standard library plus `rich` for progress display.

```powershell
uv sync
uv run python run.py init
```

`init` copies `config.example.env` to `config.env` without overwriting an existing configuration. Edit `config.env` with credentials for the stages you intend to run. Graph construction requires a text model, an embedding model, and—under the default multimodal configuration—a vision model. Evaluation also requires a judge model and the target model. OCR credentials are needed only when parsing a PDF. See [Configuration](docs/configuration.md) for the stage-by-stage requirements.

Check configuration before making API calls:

```powershell
uv run python run.py check --stage all
```

This is a local preflight, not a paid API test. Add `--probe-apis` only when you explicitly want to test provider connectivity. A successful local check does not guarantee that an external API will accept a later request.

To rebuild and evaluate an OCR-parsed paper already under `rawPaper/ConVis/`:

```powershell
uv run python run.py run --paper-id ConVis
```

`run` executes graph construction, question generation, then evaluation. By default, evaluation uses only `EVAL_TARGET_MODEL` from `config.env`. To start from a new PDF, including OCR:

```powershell
uv run python run.py run --pdf pdfInput/your-paper.pdf
```

To process all eligible papers with two paper-level workers:

```powershell
uv run python run.py run --all --workers 2
```

If you only want to evaluate an existing released package in `data/ConVis/`, without rebuilding its graph or questions:

```powershell
uv run python run.py evaluate --paper-id ConVis
```

Use `--dry-run` to see the planned stages and paths without executing them. Use `--config path/to/config.env` to select another configuration file. For example:

```powershell
uv run python run.py run --paper-id ConVis --dry-run --config path/to/config.env
```

## Inputs and outputs

| Path | Purpose |
| --- | --- |
| `pdfInput/` | Optional input PDFs for OCR. |
| `rawPaper/<paper_id>/` | OCR pages, extracted images, and completion manifest. |
| `data/<paper_id>/` | Paper graph, multimodal assets, clean text, and graph-linked question package. |
| `outputs/evaluation/<model>/<paper_id>/` | New dialogue trajectories, reports, and evaluation checkpoints. |
| `outputs/logs/` | New pipeline logs. |
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

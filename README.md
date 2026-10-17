# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench evaluates scientific-paper understanding through graph-guided, multi-turn reading tasks. A paper graph connects the paper's argument structure, evidence-grounded knowledge components, reasoning relations, and figures or tables. It guides questions about coverage, misleading claims, and reasoning, while the evaluation follows hallucinations and their repair throughout the dialogue.

This process reveals whether a model can understand the paper as a whole, stay grounded in textual and visual evidence, and correct mistakes during continued reading.

## Quick start: test a model on the released dataset

The repository includes **12 paper packages** in `data/`. The default launcher evaluates your model on these packages and automatically generates statistics.

### 1. Prepare the environment and API settings

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11+. Clone the complete repository, open its root directory, and run:

```powershell
.\start.ps1
```

On macOS/Linux:

```bash
bash ./start.sh
```

The launcher prepares dependencies. On the first launch, it creates `config.env` and prompts you to fill in the required settings.

> **Before testing, provide your own API key, API base URL, and callable model ID for both services below in `config.env`. Then run the same command again. API calls use your provider's quota and may incur charges.**

| Service | API key | API base URL | Model ID |
| --- | --- | --- | --- |
| Evaluation judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` |
| Target model | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` |

Both services use OpenAI-compatible APIs. The judge needs structured JSON output; the target model needs image input. Fill in the provider's API root URL, such as `https://your-provider.example/v1`. These two services cover the default existing-dataset test. See [Configuration](docs/configuration.md) for further options.

### 2. Run the evaluation

After configuration, `.\start.ps1` evaluates all 12 papers once and produces the result tables. The following options support a trial run, repeated testing, and model comparison:

```powershell
# Try one paper
.\start.ps1 test --paper-id ConVis

# Evaluate all papers three times, with two parallel tasks
.\start.ps1 test --workers 2 --repeats 3

# Compare models available through your configured target service
.\start.ps1 test --models model-a model-b --workers 2
```

On macOS/Linux, replace `.\start.ps1` with `bash ./start.sh`. Use `--dry-run` to preview the test plan, `--data-root` to select another dataset, or `--config` to select another configuration file. For a Windows execution-policy restriction, use `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`.

### 3. Read the results

Each run prints its output directory: `outputs/testing/<session>/`. Open the following files:

| File | Contents |
| --- | --- |
| `statistics/four_dimensions.csv` | Results for the four evaluation dimensions |
| `statistics/failure_category_by_model.csv` | Counts of the four fine-grained hallucination categories |
| `statistics/summary.csv` | Model-level metric summary |
| `statistics/per_paper.csv` | Per-paper and per-round results |
| `main.log` | Overall progress and execution log |

The four evaluation dimensions follow the paper:

| Dimension | What is measured |
| --- | --- |
| Knowledge coverage | Knowledge coverage and macro-stage completion in initial answers |
| Challenge-induced hallucination | Failure under misleading challenges, overall and for text/multimodal questions |
| Hallucination occurrence | Average hallucination events per paper and the proportion of turns containing hallucinations |
| Error repair | Success rate and average attempts for omission repair and hallucination repair |

Fine-grained hallucination analysis distinguishes **contradiction with paper evidence**, **acceptance of false premises**, **acceptance of overclaims**, and **acceptance of incorrect reasoning relations**. Together, these results describe both the model's reading ability and the kinds of errors it makes.

For the testing procedure and acceptance criteria, see the [Indicator 1.3 evaluation plan (Chinese)](docs/evaluation_plan.zh-CN.md).

## Summarize existing results offline

Use saved reports and dialogue trajectories to generate the same result tables:

```powershell
uv run --locked python scripts/summarize_results.py --result-root eval_result
```

Tables are saved to `eval_result/statistics/`. Replace `eval_result` with a previous test-session directory to summarize your own results. This step runs locally and requires no model service.

## Optional: construct a new benchmark

To build a graph, generate questions, and evaluate a parsed paper:

```powershell
.\start.ps1 run --paper-id ConVis
```

This workflow uses text/judge, vision, embedding, and target-model services. Fill in all four service groups described in [Configuration](docs/configuration.md). Use `run --all --workers 2` to process all parsed papers. The default launcher remains the existing-dataset test described above.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/` | Released paper graphs, questions, text, and visual evidence |
| `rawPaper/` | Parsed source papers for benchmark construction |
| `papergraph_demo/`, `scripts/` | Construction, evaluation, and statistics code |
| `eval_result/` | Saved results from the paper's experiments |
| `outputs/` | Results and logs from new runs |

The supported entry points are `start.ps1`, `start.sh`, and `run.py`. See [Architecture](docs/architecture.md) for the code structure.

## Documentation and license

- [Evaluation plan and acceptance criteria](docs/evaluation_plan.zh-CN.md)
- [API configuration](docs/configuration.md)
- [Code architecture](docs/architecture.md)

Code is licensed under [Apache License 2.0](LICENSE). Paper text, figures, and dataset artifacts remain subject to the original sources' copyright and redistribution terms.

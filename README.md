<div align="center">

<h1>PaperGraph-Bench</h1>
<h3>Graph-Guided Benchmark Construction and Multi-Turn Evaluation<br>for Scientific Paper Reading</h3>

<p><strong>Official code and benchmark · Findings of EMNLP 2026</strong></p>

<p>
  <a href="https://aclanthology.org/people/rao-fu/">Rao&nbsp;Fu</a> &nbsp;·&nbsp;
  <a href="https://aclanthology.org/people/qing-dai/">Qing&nbsp;Dai</a> &nbsp;·&nbsp;
  <a href="https://aclanthology.org/people/kaitai-ou-yang/">Kaitai&nbsp;Ou-yang</a> &nbsp;·&nbsp;
  <a href="https://aclanthology.org/people/kang-he-wuhan/">Kang&nbsp;He</a> &nbsp;·&nbsp;
  <a href="https://aclanthology.org/people/fei-li/">Fei&nbsp;Li</a> &nbsp;·&nbsp;
  <a href="https://aclanthology.org/people/donghong-ji/">Donghong&nbsp;Ji</a> &nbsp;·&nbsp;
  <a href="https://aclanthology.org/people/chong-teng/">Chong&nbsp;Teng</a>
</p>

<p>Wuhan University</p>

<p>
  <a href="https://2026.emnlp.org/program/find_papers/"><img src="https://img.shields.io/badge/EMNLP_2026-Findings-6366f1?style=flat-square" alt="EMNLP 2026 Findings"></a>
  <a href="https://aclanthology.org/2026.findings-emnlp.804/"><img src="https://img.shields.io/badge/ACL_Anthology-2026.findings--emnlp.804-b31b1b?style=flat-square" alt="ACL Anthology paper"></a>
  <a href="https://aclanthology.org/2026.findings-emnlp.804.pdf"><img src="https://img.shields.io/badge/Paper-PDF-ea580c?style=flat-square" alt="Paper PDF"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-4d7c0f?style=flat-square" alt="Apache License 2.0"></a>
</p>
<p>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11 or newer"></a>
  <a href="data/"><img src="https://img.shields.io/badge/Dataset-12_papers-0f766e?style=flat-square" alt="Dataset with 12 papers"></a>
  <a href="eval_result/"><img src="https://img.shields.io/badge/Experiments-10_models-475569?style=flat-square" alt="Results for 10 models"></a>
</p>

<p>
  <a href="https://aclanthology.org/2026.findings-emnlp.804/">Paper</a> &nbsp;·&nbsp;
  <a href="https://aclanthology.org/2026.findings-emnlp.804.pdf">PDF</a> &nbsp;·&nbsp;
  <a href="https://2026.emnlp.org/program/find_papers/">Conference</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>
<p>
  <a href="data/">Dataset</a> &nbsp;·&nbsp;
  <a href="eval_result/">Results</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="docs/configuration.md">API Configuration</a> &nbsp;·&nbsp;
  <a href="docs/evaluation_plan.zh-CN.md">Evaluation Plan</a> &nbsp;·&nbsp;
  <a href="https://github.com/furao0011/PaperGraphBench/issues">Issues</a>
</p>

<p><a href="README.md"><strong>English</strong></a> &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a></p>

</div>

PaperGraph-Bench evaluates scientific-paper understanding through graph-guided, multi-turn reading tasks. A paper graph connects the paper's argument structure, evidence-grounded knowledge components, reasoning relations, and figures or tables. It guides questions about coverage, misleading claims, and reasoning, while the evaluation follows hallucinations and their repair throughout the dialogue.

This process reveals whether a model can understand the paper as a whole, stay grounded in textual and visual evidence, and correct mistakes during continued reading.

## News

- **2026-10:** PaperGraph-Bench was published in **Findings of EMNLP 2026**. The [proceedings paper](https://aclanthology.org/2026.findings-emnlp.804/) and [official acceptance listing](https://2026.emnlp.org/program/find_papers/) are available.

<a name="quick-start"></a>

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

<a name="citation"></a>

## Citation

If you use PaperGraph-Bench in your research, please cite the proceedings paper:

```bibtex
@inproceedings{fu-etal-2026-papergraph,
    title = "{P}aper{G}raph-Bench: Graph-Guided Benchmark Construction and Multi-Turn Evaluation for Scientific Paper Reading",
    author = "Fu, Rao  and
      Dai, Qing  and
      Ou-yang, Kaitai  and
      He, Kang  and
      Li, Fei  and
      Ji, Donghong  and
      Teng, Chong",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.804/",
    doi = "10.18653/2026.findings-emnlp.804",
    pages = "17028--17055"
}
```

## Questions and feedback

For usage questions or reproducibility issues, please open a [GitHub issue](https://github.com/furao0011/PaperGraphBench/issues) with the command, model IDs, and relevant logs. Remove API keys and other credentials before sharing logs.

# autotrain

Give it a domain. Get back a fine-tuned Qwen model trained on that website's content.

```bash
autotrain https://example.com
```

That is the entire required input. The pipeline crawls the site, extracts and cleans the
content, synthesises an instruction dataset, fine-tunes Qwen with LoRA/QLoRA, evaluates
the result, exports it, benchmarks it against the base model, and writes a report — with
no further interaction.

No RAG, no vector database, no embeddings, no retrieval at inference time. The knowledge
ends up in the weights.

---

## Contents

- [Install](#install)
- [Usage](#usage)
- [The twelve phases](#the-twelve-phases)
- [Output layout](#output-layout)
- [Configuration](#configuration)
- [Dataset generation backends](#dataset-generation-backends)
- [Hardware and the QLoRA path](#hardware-and-the-qlora-path)
- [Resume, caching, and incremental updates](#resume-caching-and-incremental-updates)
- [Docker](#docker)
- [Testing](#testing)
- [Design notes](#design-notes)

---

## Install

```bash
git clone <repo> autotrain && cd autotrain
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -e .                 # phases 1-7: crawl -> clean -> dataset -> split
pip install -e ".[train]"        # phase 8+:   fine-tuning, evaluation, export
pip install -e ".[cuda]"         # 4-bit QLoRA (Linux + CUDA only)
pip install -e ".[dev]"          # tests
```

The base install is deliberately light — no torch — so the data pipeline runs on any
machine. The training extras are only needed once you reach phase 8.

Requires Python 3.10+.

## Usage

```bash
# The whole thing
autotrain https://example.com

# Also works without installing
python main.py https://example.com

# Crawl a subtree only
autotrain https://example.com \
  --set 'crawler.include_patterns=^https://example\.com/docs/'

# Build the dataset now, train later on a GPU box
autotrain https://example.com --skip-training

# Re-run one phase
autotrain https://example.com --phases generate,validate,split

# See what it would do
autotrain https://example.com --dry-run

# Inspect progress and results
autotrain status https://example.com
autotrain inspect https://example.com -n 10

# Force a phase to rebuild
autotrain clean https://example.com --phase generate
```

### Common options

| Flag | Purpose |
|---|---|
| `--max-pages N` | Cap the crawl |
| `--config FILE` | Layer a YAML/TOML file over the defaults |
| `--set KEY=VALUE` | Override any single setting (repeatable) |
| `--phases A,B,C` | Run a subset |
| `--skip-training` | Phases 1-7 only |
| `--force` | Re-run even when outputs exist |
| `--seed N` | Make the run reproducible |
| `--workspace DIR` | Where runs are stored (default `runs/`) |
| `--log-level DEBUG` | Verbose logging |

## The twelve phases

| # | Phase | What it does | Key outputs |
|---:|---|---|---|
| 1 | Discovery | robots.txt, sitemap.xml, link graph, URL normalisation, trap detection | `urls.json`, `sitemap.json`, `crawl_report.json` |
| 2 | HTML collection | Raw HTML plus status, headers, redirect chain, timestamp | `dataset/raw_html/page_NNNN.html` |
| 3 | Extraction | Strips nav/footer/sidebar/cookie banners/ads/hidden nodes; keeps headings, prose, lists, tables, code | `clean_docs/page_NNNN.md` |
| 4 | Cleaning | Exact + near-duplicate removal, corpus-wide boilerplate stripping, menu residue removal | `clean_dataset/corpus.jsonl` |
| 5 | Dataset generation | Eight instruction types across windowed documents | `generated/dataset.jsonl` |
| 6 | Validation | Schema, encoding, length, duplicate, refusal and truncation checks | `validated/dataset.jsonl`, `validation_report.json` |
| 7 | Splitting | 90/10, grouped by source document | `splits/train.jsonl`, `splits/validation.jsonl` |
| 8 | Fine-tuning | QLoRA on CUDA, LoRA elsewhere; checkpointing and resume | `training/adapter/`, `training/metrics.json` |
| 9 | Evaluation | Loss curves, perplexity, sample generations, instruction-following, grounding, consistency | `evaluation/` |
| 10 | Export | Adapter, merged HF checkpoint, GGUF, Ollama Modelfile | `exports/` |
| 11 | Benchmark | Held-out questions against base vs fine-tuned | `benchmark/benchmark_report.md` |
| 12 | Report | Everything above, with ASCII loss curves | `reports/final_report.md` |

Phases 1 and 2 run as a single pass — you have to fetch a page to read its links, so
crawling it twice would be wasted work.

### The eight instruction types

`doc_qa` · `summarization` · `explanation` · `step_by_step` · `api_understanding` ·
`code_explanation` · `concept_learning` · `troubleshooting`

Each is gated on a structural test of the source document. A page with no code produces
no `code_explanation` samples, because the alternative is a fabricated answer — and
fabricated training data teaches fabrication.

## Output layout

```
runs/example.com/
├── run.json                  phase state (drives resume)
├── logs/                     one JSONL log per phase
├── discovery/                urls.json · sitemap.json · crawl_report.json
├── dataset/raw_html/         page_0001.html … + index.jsonl
├── clean_docs/               per-page Markdown
├── clean_dataset/            deduplicated corpus + corpus.jsonl
├── generated/                dataset.jsonl + generation_state.json
├── validated/                dataset.jsonl + validation_report.json
├── splits/                   train.jsonl · validation.jsonl
├── training/                 checkpoints/ · adapter/ · metrics.json
├── evaluation/               evaluation_report.json · sample_generations.json
├── exports/                  adapter/ · merged/ · gguf/ · ollama/
├── benchmark/                benchmark_report.md
└── reports/                  final_report.md · final_report.json
```

## Configuration

Defaults live in `configs/default.yaml`; every value is documented there. Resolution
order, later winning:

1. Field defaults in `src/autotrain/utils/config.py`
2. `configs/default.yaml`
3. `--config myfile.yaml`
4. `--set key.path=value`
5. The positional domain

```yaml
crawler:
  max_pages: 10000
  concurrency: 8
  respect_robots: true
  include_subdomains: false
  include_patterns: ['^https://example\.com/docs/']

dataset:
  provider:
    backend: auto          # ollama -> openai -> anthropic -> structural
  pairs_per_type: 2

training:
  base_model: auto         # 7B on CUDA, 0.5B on CPU
  load_in_4bit: true
  lora_r: 16
  num_train_epochs: 3.0
```

In containers, `AUTOTRAIN_SET="a.b=1;c.d=2"` applies overrides without changing the command.

## Dataset generation backends

Phase 5 needs something to turn documents into instruction pairs. Four options, tried in
order under `backend: auto`:

| Backend | Needs | Quality | Notes |
|---|---|---|---|
| `ollama` | Ollama running locally | High | Free, private, no API key. Default target `qwen2.5:7b` |
| `openai` | `OPENAI_API_KEY` | High | Any OpenAI-compatible endpoint — vLLM, LM Studio, together.ai |
| `anthropic` | `ANTHROPIC_API_KEY` | Highest | Best pairs, costs per page |
| `structural` | nothing | Lower | Derives pairs from document structure. Offline, free, deterministic |

The structural generator is the reason the pipeline never hard-blocks. It builds
questions from headings, answers from section bodies, step lists from ordered lists, and
code tasks from fenced blocks. Every answer is verbatim source text, so it cannot
hallucinate — but it recombines rather than reasons, so the samples are flatter than an
LLM's. Start Ollama if you want the good version:

```bash
ollama pull qwen2.5:7b
autotrain https://example.com
```

## Hardware and the QLoRA path

The trainer detects what the host can do and says what it chose.

| Host | Method | Model default | Precision | Optimiser |
|---|---|---|---|---|
| CUDA + bitsandbytes | 4-bit NF4 QLoRA, double quant | Qwen2.5-7B-Instruct | bf16 (fp16 pre-Ampere) | `paged_adamw_8bit` |
| CUDA, no bitsandbytes | LoRA | Qwen2.5-7B-Instruct | bf16/fp16 | `adamw_torch` |
| Apple MPS | LoRA | Qwen2.5-0.5B-Instruct | fp16 | `adamw_torch` |
| CPU | LoRA | Qwen2.5-0.5B-Instruct | fp32 | `adamw_torch` |

`bitsandbytes` has no CPU or Metal 4-bit kernel, so true QLoRA requires CUDA. Rather than
crash, the trainer disables quantisation and logs the downgrade; every downgrade also
appears in the final report. Flash-attention-2 is used when installed and falls back to
SDPA, then eager.

Setting `training.device=cuda` explicitly on a machine with no GPU raises instead of
silently degrading — an explicit request deserves an explicit failure.

## Resume, caching, and incremental updates

Interrupt any run and re-issue the same command. `run.json` records each phase's status;
a phase is skipped only if it completed **and** its declared outputs still exist.
Deleting an artifact re-arms that phase.

- **Crawl cache** — `raw_html/index.jsonl` is replayed at startup. Already-fetched pages
  keep their page numbers and are not re-requested. A second run over an unchanged site
  costs seconds. Set `runtime.cache_ttl_hours` to expire the cache.
- **Incremental re-crawl** — cached pages are still mined for links, so pages added since
  the last run are discovered and fetched.
- **Generation** — `generation_state.json` tracks completed (document, window, type)
  triples. An interrupted 6-hour generation resumes exactly where it stopped. If the
  corpus itself changed, generation restarts rather than mixing stale pairs with fresh.
- **Training** — checkpoints every `save_steps`; the newest is picked up automatically.

## Docker

```bash
# Full pipeline on a GPU host
docker build -t autotrain:cuda .
docker run --gpus all -v "$PWD/runs:/work/runs" autotrain:cuda https://example.com

# Data phases only — small image, no GPU, no torch
docker build --target base -t autotrain:data .
docker run -v "$PWD/runs:/work/runs" autotrain:data https://example.com --skip-training

# With compose, including an optional Ollama sidecar for phase 5
docker compose run --rm autotrain https://example.com
docker compose --profile ollama up -d
docker compose exec ollama ollama pull qwen2.5:7b
```

## Testing

```bash
pytest                      # unit + integration
pytest -q tests/test_urlnorm.py
```

The crawler integration tests use `respx` to serve a synthetic site in-process, so they
cover link discovery, scope enforcement, robots.txt obedience and duplicate-body
collapsing without touching the network.

## Design notes

A few decisions that are not obvious from the file layout.

**Grouped train/validation split.** Phase 5 produces several samples per source page, and
they share vocabulary and whole sentences. A random split puts near-copies of training
samples into validation, so validation loss measures memorisation and falls smoothly
while telling you nothing. Splitting by source document — every sample from a page lands
on one side — makes the number mean something. The split hashes the group key rather than
shuffling, so adding pages later does not reshuffle existing assignments.

**Applicability gating.** Every (document, sample type) pair is checked against the
document's structure before a generation call. Generating a troubleshooting sample from a
page describing no failures yields a confident invention. Skipping it yields nothing.
Nothing is better, and it is also cheaper.

**Completion-only loss.** Loss is masked over the prompt so the model is graded only on
the assistant turn. Without this, a corpus sharing one system prompt spends real capacity
learning to reproduce that prompt.

**Conservative cleaning.** Boilerplate removal never touches fenced code, never removes a
line longer than 200 characters, and never removes Markdown structure. Over-cleaning
deletes training signal and — unlike under-cleaning — leaves no trace in the output.

**Grounding as a hallucination proxy.** Evaluation scores what fraction of an answer's
content words appear anywhere in the crawled corpus. It cannot catch a confident wrong
number that happens to appear elsewhere in the corpus, but it reliably catches wholesale
invention, which is the failure that matters on a closed-domain fine-tune.

**Benchmarks on held-out questions.** Phase 11 draws from the validation split, never the
training set. Benchmarking on training questions measures recall and always shows a large
fake improvement.

## Licence

MIT.

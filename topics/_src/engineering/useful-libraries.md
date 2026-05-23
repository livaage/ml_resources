---
title: Useful Libraries — ML Resources Hub
eyebrow_text: ← Engineering · Time-Savers
eyebrow_href: {{root}}engineering.html
heading: Useful Libraries
lead: Small libraries that punch above their weight — keep them on your radar so you don't reinvent them.
active_nav: engineering
prev_href: cli-patterns.html
prev_title: CLI Patterns
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Most "I just need to..." problems have a library.** Knowing the right small library beats writing 200 lines of bespoke code. This page is a curated list — not exhaustive, but the ones that come up in nearly every ML project.

</div>

<article class="tldr-body" markdown="1">

The set below isn't a tier list — every library here is worth a 30-minute investment. If you haven't used 4 of them, that's 4 places where you've probably been writing more code than you needed to.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Data & configs

- **polars**: faster pandas. Eager + lazy API. The new default for tabular work above ~1M rows.
- **pydantic**: typed dataclasses with validation. Standard for configs and API schemas.
- **hydra**: composable YAML configs with CLI overrides.
- **jsonargparse**: argparse + dataclasses + YAML. Less heavy than Hydra.
- **great_expectations**: declarative data validation.

</div>

<div class="no" markdown="1">

### Modelling & training

- **lightning**: training-loop scaffolding
- **accelerate**: distributed + mixed precision for your custom loop
- **einops**: tensor reshaping with named axes — readable in months
- **timm**: every vision backbone, pretrained
- **transformers**: every NLP / multimodal model, pretrained
- **peft**: LoRA, adapters, prompt tuning

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# einops — readable tensor reshaping
from einops import rearrange, repeat, reduce, einsum

# Batch × Heads × Seq × Dim → Batch × Seq × (Heads · Dim)
out = rearrange(x, "b h n d -> b n (h d)")

# Tile a vector across the batch
batched = repeat(v, "d -> b d", b=32)

# Reduce over the spatial axes
pooled = reduce(features, "b c h w -> b c", "mean")

# Explicit einsum
attn = einsum(q, k, "b h n d, b h m d -> b h n m")
```

</div>

<div class="level-next">
<span>Want experiment tooling, dev quality of life, & the production stack?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The "always reach for" set</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{uv}, \text{ruff}, \text{pre-commit}, \text{loguru}, \text{rich}, \text{pytest} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

These six are the "every project, no excuse" baseline

</li>
<li markdown="1">

Together: dependency mgmt, lint, hooks, logging, pretty output, tests

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{uv} + \text{ruff} + \text{pre-commit} + \text{loguru} + \text{rich} + \text{pytest} \;=\; \text{every-project baseline} $$</span>

**In words.** Not a real equation — just a checklist. These six tools cover the six things every Python project needs: `uv` for dependencies, `ruff` for linting and formatting, `pre-commit` for git hooks that run them automatically, `loguru` for logging, `rich` for pretty terminal output, and `pytest` for tests. Install all six at project setup and you've eliminated a dozen ad-hoc decisions.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

These six are the "every project, no excuse" baseline

</li>
<li markdown="1">

Together: dependency mgmt, lint, hooks, logging, pretty output, tests

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Experiment tracking.** `wandb`, `mlflow`, `aim`, `comet`, `neptune`. Pick one and stick with it. See [Experiment Tracking](experiment-tracking.html).

**Dev quality of life.** `uv` (fast Python package manager, replaces pip/poetry/venv). `ruff` (linter + formatter; replaces black + flake8 + isort + many more). `pre-commit` (run hooks on git commit). `rich` (pretty terminal output, progress bars). `loguru` (logging that just works). `pytest` (the testing framework).

**Tabular & data.** `polars` (faster pandas). `duckdb` (SQL on parquet, blazingly fast for analytics). `pyarrow` (the columnar foundation). `fsspec` (uniform interface to local/S3/GCS).

**Modelling.** `scikit-learn` (still the right answer for tabular ML). `xgboost` / `lightgbm` / `catboost` (gradient boosting). `statsmodels` (classical statistics). `scipy` (everything else).

**NLP.** `sentence-transformers` for embedding. `tiktoken` for OpenAI-compatible tokenisation. `spacy` for classical NLP. `llama-cpp-python` for local LLMs.

**Inference / serving.** `onnxruntime`, `vllm`, `tgi`, `bentoml`, `fastapi`. See [Serving](serving.html).

**Visualisation.** `matplotlib` + `seaborn` (the classics). `plotly` (interactive). `altair` (declarative). `holoviews` for time series.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# uv — fast dependency management. Replaces pip + venv + pip-tools.
# pip install uv
# uv venv && source .venv/bin/activate
# uv pip install torch transformers wandb hydra-core polars
# uv pip compile pyproject.toml -o requirements.lock      # lockfile

# ruff — lint + format. Replaces black + flake8 + isort.
# pip install ruff
# ruff check .              # lint
# ruff format .             # format
# Add to pre-commit:
# - repo: https://github.com/astral-sh/ruff-pre-commit
#   rev: v0.4.0
#   hooks:
#     - id: ruff
#     - id: ruff-format

# rich — pretty terminal output
from rich.progress import track
from rich.console import Console
console = Console()
for item in track(items, description="Training..."):
    process(item)
console.print(f"[bold green]Done![/]")
```

</div>

<div class="level-next">
<span>Want the niche-but-saves-a-day libraries?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">"I wish I'd known about that"</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{specialised library} \;\gg\; \text{your hand-rolled version} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Niche libraries usually beat hand-rolled code by 5–50×

</li>
<li markdown="1">

Maintainers have already hit the edge cases you'll hit later

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{specialised library} \;\;\gg\;\; \text{your hand-rolled version} $$</span>

**In words.** The double-greater `≫` is mathematician shorthand for "much, much greater than". The point is a research-engineering principle: when a focused library exists for the niche problem you're solving, it's almost always faster, more correct, and better-tested than a quick implementation you'd write yourself. The maintainer has already hit the edge cases you haven't even imagined. Pay the 30-minute cost of learning the API and pocket the 5–50× speed-up.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

Niche libraries usually beat hand-rolled code by 5–50×

</li>
<li markdown="1">

Maintainers have already hit the edge cases you'll hit later

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Optimisation.** `optuna` for hyperparameter search. `scipy.optimize` for classical optimisation. `pyomo` / `cvxpy` for math programming.

**Geometry & signal.** `scikit-image` for image processing. `librosa` for audio. `shapely` + `geopandas` for geospatial. `biopython` for bio.

**Profiling.** `py-spy` (sampling profiler, attach to running process). `scalene` (CPU + GPU + memory). `line_profiler` (per-line CPU). `memray` (per-allocation memory).

**Storage.** `safetensors` (safe alternative to pickle for model weights). `zarr` (chunked array storage). `lmdb` (memory-mapped key-value store). `webdataset` (tar-based streaming for vision).

**Reproducibility.** `dvc` (git for data). `nbstripout` (strip notebook outputs before commit). `mlflow` for experiment + model versioning.

**Causality & uncertainty.** `dowhy`, `EconML` for causal inference. `NumPyro` / `PyMC` for Bayesian modelling. `arviz` for posterior diagnostics.

**Speed-ups.** `numba` (JIT for numpy). `cython` (compile-to-C for hot loops). `cupy` (numpy on GPU). `jax` (numpy + autodiff + JIT + vmap — the underrated alternative to PyTorch for research).

**Concurrency.** `joblib` (parallel for-loops). `dask` (pandas / numpy at scale). `ray` (distributed Python, including Ray Train + Ray Serve).

**The "I write this in every project" set.** A function to fix all seeds. A function to log GPU memory. A function to count parameters. Put them in `utils/`. Steal from your last project.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# A few utility functions worth copying between projects

import os, random
import numpy as np
import torch

def fix_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def gpu_memory_summary():
    if not torch.cuda.is_available(): return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"alloc {alloc:.2f} GB  reserved {reserved:.2f} GB")
```

</div>

<div class="level-next">
<span>Too dense?</span>
<button data-go-to="fundamentals" type="button">← Back to Standard</button>
</div>

</section>

<!-- TOPIC SIDEBAR -->

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="fundamentals" markdown="1">

[uv — Python Package Manager <i class="fas fa-external-link-alt"></i>](https://docs.astral.sh/uv/){: target="_blank" }
<span class="annotation">Astral's pip / venv / poetry replacement. 10–100× faster. The new default.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[einops <i class="fas fa-external-link-alt"></i>](https://einops.rocks/){: target="_blank" }
<span class="annotation">Tensor operations with named axes. Replaces <code>view</code> / <code>permute</code> with code your reviewer can read. Worth learning thoroughly.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Polars <i class="fas fa-external-link-alt"></i>](https://www.pola.rs/){: target="_blank" }
<span class="annotation">The pandas successor. Faster, more memory-efficient, lazy execution. Reads / writes everything pandas does.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[awesome-python <i class="fas fa-external-link-alt"></i>](https://github.com/vinta/awesome-python){: target="_blank" }
<span class="annotation">Curated list of useful Python libraries by category. Not ML-specific; still useful for the broader Python tools.</span>

</li>
</ul>

</div>

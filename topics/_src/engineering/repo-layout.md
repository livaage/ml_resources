---
title: Repository Layout — ML Resources Hub
eyebrow_text: ← Engineering · Project Structure
eyebrow_href: {{root}}engineering.html
heading: Repository Layout
lead: Where to put your data, models, configs, and code so future-you doesn't curse past-you.
active_nav: engineering
next_href: configuration.html
next_title: Configuration Management
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Separate the things that change at different rates.** Code changes daily, configs change per-experiment, data and models change rarely (and live elsewhere). Put each in its own directory; never mix them.

</div>

<article class="tldr-body" markdown="1">

The single biggest predictor of an ML project becoming unmaintainable is everything-in-one-notebook. Six months in, nobody — including you — knows which version of which preprocessing belongs to which model.

A clean repo splits responsibilities. A reasonable starting layout for a research project:

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```text
my-project/
├── README.md
├── pyproject.toml          # dependencies + project metadata
├── .gitignore
├── configs/                # YAML/Hydra configs (one per experiment)
├── data/                   # gitignored — actual data lives elsewhere
├── notebooks/              # exploration only; no business logic
├── src/my_project/         # the importable package
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   └── eval.py
├── scripts/                # CLI entry points
│   └── train.py
└── tests/
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Stick to it when

- You're starting any project meant to last more than a week
- Multiple people will work on the code
- You want to reproduce an experiment 3 months from now
- You'll eventually package or deploy the code

</div>

<div class="no" markdown="1">

### Don't over-engineer when

- It's a one-off experiment that lives in a single notebook
- You're prototyping for an afternoon and will throw it away
- The "project" is two files of glue between two libraries
- Adding structure would slow you down more than it helps

</div>

</div>

<div class="level-next">
<span>Want a concrete starter template and the reasoning behind each directory?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

Use **src layout** (your code is an importable package in `src/`) and **config-driven entry points** (scripts in `scripts/` read configs, never embed hyperparameters). Treat notebooks as scratch paper, not source of truth.

</div>

<article class="tldr-body" markdown="1">

The "src layout" puts your package one directory deeper than the project root. Why bother? It prevents you from accidentally importing your code from the project root instead of the installed version — a real bug that wastes hours when tests pass locally but fail in CI.

**Why scripts/ separately from src/?** Things in `src/my_project/` are *library code*: importable, testable, no `argparse`. Things in `scripts/` are *entry points*: argparse / Hydra, side effects, calls into the library. This split makes the library reusable from notebooks, CLIs, and other scripts without modification.

**Notebooks are scratch paper.** Notebooks for exploration, plotting, debugging. Anything you'd want to call again belongs in `src/`. Promote code out of notebooks aggressively — every function that survives to a second notebook should be in the package.

**Don't commit data.** Even small data. Use `.gitignore` + a download script (`scripts/download_data.py`) or a versioned store (DVC, S3 with a config). Repos with committed CSVs are repos that go bad.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```text
my-project/
├── pyproject.toml                 # uv / pip / poetry — pick one
├── .python-version
├── .gitignore
├── README.md
├── configs/
│   ├── base.yaml                  # shared defaults
│   ├── data/mnist.yaml
│   ├── model/resnet18.yaml
│   └── experiment/exp01.yaml
├── data/                          # GITIGNORED
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── 01-data-exploration.ipynb
├── src/my_project/
│   ├── __init__.py
│   ├── data/                      # data loading + transforms
│   ├── models/                    # model definitions
│   ├── training/                  # training loop, callbacks
│   ├── eval/                      # metrics, reports
│   └── utils.py
├── scripts/
│   ├── train.py                   # python scripts/train.py +experiment=exp01
│   ├── eval.py
│   └── serve.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── conftest.py
└── outputs/                       # GITIGNORED — Hydra writes runs here
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Project is medium-sized (≥ 1 month, ≥ 1 person)
- You'll run multiple experiments with different configs
- You want to publish the code or share it with collaborators
- You'll deploy a model from this codebase

</div>

<div class="no" markdown="1">

### Lighter shape when

- Quick proof-of-concept that won't survive a week
- Single-file utility that wraps an existing library
- Notebook-driven analysis on a single dataset
- You're learning, not building

</div>

</div>

<div class="level-next">
<span>Want monorepo tradeoffs, naming conventions, and templates?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

Beyond the basic shape, the harder decisions are about **boundaries** — package per concern vs. flat package, monorepo vs. polyrepo, where to draw the line between library and application code, and how to keep coupling low across teams.

</div>

<article class="tldr-body" markdown="1">

**Cookiecutter / templates.** Don't hand-roll. [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/){: target="_blank" } is the canonical research-project template. [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template){: target="_blank" } for PyTorch Lightning + Hydra. [cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv){: target="_blank" } for modern uv-based projects. Pick one, customize once, stop bikeshedding.

**Monorepo vs. polyrepo.** ML projects often involve a data-prep service, a training service, a serving service, and a shared library. Monorepo (Bazel, Pants, uv workspaces) makes refactors across boundaries cheap but raises CI cost. Polyrepo isolates and versions independently but creates coordination overhead. For ≤ 4 services / 1 team: monorepo wins. Beyond that it depends.

**Library / application split.** The hardest line. Anything reusable (data loaders, model definitions, metrics) is library; anything specific to one experiment (hyperparameters, paths, schedules) is application. The test: could another project consume this without modification? If yes, it's library. If no, it doesn't belong in `src/`.

**Versioning & releases.** Use `importlib.metadata.version` to surface the package version in logs and run metadata. Tag releases (git tags + semver). Log the git SHA on every training run; you'll thank yourself when "model A from June" needs reproducing in October.

**Naming conventions.** Boring is good. Plurals for collections (`models/`, not `model/`). Verbs for action modules (`training.py`), nouns for data structures (`dataset.py`). Avoid `utils.py` if you can — it tends to become a graveyard.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Multi-service ML platform — monorepo plus shared libs
- Library you intend others to depend on
- Production deployment with strict release process
- You're standardizing across multiple internal ML teams

</div>

<div class="no" markdown="1">

### Skip it when

- Research project where boundaries shift weekly
- Solo dev — coordination overhead outweighs the structure
- Throw-away exploration code
- Strict velocity constraints prevent template setup

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```toml
# pyproject.toml — modern src-layout project (uv-friendly)
[project]
name = "my-project"
version = "0.1.0"
description = "ML research project"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "hydra-core>=1.3",
    "wandb>=0.16",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.5", "mypy>=1.10"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/my_project"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"

[project.scripts]
train = "my_project.cli:train"
eval  = "my_project.cli:eval"
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

[Cookiecutter Data Science <i class="fas fa-external-link-alt"></i>](https://cookiecutter-data-science.drivendata.org/){: target="_blank" }
<span class="annotation">The canonical research-project template. Read the rationale section even if you don't use the template directly.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[lightning-hydra-template <i class="fas fa-external-link-alt"></i>](https://github.com/ashleve/lightning-hydra-template){: target="_blank" }
<span class="annotation">Modern PyTorch Lightning + Hydra starter. Strong opinions, sensible defaults — copy and adapt.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyPA — src vs flat layout <i class="fas fa-external-link-alt"></i>](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/){: target="_blank" }
<span class="annotation">Definitive comparison from the Python Packaging Authority. The "why src layout" argument in one short page.</span>

</li>
<li data-tier="indepth" markdown="1">

[Fowler — Monorepo vs Polyrepo <i class="fas fa-external-link-alt"></i>](https://martinfowler.com/articles/monorepo.html){: target="_blank" }
<span class="annotation">Domain-neutral but the trade-offs apply directly to ML platforms. Read before choosing.</span>

</li>
</ul>

</div>

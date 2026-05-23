---
title: Environments &amp; Dependencies — ML Resources Hub
eyebrow_text: ← Engineering · Project Structure
eyebrow_href: {{root}}engineering.html
heading: Environments &amp; Dependencies
lead: uv, conda, Docker — choosing the right level of isolation for your workflow.
active_nav: engineering
prev_href: reproducibility.html
prev_title: Reproducibility
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Every project gets its own Python environment.** Never install packages globally. Never share an environment between projects. The cost of setup is ten minutes; the cost of skipping it is six months of cryptic version conflicts.

</div>

<article class="tldr-body" markdown="1">

Two projects, both needing different versions of `torch` or `numpy`. Without isolation: one breaks. With isolation: both work. The same applies to Python versions, CUDA versions, system libraries.

The modern recommendation: **use [uv](https://docs.astral.sh/uv/){: target="_blank" }** (a fast, sane replacement for pip/virtualenv/poetry, written in Rust). [conda/mamba](https://docs.conda.io/){: target="_blank" } is the classic alternative; useful when you need non-Python dependencies. For deployment, wrap in a Docker container.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```bash
# With uv — fast, modern, recommended
uv init my-project
cd my-project
uv add torch numpy
uv run python train.py    # automatically uses the project env

# Or with the classic approach
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Isolate when

- Always. Every Python project, every time.
- You're using a different Python version than system Python
- You'll share the project with anyone (including future-you)
- The project will run on more than one machine

</div>

<div class="no" markdown="1">

### Skip only when

- You're running a single one-off script with no dependencies
- You're in an interactive REPL just trying something out

</div>

</div>

<div class="level-next">
<span>Want pip vs conda vs uv, lockfiles, and Docker?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

The choice has two axes. **Python-only or polyglot?** (pip / uv handle Python; conda handles non-Python like CUDA, MKL.) **Local development or production deployment?** (Local: env manager. Production: container.)

</div>

<article class="tldr-body" markdown="1">

**uv** (recommended for new projects) is a drop-in replacement for pip + virtualenv + pip-tools + poetry, written in Rust. ~10-100× faster than pip. Manages Python versions too (replaces pyenv). Single tool, no shell activation needed. If you're starting fresh, use this.

**pip + venv**: the bare-bones option. Use `pip-tools` (or `pip freeze`) to produce a lockfile from your `requirements.in`. Fine for simple projects; uv just makes the same workflow faster.

**conda / mamba**: when you need non-Python dependencies — CUDA toolkit, system libraries, MKL, R, etc. Heavier than pip but invaluable when the alternative is "install CUDA system-wide". `mamba` is the fast C++ implementation; almost always use it instead of `conda` directly.

**poetry**: was the modern choice before uv. Still solid; uv is faster and more focused. New projects: pick uv. Existing poetry projects: migrate when you have time.

**Docker**: for deployment and CI, not local development. Start `FROM python:3.11-slim`, install your locked dependencies, copy your code, set the entrypoint. The container is the unit of "deployable artefact" — even your training jobs should run in one for production work.

**Lockfiles matter.** `pyproject.toml` says "I want torch ≥ 2.0"; the lockfile (`uv.lock`, `poetry.lock`) records "I got torch 2.4.1, numpy 1.26.4, ...". Commit the lockfile. CI builds from the lockfile. This is what makes "reproducible environment" actually true.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```bash
# uv workflow — modern Python project from zero
uv init my-project --python 3.11
cd my-project

# Add dependencies
uv add "torch>=2.0" hydra-core wandb
uv add --dev pytest ruff mypy

# Install (creates .venv/, writes uv.lock)
uv sync

# Run scripts
uv run python scripts/train.py +experiment=exp01

# Update dependencies
uv lock --upgrade-package torch

# Run a tool without adding it
uv run --with ipython python
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- **uv:** any new Python project — research or production
- **conda / mamba:** need non-Python deps (CUDA, R, system libs)
- **Docker:** deployment, CI, multi-machine training, isolation from host
- **pip + venv:** minimal projects with no extra ceremony needed

</div>

<div class="no" markdown="1">

### Skip / migrate when

- **Anaconda:** the bundled distribution — heavy; prefer Miniconda or mamba
- **Pipenv:** mostly unmaintained — migrate to uv or poetry
- **Global pip install:** never, on any machine you care about
- **Custom shell scripts:** use a real tool instead

</div>

</div>

<div class="level-next">
<span>Want GPU-specific Docker, multi-stage builds, and dependency hell debugging?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

For production ML, you need **reproducible builds** from source to deployed container. That means pinned base images, locked dependencies, and CI that builds the artefact deterministically — not just `docker build` with a `:latest` base.

</div>

<article class="tldr-body" markdown="1">

**Base image discipline.** Pin a specific digest: `FROM python:3.11-slim@sha256:abcdef…`, not `FROM python:3.11`. The `3.11` tag rolls forward when the upstream image is rebuilt; the digest doesn't. For GPU work: [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda){: target="_blank" } images at a specific tag, or PyTorch's official [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch){: target="_blank" }.

**Multi-stage builds.** A "builder" stage installs build tools and compiles wheels; a "runtime" stage copies only the installed packages + your code. Final image is smaller and has fewer attack surfaces. Essential for production deployments.

**Dependency hell on GPU stacks.** CUDA version, cuDNN, NCCL, PyTorch, and the kernel driver all need to be compatible. Use PyTorch's published wheel matrix as ground truth, or install via conda/mamba which pulls compatible CUDA runtimes. Common failure: PyTorch built for CUDA 12.1 on a host with CUDA 11.8 driver.

**Caching strategies for fast iteration.** Layer order in Dockerfile matters: copy `pyproject.toml` + `uv.lock` first, install deps, then copy source. Only the source layer rebuilds on code changes — deps stay cached. With `uv`, pass `--mount=type=cache,target=/root/.cache/uv` in BuildKit to cache across builds.

**Editable installs vs. wheels.** Local dev: editable (`pip install -e .` or `uv pip install -e .`) so code changes are picked up without reinstall. Production: build a wheel and install it. Don't ship editable installs to production — they leak source paths and assume a writable filesystem layout.

**Vendoring vs. dynamic resolution.** For maximum reproducibility (think regulated ML), vendor wheels into a private registry. Your build never touches PyPI. Heavier but bulletproof — no "wheel withdrawn from PyPI" surprises.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```dockerfile
# Multi-stage Dockerfile for a uv-based ML project
FROM python:3.11-slim AS builder

# Install uv
RUN pip install --no-cache-dir uv==0.4.20

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Install deps into a venv at /app/.venv (fast with uv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src/ ./src/
RUN uv sync --frozen

# ---------- Runtime stage ----------
FROM python:3.11-slim AS runtime

# Copy only the resolved environment + code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src   /app/src
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app/src"

ENTRYPOINT ["python", "-m", "my_project.cli"]
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

[uv documentation <i class="fas fa-external-link-alt"></i>](https://docs.astral.sh/uv/){: target="_blank" }
<span class="annotation">Modern Python project tooling. The "Concepts" section explains how uv differs from pip / virtualenv / poetry — read it once before adopting.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[mamba documentation <i class="fas fa-external-link-alt"></i>](https://mamba.readthedocs.io/){: target="_blank" }
<span class="annotation">Fast conda-compatible package manager. Use this when you need non-Python deps (CUDA, system libs).</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Docker — Build best practices <i class="fas fa-external-link-alt"></i>](https://docs.docker.com/build/building/best-practices/){: target="_blank" }
<span class="annotation">The official guide. Multi-stage builds, layer caching, and image size hygiene — the parts that actually matter for production.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[uv + Docker guide <i class="fas fa-external-link-alt"></i>](https://github.com/astral-sh/uv/blob/main/docs/guides/integration/docker.md){: target="_blank" }
<span class="annotation">Specific recipes for using uv inside Docker. Saves you the trial-and-error.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyTorch install matrix <i class="fas fa-external-link-alt"></i>](https://pytorch.org/get-started/locally/){: target="_blank" }
<span class="annotation">Authoritative source for matching CUDA / cuDNN / driver combinations to PyTorch wheels. Bookmark this.</span>

</li>
</ul>

</div>

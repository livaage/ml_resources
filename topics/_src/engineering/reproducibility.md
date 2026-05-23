---
title: Reproducibility — ML Resources Hub
eyebrow_text: ← Engineering · Project Structure
eyebrow_href: {{root}}engineering.html
heading: Reproducibility
lead: Seeds, deterministic ops, locked dependencies — making "it worked on my machine" actually portable.
active_nav: engineering
prev_href: configuration.html
prev_title: Configuration Management
next_href: env-deps.html
next_title: Environments &amp; Dependencies
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Same code + same data + same seed should produce the same result.** If it doesn't, you can't reliably tell whether your latest change improved the model or just rolled a luckier dice. Three things to control: random seeds, dependency versions, and data versions.

</div>

<article class="tldr-body" markdown="1">

You change a line of code, retrain, and accuracy jumps 0.5%. Was it the code change, or just stochastic luck? Without reproducibility, you can't say.

The bare minimum: set every random seed you can, log the seed alongside results, and lock your dependencies so an unrelated package update doesn't silently change behaviour. Do it on day one; it costs nothing then and is painful to retrofit.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Any project where you'll compare results across runs
- You publish results or share them with collaborators
- You want to debug "why did this work yesterday and not today"
- You need to defend a model's behaviour in production

</div>

<div class="no" markdown="1">

### Skip it when

- Truly throwaway experiments where variation across runs is informative
- You explicitly want to study stochastic effects (average across many seeds)
- Cost of exact reproducibility (e.g. forcing deterministic CUDA) outweighs benefit

</div>

</div>

<div class="level-next">
<span>Want the three pillars in detail?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Three pillars</span>

**(1) Seeds.** Every PRNG seeded and the seed logged.<br/>
**(2) Dependencies.** Exact package versions pinned and the lockfile checked in.<br/>
**(3) Data.** Versioned (DVC / S3 / hash of inputs) and the version logged with each run.

</div>

<article class="tldr-body" markdown="1">

**Seeds.** Set the seed for every PRNG you touch: Python's `random`, NumPy, PyTorch (CPU + CUDA), and any library that has its own (hf `set_seed`, scikit-learn estimators' `random_state`). Log the seed in your config / output so the run is reproducible. Different seeds for different things if you want to ensure independence (data split seed, model init seed, augmentation seed).

**Deterministic operations.** Even with a seed, some CUDA operations are non-deterministic by default (cuDNN's auto-tuner, atomics). Set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for strict reproducibility, accepting the speed hit. Often not worth it for research — but is for regulatory contexts.

**Dependencies.** Use a lockfile that pins exact versions (uv.lock, poetry.lock, pip-tools' requirements.txt). `pyproject.toml` says what you want; lockfile says what you got. CI builds from the lockfile. Without this, a transitive dependency can update overnight and silently change your model's outputs.

**Data versioning.** Just like code, your data needs a version. Three popular approaches: hash the data files (and log the hashes); use [DVC](https://dvc.org/){: target="_blank" } (git-like CLI over external storage); or store immutable snapshots in S3 with timestamps. Pick one. The goal is "given a run's logs, I can locate the exact data it was trained on".

**Track the code version too.** Log the git SHA on every run. If you have uncommitted changes, log a diff (or refuse to run from a dirty tree). "Trained on commit *abc123*" beats "trained sometime last Tuesday".

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import os, random, hashlib, subprocess
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def git_sha() -> str:
    """Current commit SHA — refuse if working tree is dirty."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"

def file_hash(path: str) -> str:
    """SHA256 of a file — log this to identify exact data versions."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

# At the start of every run
run_metadata = {
    "seed": 42,
    "git_sha": git_sha(),
    "data_hash": file_hash("data/processed/train.parquet"),
}
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You're tracking experiments to choose between models
- Collaborating across machines or cloud / on-prem
- Building a benchmark or publishing results
- The model goes anywhere near production

</div>

<div class="no" markdown="1">

### Skip when

- You're aggregating over many seeds anyway (stochasticity is part of the analysis)
- Exact determinism costs more than its worth (some research code)
- Quick ad-hoc analysis where you'll never repeat the run

</div>

</div>

<div class="level-next">
<span>Want determinism caveats, DVC, and the "is this even reproducible?" check?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Bit-exact reproducibility is harder than it looks.** Hardware differences, library versions, and parallel-reduction non-determinism can defeat the simplest reproducibility setup. Decide what level you actually need and design for it.

</div>

<article class="tldr-body" markdown="1">

**Levels of reproducibility.** (1) *Run-to-run on the same machine*: easy. (2) *Across machines with the same hardware*: hard. (3) *Across different GPUs / CPUs*: bit-exact is essentially impossible; statistically equivalent is the realistic target.

**CUDA non-determinism.** Atomic operations on GPU (used in many backward passes) introduce non-determinism even with seeds set. `torch.use_deterministic_algorithms(True)` errors on calls without a deterministic equivalent. [CuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility){: target="_blank" } requires specific workspace configs. The speed hit can be 10-30%.

**Mixed precision and order-of-operations.** FP16 / BF16 training is non-associative — different batch orderings yield slightly different results even with identical seeds. Beyond the "model trains fine" bar, this can matter for unit tests of numerical kernels.

**Data versioning options.** [DVC](https://dvc.org/){: target="_blank" }: git-like commands for data, pluggable storage backends. [lakeFS](https://lakefs.io/){: target="_blank" }: branches and merges over object storage. [Pachyderm](https://docs.pachyderm.com/){: target="_blank" }: pipeline-aware versioning. Simpler: hashed directory names + S3 paths in your run metadata. The right tool depends on data size, mutability, and team size.

**The reproducibility audit.** Periodically retrain a "golden" model from a fixed commit + data hash + seed and check it gets the same metric to within tolerance. If not, find out why. Common causes: a newer CUDA version, a transitive dependency update, a corrupted data file.

**Reproducibility ≠ correctness.** A run can be perfectly reproducible and still wrong. Pair reproducibility checks with proper testing (unit tests for data preprocessing, regression tests for full pipelines).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, hashlib, pathlib

# 1. Strict deterministic mode (slower but bit-reproducible on same hardware)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 2. Independent generators per task — clearer than reusing one global seed
data_gen   = torch.Generator().manual_seed(42)
model_gen  = torch.Generator().manual_seed(100)
shuffle_gen = torch.Generator().manual_seed(7)

# 3. Hash data directory contents (cheap reproducibility check)
def hash_dir(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.stat().st_size.to_bytes(8, "little"))
    return h.hexdigest()[:16]

assert hash_dir(pathlib.Path("data/processed")) == EXPECTED_HASH, \
    "data has changed since last run — verify before training"
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

[PyTorch — Reproducibility <i class="fas fa-external-link-alt"></i>](https://pytorch.org/docs/stable/notes/randomness.html){: target="_blank" }
<span class="annotation">Definitive list of what's deterministic vs. not in PyTorch, with the specific knobs to flip. Read once, refer back often.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[DVC — Get Started <i class="fas fa-external-link-alt"></i>](https://dvc.org/doc/start){: target="_blank" }
<span class="annotation">Quickest way to add data versioning. Skip the ML pipeline parts if you only need versioning.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[The Turing Way — Reproducibility <i class="fas fa-external-link-alt"></i>](https://reproducibility-guide.github.io/){: target="_blank" }
<span class="annotation">Broader guide covering reproducibility in computational research. Useful even outside ML.</span>

</li>
<li data-tier="indepth" markdown="1">

[ML Reproducibility Challenge <i class="fas fa-external-link-alt"></i>](https://paperswithcode.com/rc2022){: target="_blank" }
<span class="annotation">Annual challenge attempting to reproduce published ML papers. Reports document the typical failure modes — instructive.</span>

</li>
</ul>

</div>

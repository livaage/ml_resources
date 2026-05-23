---
title: Testing ML Code — ML Resources Hub
eyebrow_text: ← Engineering · Development Loop
eyebrow_href: {{root}}engineering.html
heading: Testing ML Code
lead: Unit tests on tensors, integration tests on pipelines, data tests on inputs — applying normal-software discipline to ML.
active_nav: engineering
prev_href: hyperparameter-search.html
prev_title: Hyperparameter Search
next_href: logging-debugging.html
next_title: Logging &amp; Debugging
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**ML code is still code.** The hard part of ML code is what happens with the data and the model; the rest is plain Python that benefits from the same testing you'd write for anything else. Plus three ML-specific tests: shape, range, gradient flow.

</div>

<article class="tldr-body" markdown="1">

The disasters: silent shape bugs (broadcasting where you didn't expect), gradient-stopping mistakes (a `.detach()` in the wrong place), data corruption (NaN propagation), reproducibility loss (a seed didn't take). All of these are catchable with cheap tests.

**Five tests every ML repo should have.** (1) Forward pass produces the right output shape. (2) Backward pass produces non-zero gradients on every learnable parameter. (3) A single training step decreases the loss on a tiny batch. (4) Loaded checkpoints reproduce exactly. (5) The same seed produces the same output.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### What to test

- **Shapes**: inputs, outputs, intermediate activations
- **Numeric range**: no NaNs / Infs after forward / backward
- **Gradients**: non-zero on every trainable parameter
- **Loss decrease**: 100 steps on 1 batch should overfit it
- **Determinism**: same seed → same loss to 1e-6
- **Data pipeline**: schemas, ranges, missing rates

</div>

<div class="no" markdown="1">

### What to NOT test

- Final accuracy on the real dataset (too slow + brittle)
- Exact-equality of arbitrary intermediate tensors (floating-point)
- Library internals (PyTorch already tests itself)
- Anything that needs a GPU (run those separately, less often)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import pytest
import torch
from my_model import MyNet

@pytest.fixture
def model(): return MyNet(in_dim=128, out_dim=10)

def test_forward_shape(model):
    x = torch.randn(4, 128)
    assert model(x).shape == (4, 10)

def test_grad_flow(model):
    x = torch.randn(4, 128); y = torch.randint(10, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad: {name}"
        assert (p.grad != 0).any(), f"all-zero grad: {name}"

def test_overfit_tiny_batch(model):
    x = torch.randn(2, 128); y = torch.tensor([3, 7])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward(); opt.step()
    assert loss.item() < 0.1, f"can't overfit 2 samples: {loss.item()}"
```

</div>

<div class="level-next">
<span>Want data tests, property-based tests, & CI integration?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The test pyramid for ML</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \underbrace{\text{unit (fast, many)}}_{\text{shapes, ranges, grads}} \;\;\;>\;\;\; \underbrace{\text{integration (medium)}}_{\text{1-batch overfit, save/load}} \;\;\;>\;\;\; \underbrace{\text{end-to-end (slow, few)}}_{\text{tiny full pipeline}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Many cheap tests at the bottom — they catch most bugs

</li>
<li markdown="1">

A few expensive tests at the top — only what really requires it

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \underbrace{\text{lots of unit tests}}_{\text{shapes, ranges, grads}} \;\;>\;\; \underbrace{\text{some integration tests}}_{\text{1-batch overfit, save/load}} \;\;>\;\; \underbrace{\text{few end-to-end tests}}_{\text{tiny full pipeline}} $$</span>

**In words.** A pyramid of test layers, with the wider (more numerous) layer at the bottom. `>` here just means "many more of these than of those". Unit tests are individual function checks — cheap, fast, run every save. Integration tests glue a few units together (one batch through the model). End-to-end tests run the whole pipeline on a tiny dataset. The underbraces (`⏟`) annotate what each layer typically checks. Test pain comes from inverting this shape — the moment you have more slow tests than fast ones, the suite stops getting run.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

Many cheap tests at the bottom — they catch most bugs

</li>
<li markdown="1">

A few expensive tests at the top — only what really requires it

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Data validation.** Schema (column names, types). Ranges (age > 0). Missingness (fewer than X% nulls per column). Cardinality (distinct categories within expected set). Tools: Great Expectations, Pandera, deequ.

**Property-based testing.** Hypothesis library — generate random tensors and assert properties (output shape always matches input batch size; gradient norm bounded). Catches edge cases hand-written tests miss.

**Snapshot tests.** Save the output of a deterministic pipeline once, assert it matches on subsequent runs. Catches accidental behaviour changes; brittle against intentional ones — has to be regenerated when you change the model.

**Integration test recipe.** A tiny synthetic dataset (10 examples), a 1-batch training run, a save-and-reload, an inference call. Total runtime: under a second. Catches 90% of "did I break the pipeline" bugs.

**CI for ML.** Run the unit + fast integration tests on every PR. Run slower (1-minute) end-to-end tests on merge. Keep GPU tests on a separate workflow that runs on demand or nightly. See [CI for ML](ci-for-ml.html).

**Test the failure modes.** What happens with an empty batch? With NaN inputs? With a class that wasn't seen at training? Each of these usually warrants a test that asserts an explicit error rather than silent garbage.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import pandera as pa
from pandera import Column, Check
from hypothesis import given, strategies as st
import torch

# Pandera schema for input data
schema = pa.DataFrameSchema({
    "age":    Column(int,   Check.in_range(0, 120)),
    "income": Column(float, Check.greater_than_or_equal_to(0)),
    "city":   Column(str,   Check.isin(["London", "Paris", "Berlin"])),
})

def load_data(path):
    df = pd.read_csv(path)
    return schema.validate(df)               # raises if input doesn't conform

# Property-based: model behaviour across random shapes
@given(
    batch=st.integers(min_value=1, max_value=64),
    dim=st.integers(min_value=4, max_value=256),
)
def test_forward_arbitrary_shape(batch, dim):
    model = MyNet(in_dim=dim, out_dim=10)
    x = torch.randn(batch, dim)
    out = model(x)
    assert out.shape == (batch, 10)
    assert not out.isnan().any()
```

</div>

<div class="level-next">
<span>Want fuzzing, mutation testing, golden datasets, & flake hunting?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Determinism budget</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \|\text{out}(s_1) - \text{out}(s_2)\|_\infty < \epsilon $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Two seeded runs should agree to *ε*

</li>
<li markdown="1">

cuDNN nondeterminism, atomic adds, multi-GPU sync — all break this

</li>
<li markdown="1">

The right ε is small but not zero on GPU

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{biggest difference between outputs of two seeded runs} \;<\; \epsilon $$</span>

**In words.** Two runs with the same code, same data, same seed should produce outputs that differ by at most `ε` (epsilon, a tiny tolerance number you pick). `‖·‖∞` ("infinity norm") just means "the biggest absolute difference across the whole output tensor". On CPU you can usually demand `ε = 0`; on GPU, atomic-add ordering and cuDNN's auto-tuner inject small non-determinism so you need a small positive `ε`. Setting `ε` too tight makes your tests flaky; too loose and they miss real regressions.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`biggest difference`worst-case discrepancy between the two output tensors

</li>
<li markdown="1">

`two seeded runs`same code + same seed, run twice (different seeds `s₁, s₂` would be a more relaxed test)

</li>
<li markdown="1">

`ε`tolerance — small but not zero on GPU

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Golden datasets.** Tiny, hand-curated, labelled by you, kept in the repo. Run the trained model on these every CI; assert accuracy > threshold and that specific edge-case predictions still pass. Catches regression from accidental changes.

**Mutation testing.** Tools like mutmut systematically introduce small bugs into your code (flipping conditions, changing constants) and check if your tests catch them. If a mutation survives, the test suite has a gap.

**Flaky test hunting.** ML tests that fail intermittently — almost always due to seeded but non-deterministic ops, or to data loaders with random sampling without seeds. Use `pytest-rerunfailures` to surface them; fix the root cause, don't just retry.

**Test data versioning.** Your tests are only as stable as their data. DVC / LakeFS for test datasets, or commit a small parquet directly. Either way: identify each test by its data version.

**Determinism on GPU.** cuDNN benchmark mode picks fastest kernels, which can differ between runs. `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` get you most of the way; some ops still aren't deterministic. Tests that assert bit-exact GPU equality are usually a mistake.

**Differential testing.** Compare two implementations of the same thing — your model vs a reference implementation, your CUDA kernel vs the CPU version, the new optimiser vs the old one. Run both on the same input, assert output close.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import os, random
import numpy as np
import torch

def fully_deterministic(seed=42):
    """Best-effort determinism on PyTorch + CUDA."""
    os.environ["PYTHONHASHSEED"]      = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Differential test — model.eval() should equal a reference
def test_against_reference(model, ref_model, x):
    model.eval(); ref_model.eval()
    with torch.no_grad():
        a = model(x); b = ref_model(x)
    assert (a - b).abs().max() < 1e-4, (a - b).abs().max()
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

[Eugene Yan — Testing ML Systems <i class="fas fa-external-link-alt"></i>](https://eugeneyan.com/writing/testing-ml/){: target="_blank" }
<span class="annotation">Practical taxonomy of ML tests (data, model, pipeline) with examples. Best one-stop overview.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[pytest Documentation <i class="fas fa-external-link-alt"></i>](https://docs.pytest.org/){: target="_blank" }
<span class="annotation">The Python testing framework. Fixtures, parametrize, plugins — read at least the basics if you're doing serious ML engineering.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Pandera <i class="fas fa-external-link-alt"></i>](https://pandera.readthedocs.io/){: target="_blank" }
<span class="annotation">Statistical schema validation for pandas / Polars DataFrames. Cleanest API for "this column should be int in [0, 120]" checks.</span>

</li>
<li data-tier="indepth" markdown="1">

[Hypothesis — Property-Based Testing <i class="fas fa-external-link-alt"></i>](https://hypothesis.readthedocs.io/){: target="_blank" }
<span class="annotation">Generate random inputs to your tests. Catches edge cases you'd never write by hand.</span>

</li>
</ul>

</div>

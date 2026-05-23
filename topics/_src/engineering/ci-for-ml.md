---
title: CI for ML — ML Resources Hub
eyebrow_text: ← Engineering · CI / CD
eyebrow_href: {{root}}engineering.html
heading: CI for ML
lead: GitHub Actions for an ML repo — what to run on every PR, what to schedule, what's worth a GPU.
active_nav: engineering
prev_href: logging-debugging.html
prev_title: Logging &amp; Debugging
next_href: data-validation.html
next_title: Data Validation
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Run the cheap checks on every PR; run the expensive ones less often.** Lint + types + unit tests + 1-batch overfit on CPU on every push. Full training runs on a schedule. GPU-required tests on demand or nightly. Don't make every contributor wait for a model to train.

</div>

<article class="tldr-body" markdown="1">

The setup most repos converge on: GitHub Actions for orchestration, three workflow files (PR-fast, PR-slow, nightly), Docker for env consistency, and clear separation between "lint and test the code" and "actually validate the model".

**What runs on every PR.** ruff / black / mypy (linting + types). Unit tests on CPU. The 1-batch overfit smoke test. A data-schema validation. Total time target: under 5 minutes.

**What runs less often.** Full training on a tiny dataset. Integration test against a held-out evaluation set. GPU benchmarks. Model registry promotion gates.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```yaml
# .github/workflows/pr.yml — runs on every push to a PR
name: PR Checks
on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install uv && uv pip install --system -r requirements.txt
      - run: ruff check .
      - run: mypy --strict src/
      - run: pytest -q --cov=src tests/unit
      - run: pytest -q tests/smoke      # 1-batch overfit, schema checks
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Three workflow tiers

- **PR (every push)**: lint, types, unit tests, 1-batch overfit. < 5 min.
- **Nightly**: full pipeline on tiny dataset, model evals on held-out set, security scan
- **On-demand / release**: full training, GPU benchmarks, model registry promotion

</div>

<div class="no" markdown="1">

### Common mistakes

- Running everything on every PR — devs stop reading red builds
- Caching deps badly — flaky installs slow every run
- No matrix testing across Python / CUDA versions you actually support
- Storing model artefacts in the repo (use a registry instead)

</div>

</div>

<div class="level-next">
<span>Want matrix builds, caching, & GPU CI strategies?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">CI cost model</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{cost} = (\text{frequency}) \times (\text{duration}) \times (\text{rate}) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

The PR loop is high-frequency, low-duration — optimise duration

</li>
<li markdown="1">

The release loop is low-frequency, high-duration — duration matters less

</li>
<li markdown="1">

Spend GPU-minutes where they actually catch bugs

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{total cost} \;=\; (\text{how often it runs}) \times (\text{how long each run takes}) \times (\text{$ per minute}) $$</span>

**In words.** CI bills look intimidating but break down into three multiplicative factors. The product structure tells you where to optimise: cutting any factor in half halves the total. For the PR loop, you can't easily change frequency (developers push when they push) and rate is set by your runner choice, so the only knob is *duration* — that's why caching, parallel jobs, and skipping expensive checks on PR pay off so much. For release-day full training, duration is large but rate is amortised across many PRs, so it's worth running.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`frequency`how many times per day/week this workflow runs

</li>
<li markdown="1">

`duration`wall-clock minutes per run

</li>
<li markdown="1">

`rate`$ per minute for that runner (CPU, GPU, self-hosted)

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Cache aggressively.** pip / uv cache, model checkpoints, dataset downloads, Docker layers. Most ML CI time is waiting for installs and downloads. `actions/cache` with a key on the lockfile hash gets you 90% of the way.

**Matrix builds.** Test against the Python + CUDA + torch versions you support. `strategy.matrix` in GitHub Actions; `fail-fast: false` so one failure doesn't kill the others.

**GPU CI.** Self-hosted runners with GPUs, or paid services (Modal, CoreWeave, RunPod). Run only the tests that genuinely need a GPU — most ML code can be unit-tested on CPU.

**Container-based CI.** Build a Docker image with your env once; reuse across jobs. `nvcr.io/nvidia/pytorch:24.05-py3` is a sensible base; pin the tag.

**Pre-commit hooks.** Run lint + format + simple checks locally before the commit lands. Pre-commit catches typos and format issues without burning CI minutes.

**Branch protection.** Require PR review, require all checks pass, require linear history. The "checks pass" is what turns CI into actual enforcement instead of a vibes-based warning.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```yaml
name: PR Checks
on: [pull_request]

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.11", "3.12"]
        torch:  ["2.4", "2.5"]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python }} }
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ matrix.python }}-${{ hashFiles('uv.lock') }}
      - run: pip install uv
      - run: uv pip install --system -r requirements.txt torch==${{ matrix.torch }}
      - run: pytest -q --cov=src
```

</div>

<div class="level-next">
<span>Want artifact registries, automatic version bumps, & ML-specific check gates?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Promotion pipeline</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{commit} \to \text{lint/test} \to \text{train} \to \text{eval} \to \text{registry: staging} \to \text{promote: production} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Each stage has explicit criteria for the next

</li>
<li markdown="1">

Failing criteria => roll back; passing => promote

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{commit} \;\to\; \text{lint and test} \;\to\; \text{train} \;\to\; \text{evaluate} \;\to\; \text{staging registry} \;\to\; \text{production} $$</span>

**In words.** A linear pipeline of stages — the arrows `→` mean "and only proceed if the previous stage passed". Each stage has its own pass/fail criteria: lint complains about style, tests fail on broken code, training fails on bad data, evaluation fails on regression. The model lands in a *staging* registry first (where it can be inspected but isn't yet serving traffic), and only graduates to *production* after passing one more gate. Any failure breaks the chain and triggers a rollback or alert.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`lint and test`code-quality gate — style, types, unit tests

</li>
<li markdown="1">

`train`fit the model on current data

</li>
<li markdown="1">

`evaluate`measure metrics on held-out set; gate on improvement

</li>
<li markdown="1">

`staging`tagged in registry but not yet serving live traffic

</li>
<li markdown="1">

`production`actively serving requests; rollback path must exist

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Reproducible runs in CI.** Pin everything: Python, torch, CUDA, NumPy, system OS. Use a hashed lockfile. Set determinism flags. Save the run's full provenance (config, git sha, data hash) as an artefact.

**Model evaluation gates.** A model is only promoted if it beats the production baseline on held-out evals and clears subgroup-specific bars (no regressions on minority groups). Tools: model registries (MLflow, W&B), promotion workflows, model cards.

**Data drift checks in CI.** Before retraining, validate that the new data matches the schema and distribution of training data within thresholds. Great Expectations + a GH Action runs this on every data update.

**Security & supply chain.** Pip-audit or safety on every PR. Dependabot for automated security updates. SBOM generation. Pin git submodules / external models by SHA.

**Cost monitoring.** Track CI / training cost per PR. Quotas per team. Alert when a run uses 10× normal compute — usually means a bug.

**Self-hosted runners.** For GPU jobs or special hardware. Manage scaling (Karpenter, RunPod) so you only pay for time you use. Worth it when GitHub-hosted GPU pricing exceeds your dedicated-runner cost.

**Release engineering.** Semantic versioning for the package; a separate version for the trained model (commit + checkpoint id). Changelog auto-generation from conventional commits.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```yaml
name: Nightly Train + Eval
on:
  schedule: [{ cron: '0 6 * * *' }]   # daily at 06:00 UTC

jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Train tiny model on subset
        run: python -m my_project.train --config configs/ci-tiny.yaml
      - name: Eval against baseline
        run: |
          python -m my_project.eval \
            --checkpoint outputs/latest.pt \
            --baseline   gs://my-bucket/baseline.pt \
            --fail-if-worse-than 0.02       # > 2pp regression fails the job
      - name: Promote on success
        if: ${{ success() }}
        run: mlflow models register --name my-model --source outputs/latest.pt
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

[GitHub Actions Documentation <i class="fas fa-external-link-alt"></i>](https://docs.github.com/en/actions){: target="_blank" }
<span class="annotation">The reference for the most-used CI platform in open-source ML. Workflow syntax, marketplace actions, secrets handling.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Sato — Continuous Delivery for ML <i class="fas fa-external-link-alt"></i>](https://martinfowler.com/articles/cd4ml.html){: target="_blank" }
<span class="annotation">ThoughtWorks' essay on CD for ML systems. Long but the canonical reference for the data + model + code triple.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[pre-commit Framework <i class="fas fa-external-link-alt"></i>](https://pre-commit.com/){: target="_blank" }
<span class="annotation">Run linters and formatters before commits land. Saves CI minutes and reviewer time.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[DVC — Data Version Control <i class="fas fa-external-link-alt"></i>](https://dvc.org/){: target="_blank" }
<span class="annotation">Git for data and models. Pairs naturally with CI to detect data changes that should trigger retraining.</span>

</li>
</ul>

</div>

---
title: Notebook → Script — ML Resources Hub
eyebrow_text: ← Engineering · Time-Savers
eyebrow_href: {{root}}engineering.html
heading: Notebook → Script
lead: Notebooks are for exploration. Scripts are for production. The patterns for moving between them.
active_nav: engineering
prev_href: ab-testing.html
prev_title: A/B Testing
next_href: training-scaffolding.html
next_title: Training Scaffolding
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Notebooks let you explore; scripts let you ship.** Once a notebook proves out an idea, refactor it into a script: parameterised, importable, testable, runnable from the command line. Keep the notebook as the "lab report"; let the script do the work.

</div>

<article class="tldr-body" markdown="1">

**The mistake.** A 2 000-cell notebook with print statements, ad-hoc plotting, hardcoded paths, and a model checkpoint somewhere in `/tmp`. Works once on your machine; impossible to share, reproduce, or schedule.

**The pattern.** Pull every reusable function out of the notebook into a Python module. Move config into a YAML file. Wrap the "main flow" in a function with named arguments. Add a CLI. Keep the notebook for one-off analyses and to call the new module.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Notebooks shine for

- Exploratory data analysis, ad-hoc plotting
- Demos and walkthroughs
- Result inspection from a trained run
- Quick "is this idea worth pursuing?" checks

</div>

<div class="no" markdown="1">

### Scripts win for

- Anything that runs on a schedule
- Reproducible training runs (CI, retraining)
- Anything tested with pytest
- Code that needs to be imported into other code
- CLI-driven workflows

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# From notebook clutter to a clean script

# Before: notebook
# Cell 1
# import all the things
# DATA_PATH = "/Users/liv/data/v3.csv"   # hardcoded
# Cell 2
# df = pd.read_csv(DATA_PATH); df = df[df.x > 0]   # silent filtering
# Cell 3
# X = df[["a", "b"]]; y = df["t"]
# model = RandomForest().fit(X, y)
# Cell 4
# accuracy = model.score(X_test, y_test)   # train ≠ test? not clear

# After: src/my_project/train.py
import typer
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = typer.Typer()

@app.command()
def train(
    data: Path = typer.Option(...),
    out:  Path = typer.Option(...),
    seed: int  = 0,
):
    df = pd.read_csv(data)
    df = df[df.x > 0]
    X, y = df[["a", "b"]], df["t"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestClassifier(random_state=seed).fit(X_tr, y_tr)
    print(f"test acc = {model.score(X_te, y_te):.4f}")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.pkl")

if __name__ == "__main__": app()

# Now: python -m my_project.train --data data/v3.csv --out runs/exp01
```

</div>

<div class="level-next">
<span>Want module structure, & CI integration?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The "promote" gradient</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{notebook} \to \text{function in notebook} \to \text{function in module} \to \text{CLI entry-point} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Each step is a small refactor

</li>
<li markdown="1">

Do them when the code is reused, not when it's first written

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loose cells} \;\to\; \text{notebook function} \;\to\; \text{importable module} \;\to\; \text{CLI script} $$</span>

**In words.** A code-maturity ladder. The arrows (`→`) are "evolves into". Each step makes the code one notch more reusable: from a sequence of cells that only run top-to-bottom, to a callable function (still in the notebook), to that same function living in a .py module that other code can import, to a script with a proper command-line interface. Climb the ladder only when the code earns it — premature promotion is overhead with no payoff.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

Each step is a small refactor

</li>
<li markdown="1">

Do them when the code is reused, not when it's first written

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Refactor in place.** When a notebook cell becomes a thing you'd want to call twice, wrap it in a function (still in the notebook). When you'd want to call it from another notebook, move it to a module. When it needs to run unattended, give it a CLI.

**jupytext.** Pair every `.ipynb` with a `.py` percent-format file. The .py is what you commit and diff; the .ipynb is what you open in Jupyter. Solves the "notebooks don't diff well" pain.

**nbdev.** fastai's tool that treats notebooks as the source of truth for Python modules. Useful for very-notebook-centric teams; loses most "scripts and tests" benefits.

**Configuration externalised.** The script reads a YAML / Hydra config. Hyperparameters, paths, seeds — all in the config. The script just does what the config says.

**Logging instead of print.** `loguru` or `logging` with levels. Easy to silence in tests; routable to files in production. `print` belongs in notebooks, not scripts.

**Don't import the notebook.** Importing `.ipynb` files runs every cell. Move the functions to a module; import the module from the notebook.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Pair a notebook with a .py file for diffable version control
# pip install jupytext

# 1) Pair it once:
# jupytext --set-formats ipynb,py:percent notebook.ipynb

# 2) Now both files stay in sync. Commit only the .py:
# git add notebook.py
# (Add notebook.ipynb to .gitignore)

# The .py file looks like:
# # %% [markdown]
# # # My Analysis
# # %%
# import pandas as pd
# df = pd.read_csv("data.csv")
# # %%
# df.describe()
```

</div>

<div class="level-next">
<span>Want testable functions, side-effect isolation, & pure-function refactors?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Pure functions</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ f(x) = y \quad \text{with no global state, no I/O, no side effects} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Easy to test — just pass inputs, assert outputs

</li>
<li markdown="1">

Easy to reuse — the function doesn't depend on context

</li>
<li markdown="1">

Easy to parallelise — no shared state to coordinate

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{output} = f(\text{input}) \quad \text{— same input always gives same output, nothing else changes} $$</span>

**In words.** A *pure* function depends only on its arguments and only returns a value — no reading from disk, no writing logs, no mutating a global variable, no hidden state. `f(x) = y` just says "function `f` applied to input `x` gives output `y`", and the constraint is that the same `x` always gives the same `y` with no other effects on the world. This property is what makes a function trivially testable, reusable in any context, and safe to run in parallel.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

Easy to test — just pass inputs, assert outputs

</li>
<li markdown="1">

Easy to reuse — the function doesn't depend on context

</li>
<li markdown="1">

Easy to parallelise — no shared state to coordinate

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Pull I/O to the edges.** The middle of the pipeline should be pure: take a DataFrame, return a DataFrame. The edges (load, save, log) handle I/O. Easy to test the middle; the edges are mostly trivial.

**Side-effect isolation.** Each function does one thing — either a transformation or an I/O. Functions that mix both ("load and clean the data") are harder to test.

**Determinism in helper functions.** Anything that uses random numbers takes a seed or a generator. Then the same call always gives the same result.

**Type hints.** Catch bugs early — and serve as documentation. `pandas-stubs`, `numpy.typing`, `torchtyping` for tensor shapes. `mypy --strict` in CI catches a class of bugs that escape notebooks.

**Notebook smoke tests.** Run the notebook headlessly in CI (`jupyter nbconvert --execute`) to catch broken outputs. Treat notebooks like scripts: they should run end-to-end.

**Library + scripts pattern.** Put reusable code in a library (`src/my_project/`). Scripts in `scripts/` just orchestrate the library. Notebooks in `notebooks/` use the library too. Same code, three entry points.

**When NOT to refactor.** One-off analyses, one-time data prep, ad-hoc plots. Refactoring "for the future" that never comes is the most expensive waste in research engineering.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Pure transformation function — easy to test and reuse
from typing import Sequence
import numpy as np
import pandas as pd

def normalise_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Standardise columns. If stats given, use them; else fit them.
    Returns (normalised df, fitted stats)."""
    out = df.copy()
    new_stats = stats or {}
    for c in columns:
        if c not in new_stats:
            new_stats[c] = (out[c].mean(), out[c].std() + 1e-9)
        m, s = new_stats[c]
        out[c] = (out[c] - m) / s
    return out, new_stats

# Now testable
def test_normalise():
    df = pd.DataFrame({"x": [0., 1., 2., 3.]})
    n, stats = normalise_features(df, ["x"])
    assert abs(n.x.mean()) < 1e-9
    assert abs(n.x.std(ddof=1) - 1.0) < 0.1
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

[Jupytext <i class="fas fa-external-link-alt"></i>](https://jupytext.readthedocs.io/){: target="_blank" }
<span class="annotation">Round-trip notebooks ↔ .py files. Solves "notebooks don't diff" without making you give up Jupyter.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Typer <i class="fas fa-external-link-alt"></i>](https://typer.tiangolo.com/){: target="_blank" }
<span class="annotation">Modern CLI library for Python — type hints become arguments. The least-friction way to give your scripts proper command-line interfaces.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[cookiecutter-data-science <i class="fas fa-external-link-alt"></i>](https://github.com/khuyentran1401/data-science-template){: target="_blank" }
<span class="annotation">Project scaffolding for data-science / ML repos with reasonable defaults for the notebook + script split.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Drivendata — Cookiecutter Data Science <i class="fas fa-external-link-alt"></i>](https://drivendata.github.io/cookiecutter-data-science/){: target="_blank" }
<span class="annotation">The most-used data-science project template. Opinionated layout; battle-tested.</span>

</li>
</ul>

</div>

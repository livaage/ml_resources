---
title: Feature Engineering — ML Resources Hub
eyebrow_text: ← Theory · Data &amp; Features
eyebrow_href: ../theory.html
heading: Feature Engineering
lead: The art of giving the model the inputs it needs to succeed — domain knowledge, encoded.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The hardest problem is often the easiest one in the right space.** Linear models can't separate concentric circles in *(x, y)* — but they can in *(x², y², xy)*. Decision trees can't see the day-of-week pattern unless you extract a "day" feature. Most of classical ML's success is feature engineering pretending to be modelling.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Toggle feature transforms — watch a non-linearly-separable problem become trivially linear</span>
</div>
<div class="viz-classic-controls">
<button id="viz-fe-raw" type="button" class="active">Raw (x, y)</button>
<button id="viz-fe-poly" type="button">Polynomial features</button>
<button id="viz-fe-rbf" type="button">RBF features</button>
<button id="viz-fe-radial" type="button">r = √(x² + y²)</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-fe-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-fe-caption"></div>
</div>

<script src="{{root}}js/viz/feature-engineering.js"></script>

The same concentric-rings dataset. Raw *(x, y)* coordinates: no straight line separates the classes. Add *x²* and *y²* features and a hyperplane in the higher-dimensional space cuts cleanly. The simplest possible engineered feature — *r = √(x² + y²)* — makes the problem one-dimensional. The model didn't change; the inputs did.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Domain features.** The highest-value features encode knowledge the model can't figure out by itself: hour-of-day, day-of-week, distance to a known landmark, account-age-in-days. Trees and linear models both benefit; even deep models often work much better with a handful of well-chosen engineered features.

**Polynomial features.** Replace *(x, y)* with *(x, y, x², xy, y²)*. A linear model in this space is a quadratic in the original — captures interactions automatically. Combinatorial explosion with many features though, so combine with regularization or restrict the degree.

**Interaction features.** Multiply or combine pairs (or triples) of features. Captures "this matters more when that is high". Essential for many tabular problems where a single feature is uninformative but the pair is predictive.

**Binning & discretisation.** Convert continuous features into categories. Makes non-linear thresholds trivial for linear models; can hurt with trees, which already discover thresholds. Useful with target encoding (mean target per bin).

**Time features.** Hour, day, month, weekend, holiday, time-since-event, rolling means. Most of forecasting is feature engineering on the time axis.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Worth investing in

- Tabular data with strong domain context (finance, healthcare, supply chain)
- Small datasets — every well-chosen feature is worth dozens of training examples
- Linear models or shallow trees — they can't synthesise features themselves
- Forecasting — time features dominate

</div>

<div class="no" markdown="1">

### Less critical when

- You have huge data and a deep network can synthesise its own features
- The input is already a useful representation (image pixels, embeddings)
- You're using gradient boosting — handles interactions well already
- Domain knowledge is unavailable or untrustworthy

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
import pandas as pd

# Polynomial features — degree 2 with interactions
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)         # adds x², xy, y², x³, ... features

# Quantile binning + one-hot — useful with linear models
disc = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile")
X_bin = disc.fit_transform(X[["income"]])

# Time features from a datetime column
df["hour"]    = df["ts"].dt.hour
df["dow"]     = df["ts"].dt.dayofweek
df["weekend"] = df["dow"].isin([5, 6]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)   # periodic encoding
```

</div>

<div class="level-next">
<span>Want the kernel trick, learned embeddings, & feature selection?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The kernel trick</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ K(x, x') = \langle \phi(x), \phi(x') \rangle $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`φ`implicit feature map into a high-dimensional space

</li>
<li markdown="1">

Working with *K* directly avoids computing *φ* — feature engineering for free, if *K* is well-chosen

</li>
<li markdown="1">

SVMs, GPs, kernel ridge regression all exploit this

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{similarity}(x, x') \;=\; \text{dot product of transformed } x \text{ and transformed } x' $$</span>

**In words.** A "kernel" *K* is just a similarity score between two data points — it tells you how related *x* and *x'* are. The trick is that this score is mathematically equivalent to first transforming both points through some feature map `φ` (phi) into a richer space, then taking their dot product there. The angle brackets `⟨·, ·⟩` just mean "dot product". Because many classical algorithms only ever need similarities between points (never the points themselves), you can swap in any sensible kernel and effectively work in that richer feature space without ever computing the transformed vectors.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`K(x, x')`kernel — a similarity score between two data points

</li>
<li markdown="1">

`φ`feature map — sends each point into a (possibly much higher-dim) space

</li>
<li markdown="1">

`⟨·, ·⟩`dot product in that richer space

</li>
<li markdown="1">

Common kernels: RBF (Gaussian), polynomial, string, graph

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The kernel trick.** Many classical algorithms (SVM, ridge regression, PCA) only need inner products between data points — never the individual vectors. Replace the inner product with a kernel *K(x, x')* and you've implicitly engineered features in a (possibly infinite-dimensional) space. The RBF kernel is the most common; polynomial and string kernels are useful for structured data.

**Learned embeddings.** Deep networks learn features. The first layers of a CNN find edges; the middle layers find textures; the last find object parts. Modern foundation models (CLIP, ViT, BERT) are reusable feature extractors — embed your data once with their backbone and run classical models on top.

**Feature selection.** Filter (correlation, mutual information), wrapper (forward / backward selection), embedded (LASSO, tree importances), and permutation-importance methods. Useful when you have hundreds of candidate features and need to pick the few that matter.

**Target encoding properly.** Replace categories with the mean target value, but do it inside CV folds with smoothing. Without those, target encoding leaks target information and overstates training performance dramatically.

**Periodic features.** Hour-of-day shouldn't be encoded as 0–23 (23 isn't "far from" 0). Use *sin(2πh/24), cos(2πh/24)* — wraps continuously. Same for day-of-year, day-of-week.

**Aggregations and joins.** For tabular problems with related tables (transactions per user, clicks per ad), the right features are usually aggregates: count, mean, max, last-N-day rolling sum. SQL is feature engineering at scale.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Mutual information feature selection — works with non-linear relationships
mi = SelectKBest(mutual_info_classif, k=20)
X_top = mi.fit_transform(X, y)
print(X.columns[mi.get_support()])     # which features were kept

# Aggregations per user (e.g. for fraud detection)
agg = (df.groupby("user_id")["amount"]
         .agg(["count", "mean", "std", "max"])
         .add_suffix("_amount"))
df = df.merge(agg, on="user_id")
```

</div>

<div class="level-next">
<span>Want auto-feature-engineering, deep feature synthesis, and AutoML?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Deep Feature Synthesis</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{F} = \{\,g \circ \text{Aggr}(R, f) : f \in \mathcal{F}_{\text{prim}}, R \in \mathcal{R},\, g \in \mathcal{G}\,\} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Stack primitive transforms (count, mean, max, …) over relationships (R) to enumerate features automatically

</li>
<li markdown="1">

Featuretools' algorithm — replaces a lot of hand feature-engineering for relational data

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{generated features} \;=\; \text{transform}(\text{aggregation across relationship}(\text{primitive feature})) $$</span>

**In words.** Deep Feature Synthesis enumerates new features by composing simple operations. Start with a primitive feature (a column you already have), pick a relationship in your schema (e.g. "transactions belong to a user"), apply an aggregation across that relationship (count, mean, max, sum), then optionally transform the result (log, hour-of-day). The script symbol `F` is the resulting set of all such generated features. The `∘` means "compose" — apply one after the other. Stacking depth lets you build features like "average transaction amount per user, in the last 7 days".
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`primitive feature`a raw column in one of your tables

</li>
<li markdown="1">

`relationship`a parent–child link in the schema (e.g. user → transactions)

</li>
<li markdown="1">

`aggregation`combine many child rows into one summary (count, mean, max, …)

</li>
<li markdown="1">

`transform`a single-input function (log, day-of-week, …)

</li>
<li markdown="1">

Stacked recipes — depth 2 or 3 — automate most of what hand feature-engineering produces

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Featuretools and Deep Feature Synthesis.** Algorithmic feature engineering for relational data. Given the schema of your tables, DFS automatically constructs features by stacking aggregations and transformations across relationships. Useful for fast prototyping, especially when you don't yet know which features matter.

**AutoML feature pipelines.** AutoGluon, H2O, TPOT, and others bundle preprocessing + feature engineering + model selection into a single search. Saves time; the resulting pipelines are often opaque but competitive on tabular benchmarks.

**tsfresh and time-series feature libraries.** Hundreds of statistical features automatically extracted from each time series — autocorrelation, spectral entropy, change points. Combined with feature selection, often beats hand-tuned features.

**Pretrained embeddings as features.** Take a column of free text and run it through a sentence transformer. The 768-dim vector can be concatenated with your tabular features and dropped into a downstream model. Same for product descriptions, URLs, addresses.

**Causal features.** When the goal is to estimate a treatment effect, the features that matter are different — they should make the treatment *exogenous* conditional on them. See the causal inference literature for the right framework (DAGs, propensity scores, doubly-robust estimation).

**Adversarial feature engineering.** Feature-importance reveals what the model depends on — and what an adversary might attack. Robust features (less manipulable) sometimes deserve preference over accurate ones.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import featuretools as ft

# DFS over a relational dataset (users + transactions)
es = ft.EntitySet("payments")
es.add_dataframe(dataframe=users,    dataframe_name="users",    index="user_id")
es.add_dataframe(dataframe=transactions, dataframe_name="txns", index="txn_id",
                 time_index="ts")
es.add_relationship(parent_dataframe_name="users",
                    parent_column_name="user_id",
                    child_dataframe_name="txns",
                    child_column_name="user_id")

features, defs = ft.dfs(entityset=es, target_dataframe_name="users",
                        agg_primitives=["count", "mean", "max", "trend"],
                        trans_primitives=["hour", "day", "month"],
                        max_depth=2)
# features: a pandas DataFrame with auto-generated columns like
#   MEAN(txns.amount), MAX(txns.amount), TREND(txns.amount, ts), …
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
<li data-tier="indepth" markdown="1">

[Featuretools — Deep Feature Synthesis <i class="fas fa-external-link-alt"></i>](https://www.featuretools.com/){: target="_blank" }
<span class="annotation">The reference automated-feature-engineering library for relational data. Docs walk through DFS with worked examples.</span>

</li>
<li data-tier="intuition" markdown="1">

[Kaggle — Feature Engineering Mini-Course <i class="fas fa-external-link-alt"></i>](https://www.kaggle.com/learn/feature-engineering){: target="_blank" }
<span class="annotation">Six short interactive lessons covering target encoding, interaction terms, time features, and feature selection in pandas.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Hyndman & Athanasopoulos — Forecasting: Principles and Practice <i class="fas fa-external-link-alt"></i>](https://otexts.com/fpp3/){: target="_blank" }
<span class="annotation">Best reference for time-series feature engineering — lags, rolling stats, seasonality, calendar features.</span>

</li>
<li data-tier="indepth" markdown="1">

[tsfresh <i class="fas fa-external-link-alt"></i>](https://tsfresh.readthedocs.io/){: target="_blank" }
<span class="annotation">Hundreds of pre-built features for time-series data, with built-in relevance filtering. Useful starting point for any sequence problem.</span>

</li>
</ul>

</div>

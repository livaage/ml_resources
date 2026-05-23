---
title: Data Preprocessing — ML Resources Hub
eyebrow_text: ← Theory · Data &amp; Features
eyebrow_href: ../theory.html
heading: Data Preprocessing
lead: Scaling, encoding, normalising — making the data match what the model can actually digest.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Most models can't read raw data.** Linear models break when features are on wildly different scales; trees don't mind. Distance-based models (k-NN, k-means) are dominated by whichever feature has the biggest numbers. Neural nets train faster with normalised inputs. Encoding categories, handling missing values, normalising scales — that's preprocessing.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Toggle the scaler — watch the boundary go from useless to correct as the features get put on the same footing</span>
</div>
<div class="viz-classic-controls">
<button id="viz-pre-raw" type="button" class="active">Raw (mixed scales)</button>
<button id="viz-pre-std" type="button">Standardised</button>
<button id="viz-pre-minmax" type="button">Min-Max</button>
<button id="viz-pre-robust" type="button">Robust (median/IQR)</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-pre-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-pre-caption"></div>
</div>

<script src="{{root}}js/viz/preprocessing.js"></script>

Same dataset, three transforms. On the raw data, feature 1 ranges over [0, 1000] while feature 2 is in [0, 1]. A k-NN classifier ignores feature 2 entirely — every neighbour decision is dominated by feature 1. After **standardisation** (subtract mean, divide by std), both features get equal say. **Min-max** scales to [0, 1]; **robust** uses median and IQR so outliers don't blow up the scale.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Standardisation** (z-score): subtract the mean, divide by the standard deviation. Each feature has mean 0, std 1. Default for most linear models and neural networks.

**Min-max scaling**: linearly squash to [0, 1] (or [-1, 1]). Sensitive to outliers; useful for image pixels or where bounded input is required.

**Robust scaling**: subtract the median, divide by the IQR. Doesn't care about outliers. Reach for it when the distribution has heavy tails or known anomalies.

**Categorical encoding**: one-hot for ordinal-free categories, ordinal for ordered ones, target encoding (mean target value per category) for high-cardinality categories — used carefully to avoid leakage.

**Log / Box-Cox transforms**: when the target or feature is skewed (prices, counts, durations), a log transform often makes it Gaussian-ish and helps any model.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Must preprocess for

- k-NN, k-means, SVM with non-tree kernels — distance-based
- Linear and logistic regression with regularization
- Neural networks — converges faster, more stably
- PCA / dimensionality reduction

</div>

<div class="no" markdown="1">

### Can usually skip for

- Decision trees, random forests, gradient boosting
- Naive Bayes (depending on the variant)
- Models with built-in normalisation (BatchNorm, LayerNorm)
- When all features are already on the same scale (pixels)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, PowerTransformer,
)
from sklearn.pipeline   import Pipeline
from sklearn.compose    import ColumnTransformer

# A real preprocessing pipeline — fit on TRAIN ONLY
numerical_cols   = ["age", "income", "balance"]
categorical_cols = ["country", "occupation"]

prep = ColumnTransformer([
    ("num", StandardScaler(),              numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

pipe = Pipeline([("prep", prep), ("model", LogisticRegression())])
pipe.fit(X_train, y_train)              # scaler fitted on TRAIN
pipe.predict(X_test)                    # scaler applied to TEST
```

</div>

<div class="level-next">
<span>Want target encoding, robust scalers, and quantile transforms?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Why scaling matters for optimization</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \nabla_w \mathcal{L} = X^\top (Xw - y), \quad \|\nabla_w \mathcal{L}\| \propto \sigma_{\max}(X) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Gradient magnitude scales with feature magnitudes

</li>
<li markdown="1">

Features on very different scales ⇒ ill-conditioned gradient ⇒ slow / unstable convergence

</li>
<li markdown="1">

Standardising puts all features at *σ ≈ 1*, dramatically improving conditioning

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{slope of loss} \;=\; \text{(features)}^\top \times \text{(residuals)}, \quad \text{size of slope} \;\propto\; \text{largest feature scale} $$</span>

**In words.** The gradient (slope of the loss with respect to the weights) is just the feature matrix times the residuals — so its size grows with whatever the largest feature scale is. `σ_max` (sigma-max) is the biggest "stretch" of the feature matrix; it dominates the step size each weight gets. When one feature is in the thousands and another in the tenths, the optimiser is essentially taking giant steps for the big-scale weight and tiny steps for the small-scale one — slow, unstable, or both. Standardising every feature to roughly the same scale fixes the imbalance.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`slope of loss`gradient with respect to the weights

</li>
<li markdown="1">

`features`the design matrix *X*

</li>
<li markdown="1">

`residuals`predictions minus targets (*Xw − y*)

</li>
<li markdown="1">

`largest feature scale`biggest singular value of *X* — dominates step size

</li>
<li markdown="1">

Standardising → every feature near scale 1 → balanced step sizes → fast convergence

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Target encoding for high-cardinality categories.** Replace each category with its mean target value (e.g. mean of *y* per ZIP code). Captures the signal in a single feature — but if you compute it on the whole dataset before splitting, that's data leakage. Always compute target encodings *within* CV folds.

**Quantile transforms.** Map each feature to its rank, then optionally to a Gaussian. Robust to outliers, makes the feature uniform or normal. Useful for skewed distributions or when downstream models assume Gaussian inputs.

**Power transforms (Box-Cox, Yeo-Johnson).** Parametric family of transforms that find the closest-to-Gaussian shape. Yeo-Johnson handles negative values; Box-Cox needs positives. Stabilises variance — important for regression with heteroscedastic noise.

**Encoders for ordered categories.** Ordinal encoding when there's a natural order ("low" < "medium" < "high"). One-hot when there isn't. Trees handle ordinal encoding fine; linear models often need one-hot to avoid imposing a false ordering.

**Sparse handling.** One-hot encoding with thousands of categories blows up memory. Sparse matrices, hashing tricks, learnable embeddings, and feature crosses are all common alternatives.

**Preprocess inside the CV loop.** The single most common preprocessing mistake is fitting the scaler / encoder on the full dataset before splitting. The validation fold's mean / std then leaks into the training fold's "scaled" features. Pipelines exist to enforce the right order.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np, pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from category_encoders import TargetEncoder

# Box-Cox / Yeo-Johnson for skewed features
pt = PowerTransformer(method="yeo-johnson")
X_pt = pt.fit_transform(X_train_numeric)

# Rank-Gaussian transform for very skewed data
qt = QuantileTransformer(output_distribution="normal")
X_qt = qt.fit_transform(X_train_numeric)

# Target encoding with smoothing — high-cardinality categories
te = TargetEncoder(smoothing=10.0)
X_te = te.fit_transform(X_train[["zip"]], y_train)
# Fit on train only; transform on test
```

</div>

<div class="level-next">
<span>Want feature stores, online preprocessing, and lookup tables?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Pre-processing as a learned function</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathbf{x}_{\text{enc}} = f_\phi(\mathbf{x}_{\text{raw}}) \quad \text{(differentiable, jointly trained)} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Embeddings, BatchNorm, LayerNorm — preprocessing as part of the model

</li>
<li markdown="1">

Updated jointly with the rest of the parameters

</li>
<li markdown="1">

Generalises classical scalers — but harder to enforce no-leakage by construction

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{encoded input} \;=\; \text{learned transform}(\text{raw input}) $$</span>

**In words.** Instead of hand-picking a scaler before training, treat the preprocessing step as part of the model itself. `f_φ` is just "a function with learnable parameters *φ*" — an embedding lookup, a batch-norm layer, a small MLP. Because it's differentiable and trained jointly with the rest of the network, its parameters tune themselves to make the downstream task work better. The catch: the line between "preprocessing" and "model" blurs, so you have to be careful nothing about the validation set leaks into the encoder's parameters.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`encoded input`the cleaned-up version that the rest of the model sees

</li>
<li markdown="1">

`raw input`the original feature vector

</li>
<li markdown="1">

`learned transform`any differentiable function with parameters (*φ*) trained jointly with the model

</li>
<li markdown="1">

Examples: learned embeddings, batch / layer norm, learned tokenisers

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Feature stores.** Production ML systems share preprocessed features across training and online inference. A feature store (Feast, Tecton, Vertex, etc.) is essentially a database of precomputed features keyed by entity ID, with versioning and time-travel queries to prevent train/serve skew.

**Train / serve skew.** The classic deployment bug: training pipeline computes feature X one way, the online serving system computes it differently. Hard to detect until production data hits. Mitigation: share the same code, or enforce schema and statistic checks at both endpoints.

**BatchNorm and friends as preprocessing.** BatchNorm normalises activations by their mini-batch statistics — essentially "scale, but inside the model". LayerNorm and RMSNorm do similar things along different axes. The interaction with explicit weight decay is subtle (see Regularization).

**Embedding tables for categories.** Instead of one-hot, learn a dense vector per category. Reduces dimensionality and captures similarity automatically. Standard for high-cardinality categorical inputs in deep learning (users, items, tokens).

**Streaming statistics.** When data is streamed and you can't fit it all in memory, use online formulas (Welford's algorithm) to compute mean and variance incrementally. Same idea behind `StandardScaler.partial_fit` in scikit-learn.

**Preprocessing for fairness.** Some preprocessing is itself a fairness intervention — re-weighting examples, removing protected attributes, learning fair representations. See the Fairness page for details.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn

# Embedding table — learnable preprocessing for high-cardinality categories
class CategoricalEmbed(nn.Module):
    def __init__(self, n_categories, d=8):
        super().__init__()
        self.emb = nn.Embedding(n_categories, d)
    def forward(self, x):           # x: (B,) long tensor of category indices
        return self.emb(x)          # (B, d) dense vector per category

# Online mean/std using Welford's algorithm
class RunningStats:
    def __init__(self):
        self.n, self.mean, self.M2 = 0, 0.0, 0.0
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    @property
    def std(self):
        return (self.M2 / max(1, self.n - 1)) ** 0.5
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

[scikit-learn — Preprocessing Guide <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/preprocessing.html){: target="_blank" }
<span class="annotation">Practical reference for every scaler / encoder / transformer with worked examples and code.</span>

</li>
<li data-tier="indepth" markdown="1">

[Feast — Open-source Feature Store <i class="fas fa-external-link-alt"></i>](https://www.feast.dev/){: target="_blank" }
<span class="annotation">Reference implementation for a production feature store. Docs walk through training / serving consistency for tabular ML.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[category_encoders <i class="fas fa-external-link-alt"></i>](https://contrib.scikit-learn.org/category_encoders/){: target="_blank" }
<span class="annotation">Library of categorical encoders beyond what sklearn ships — target, hash, James-Stein, GLMM and more. Side-by-side benchmarks.</span>

</li>
<li data-tier="intuition" markdown="1">

[Kaggle — Feature Engineering Wisdom <i class="fas fa-external-link-alt"></i>](https://www.kaggle.com/competitions/feature-engineering-tips){: target="_blank" }
<span class="annotation">Community-collected practical tips that often beat fancy models. Lots of preprocessing tricks here.</span>

</li>
</ul>

</div>

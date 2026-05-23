---
title: Splits &amp; Cross-Validation — ML Resources Hub
eyebrow_text: ← Theory · Data &amp; Features
eyebrow_href: ../theory.html
heading: Splits &amp; Cross-Validation
lead: Train / val / test — the discipline of evaluating on data your model hasn't seen.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**You can't measure generalization on data you trained on.** Split the dataset into three roles: *train* (fit the model), *validation* (pick hyperparameters and compare models), and *test* (one number at the end). Touching the test set in any way during model development invalidates it.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">See how different splitting strategies divide your dataset — and which ones leak when the data isn't iid</span>
</div>
<div class="viz-classic-controls">
<button id="viz-ds-random" type="button" class="active">Random</button>
<button id="viz-ds-strat" type="button">Stratified</button>
<button id="viz-ds-time" type="button">Time-based</button>
<button id="viz-ds-group" type="button">Group</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-ds-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-ds-caption"></div>
</div>

<script src="{{root}}js/viz/data-splitting.js"></script>

Above: 60 examples coloured by class (imbalanced 80/20) and grouped (every 3 share a group ID). **Random** splitting can put almost no class-1 examples in val; **stratified** preserves the class ratio. **Time-based** always puts the future in val (no peeking). **Group** keeps all members of a group on one side — required when independence assumptions break (patient ID, household ID, session ID).
{: .viz-intro }

<article class="tldr-body" markdown="1">

**The three roles.** Train, validation, test. Train fits. Validation chooses. Test reports. The test set is touched *once*, at the end. Multiple test set evaluations during development invalidate the test as an estimate of true performance.

**Random split.** Default for iid data. Typically 60/20/20 or 70/15/15. Doesn't preserve class balance — use stratified if classes are imbalanced.

**Stratified split.** Same class proportions in each split. Critical for imbalanced classification; standard for any classification task with k > 2 classes.

**Time-based split.** Train on the past, validate / test on the future. The only correct split for sequential data. Random splits leak the future into training and overstate performance.

**Group split.** When examples are not independent — multiple records per patient, per user, per session — keep all examples from a group on one side of the split. Otherwise you measure within-group memorisation, not generalization.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Pick stratified / group / time when

- Classes are imbalanced → stratified
- Multiple examples per entity → group
- Sequential / time-series → time-based
- Geographic / spatial — try spatial blocks

</div>

<div class="no" markdown="1">

### Pitfalls to avoid

- Random splitting time-series — leaks the future
- Random splitting grouped data — leaks the group
- Refitting preprocessing on val/test — leaks statistics
- Touching the test set during model selection

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.model_selection import (
    train_test_split, StratifiedShuffleSplit,
    GroupShuffleSplit, TimeSeriesSplit,
)

# Standard 60/20/20 with stratification
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=1, stratify=y_dev)

# Group split — never put two examples from the same patient in different folds
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups))

# Time series — always train on the past
tscv = TimeSeriesSplit(n_splits=5)
for tr, val in tscv.split(X):
    fit(X[tr], y[tr]); evaluate(X[val], y[val])
```

</div>

<div class="level-next">
<span>Want nested CV, distribution shift checks, and probing splits?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Generalisation error decomposition</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{err}_{\text{test}} = \text{err}_{\text{train}} + \underbrace{\text{generalization gap}}_{\text{model-dependent}} + \underbrace{\text{distribution shift}}_{\text{often the killer}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

CV estimates the generalization gap on data drawn from the same distribution

</li>
<li markdown="1">

Distribution shift is what kills production models — your CV told you nothing about it

</li>
<li markdown="1">

Always reserve an out-of-distribution test set if you can

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{test error} \;=\; \text{training error} \;+\; \text{generalization gap} \;+\; \text{distribution shift} $$</span>

**In words.** The error you'll see at deployment breaks into three pieces. The first is just how well you fit the training data. The second, the *generalization gap*, is the extra error you pay because your model has memorised quirks of the training set — this is what cross-validation estimates. The third, *distribution shift*, is extra error because the real world doesn't look quite like your dataset (different time period, different region, different users). The third piece is invisible to CV and is usually what hurts you most in production.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`training error`how badly the model does on the data it learned from

</li>
<li markdown="1">

`generalization gap`extra error from fitting noise / quirks of the training set

</li>
<li markdown="1">

`distribution shift`extra error because the deployment data differs from the training data

</li>
<li markdown="1">

CV estimates the generalization gap; distribution shift requires a separate OOD test

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Nested CV.** Outer loop estimates generalization; inner loop chooses hyperparameters. Without it, tuning on the same folds you report from gives optimistic results. Expensive — typical setup is 5 outer × 3 inner — but the right thing for small datasets.

**Holdout vs CV.** For large datasets a single 80/20 split is fine and much cheaper. CV pays off when data is small (the variance of any single split is too high). Rule of thumb: CV below ~50k examples, single holdout above.

**Distribution-shift tests.** A common pattern: train on data from 2020–2023, test on data from 2024. Reveals whether your model relies on features that change over time. Same for geographic shift (train on Europe, test on Asia), demographic shift, etc.

**Probe / adversarial splits.** Construct a test set deliberately different from the training distribution — long-tail classes, distribution shift, adversarial perturbations. Forces you to find weaknesses before deployment does.

**The split-it-back-together trick.** When your dataset is too small for both validation and test, use nested CV — the outer fold's test is the inner fold's "held-out" validation; you cycle through. Doubles your effective data with care.

**Repeated random splits.** Report mean ± std over multiple random splits. Surprisingly informative on small data: a difference of 1 std isn't a real improvement.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.model_selection import cross_val_score, GroupKFold
import numpy as np

# Repeated stratified CV with confidence interval
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(f"{scores.mean():.3f} ± {scores.std():.3f}  n={len(scores)}")

# Group K-Fold — keep all examples from a group on one side
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(X, y, groups):
    # Train on tr, evaluate on te
    pass

# Distribution-shift split: use a different time period as test
mask_train = df["year"] <= 2023
mask_test  = df["year"] == 2024
```

</div>

<div class="level-next">
<span>Want bootstrap, conformal prediction, and OOD detection?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Bootstrap estimator</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \hat{\theta}_\text{boot} = \frac{1}{B} \sum_{b=1}^{B} \hat{\theta}\!\big(D_b^*\big), \quad D_b^* \sim D \text{ with replacement} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Resample *B* times to estimate the distribution of any statistic

</li>
<li markdown="1">

Out-of-bag examples (~37% per draw) act as a held-out validation set for free

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{bootstrap estimate} \;=\; \text{average across } B \text{ resamples of the statistic computed on each resample} $$</span>

**In words.** The bootstrap pretends your dataset is the population. Draw `B` new datasets of the same size by sampling *with replacement* from the original — meaning some examples appear multiple times and others not at all. Compute your statistic (mean, AUC, anything) on each of those `B` resamples. The spread of those `B` values approximates the sampling distribution of the statistic, giving you confidence intervals essentially for free. Around 37% of the original points get left out of any given resample — those "out-of-bag" examples can serve as a held-out test set.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`B`number of bootstrap resamples (typically 200–1000)

</li>
<li markdown="1">

`resample`a dataset of the same size drawn with replacement from the original

</li>
<li markdown="1">

`statistic`any quantity computed from a dataset (mean, accuracy, AUC, etc.)

</li>
<li markdown="1">

Out-of-bag examples (~37% per draw) act as a held-out validation set for free

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Bootstrap vs CV.** Bootstrap resamples with replacement; CV partitions. Bootstrap gives confidence intervals on metrics; CV gives a point estimate. Use both — bootstrap to report uncertainty, CV to pick the best model. Out-of-bag estimation in random forests uses the bootstrap by construction.

**Conformal prediction.** Use a held-out calibration fold to construct prediction intervals with finite-sample coverage guarantees. Distribution-free, model-agnostic. Increasingly the right answer for production deployment of regression and classification.

**OOD detection.** Many production systems combine an in-distribution model with an OOD detector that flags inputs the model hasn't seen anything like before. Methods include Mahalanobis distance in feature space, Energy-based scores, Outlier exposure, conformal anomaly detection.

**The "validation set rot" trap.** Repeatedly tuning on the same validation set selects for performance on that set specifically — eventually your "best" model is overfit to validation. Hold out a final test set, rotate validation sets, or use nested CV.

**Adaptive data analysis.** Dwork et al. (2015) showed that interactive data analysis can produce arbitrarily inflated performance estimates if you're not careful — even with held-out validation. Differential-privacy-based mechanisms (thresholdout, gauss-out) give bounds; in practice, treat each new analysis as eating into a budget.

**Time-series with non-stationarity.** The classical TimeSeriesSplit assumes a fixed underlying process. When the world genuinely changes, error grows with the gap between train and test ends. Walk-forward validation with re-training, exponentially-weighted features, and model monitoring all matter.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.utils import resample

# Bootstrap a metric with confidence interval
def bootstrap_ci(y_true, y_pred, metric, B=1000, alpha=0.05):
    rng = np.random.default_rng(0)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        vals.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return np.mean(vals), lo, hi

# Walk-forward CV for non-stationary time series
def walk_forward(X, y, n_splits=5, min_train=500):
    n = len(X); step = (n - min_train) // n_splits
    for i in range(n_splits):
        tr_end = min_train + i * step
        yield slice(0, tr_end), slice(tr_end, tr_end + step)
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

[scikit-learn — Cross-Validation <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/cross_validation.html){: target="_blank" }
<span class="annotation">Practical guide with diagrams of every split type. Best place to look up which API to use for which kind of data.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Timbers et al. — Data Science: A First Introduction <i class="fas fa-external-link-alt"></i>](https://datasciencebook.ca/){: target="_blank" }
<span class="annotation">Free textbook with an excellent intuitive chapter on train/val/test discipline and the pitfalls of touching the test set.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Hyndman & Athanasopoulos — Time-Series CV <i class="fas fa-external-link-alt"></i>](https://otexts.com/fpp3/tscv.html){: target="_blank" }
<span class="annotation">The canonical reference for sequential data evaluation. Walk-forward, expanding window, blocked CV.</span>

</li>
<li data-tier="indepth" markdown="1">

[Angelopoulos & Bates — Conformal Prediction Tutorial <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2107.07511){: target="_blank" }
<span class="annotation">Distribution-free prediction intervals with finite-sample coverage. Increasingly the answer for production-grade uncertainty.</span>

</li>
</ul>

</div>

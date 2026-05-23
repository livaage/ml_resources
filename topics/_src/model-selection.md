---
title: Model Selection &amp; Cross-Validation — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Model Selection &amp; Cross-Validation
lead: Choosing between models — and the hyperparameters that come with them — without lying to yourself about how good they are.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Pick the model on data the model has never seen.** If you use the training set, you'll prefer the most overfit model. If you use the test set, you've burnt it by the time you ship. The middle ground is *validation* — and the most reliable way to validate is *k-fold cross-validation*: split the data into k chunks, train k times leaving one chunk out, average the scores.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide K to see how cross-validation splits your data — train on the indigo, validate on the orange, average over all folds</span>
</div>
<div class="viz-classic-controls">
<button id="viz-cv-kfold" type="button" class="active">K-Fold</button>
<button id="viz-cv-strat" type="button">Stratified</button>
<button id="viz-cv-tss" type="button">Time-Series</button>
<button id="viz-cv-loocv" type="button">LOO</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                K
                <input id="viz-cv-k" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-cv-k-lbl">K = 5</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-cv-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-cv-caption"></div>
</div>

<script src="{{root}}js/viz/cross-validation.js"></script>

Each row is one of K training runs. The indigo cells are the training set for that fold; the orange cells are the validation set. Slide K up and each fold trains on more, validates on less — but you run more folds. **Stratified** preserves class proportions in each split (essential for imbalanced data). **Time-series** always validates on the future of the train set (avoiding time leakage). **LOO** sets K = N (one example per fold).
{: .viz-intro }

<article class="tldr-body" markdown="1">

**The three-way split.** Train / validate / test. Train the model. Use validation to choose hyperparameters and compare models. Use test *once* at the very end to report performance. Touching the test set during model selection invalidates it.

**K-fold CV.** When data is precious, average performance over many train/validate splits. K = 5 is the default; K = 10 for smaller datasets. Each example serves as validation exactly once. Cross-validation gives both a mean score and a variance — quote both.

**Stratified k-fold.** Make sure each fold has the same class distribution as the whole dataset. Essential when classes are imbalanced — without it, you can get a fold with zero positives.

**Time-series CV.** For sequential data, validation must come *after* training in time. Never randomly split — the model would peek at the future.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Use K-fold when

- Data is iid (independent and identically distributed)
- Dataset is moderate (a few hundred to a few hundred thousand examples)
- You need a robust score with variance estimate
- Classes are imbalanced → stratified k-fold

</div>

<div class="no" markdown="1">

### Use something else when

- Data is sequential (time series, sessions) → time-series CV
- Data is grouped (multiple samples per patient) → group k-fold
- Dataset is huge → a single big validation set is fine and much cheaper
- Dataset is tiny → LOO is more reliable than 5-fold

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, TimeSeriesSplit, GroupKFold, KFold,
)

# Default 5-fold (regression or balanced classification)
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")

# Stratified — classification with imbalanced classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Time-series — validation always comes after train
cv = TimeSeriesSplit(n_splits=5)

# Grouped — keep all samples from the same group in the same fold
cv = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=groups, cv=cv)
```

</div>

<div class="level-next">
<span>Want nested CV, bootstrap, and hyperparameter search strategies?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">CV estimate of generalization error</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \widehat{\text{err}}_{\text{CV}} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|V_k|} \sum_{i \in V_k} \ell(y_i,\, \hat f^{(-k)}(x_i)) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Vk`validation fold *k*

</li>
<li markdown="1">

`f(−k)`model trained on everything *but* fold *k*

</li>
<li markdown="1">

Average loss over all examples, each evaluated by a model that didn't see it

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{CV error} \;=\; \text{average across folds of (average loss on the held-out fold)} $$</span>

**In words.** You split the data into `K` equal-sized chunks (folds). For each fold *k*, you train a fresh model on the other *K − 1* folds, then compute the average loss `ℓ` across all examples in fold *k*. That gives you `K` per-fold scores. Average those, and you have your CV estimate — which approximates how well the model will do on data it hasn't seen. Critically, every example serves as a "test" example exactly once, so nothing is wasted.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`K`number of folds (typically 5 or 10)

</li>
<li markdown="1">

`fold k`the *k*-th held-out chunk used for validation

</li>
<li markdown="1">

`loss ℓ`how wrong each prediction is (squared error, cross-entropy, etc.)

</li>
<li markdown="1">

Each example is evaluated by a model that didn't see it during training

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Nested CV.** Outer loop estimates generalization; inner loop chooses hyperparameters. Without this, hyperparameter tuning on the same CV folds you report from gives you an optimistic estimate (you've selected for performance on those folds). Nested CV is computationally expensive — typical pattern is 5 outer × 3 inner.

**Hyperparameter search strategies.** Grid search: cheap if you have ≤ 3 hyperparameters and discrete values. Random search: better in high dimensions (Bergstra & Bengio 2012 showed it's strictly better). Bayesian optimization (e.g. Optuna, scikit-optimize): build a probabilistic model of the search space and propose informative trials. Hyperband / ASHA: aggressively prune unpromising trials.

**One-standard-error rule.** Among models within one standard error of the best CV score, pick the *simplest*. Avoids the small-data trap of picking a wiggly model that happened to win by chance.

**Bootstrap as an alternative.** Instead of folds, resample with replacement to make *B* training sets the same size as the original; the out-of-bag samples (~37%) act as validation. Useful for estimating prediction *intervals* as well as point performance.

**Data leakage.** Anything you do to your features *using information from the whole dataset* leaks. Standardise features inside each CV fold, not before splitting. Same for target encoding, imputation, feature selection. `sklearn.pipeline` exists specifically to enforce this.

**What CV does and doesn't tell you.** CV estimates how well a *model-selection procedure* generalises, on data drawn from the same distribution as your dataset. It tells you nothing about distribution shift, deployment-time data drift, or label noise.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model   import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Pipeline = pre-processing + model. Pipeline forces scaler to be re-fit
# *inside* each CV fold — no leakage.
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf",   LogisticRegression(max_iter=1000)),
])

# Nested CV — outer estimates generalization, inner picks C
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
grid  = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=inner)

scores = []
for tr, te in outer.split(X, y):
    grid.fit(X[tr], y[tr])
    scores.append(grid.score(X[te], y[te]))
print(f"Nested CV: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

</div>

<div class="level-next">
<span>Want generalization bounds, info-criteria, and conformal prediction?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Information criteria</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{AIC} = 2k - 2\,\ln L \qquad \text{BIC} = k\,\ln n - 2\,\ln L $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`L`maximised likelihood; `k` parameters; `n` sample size

</li>
<li markdown="1">

AIC: minimum predicted Kullback-Leibler divergence from the truth

</li>
<li markdown="1">

BIC: approximation to marginal log-likelihood (Bayesian model evidence)

</li>
<li markdown="1">

Lower is better; pick the model that minimises

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{AIC} = 2 \times \text{(parameter count)} - 2 \times \ln(\text{likelihood}) \qquad \text{BIC} = \text{(parameter count)} \times \ln(n) - 2 \times \ln(\text{likelihood}) $$</span>

**In words.** Both scores are "model fit minus a complexity penalty". The fit piece is `−2 ln(likelihood)` — lower means the model explains the data better. The penalty piece punishes you for using more parameters: AIC charges `2` per parameter; BIC charges `ln(n)` per parameter (which is harsher once you have more than a handful of data points). `ln` is the natural logarithm. Compute either score for several candidate models and pick the smallest — same idea as cross-validation but without re-training.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`likelihood`how probable the data is under the fitted model (bigger = better fit)

</li>
<li markdown="1">

`parameter count`how many tunable knobs the model has

</li>
<li markdown="1">

`n`number of training examples

</li>
<li markdown="1">

AIC: minimum predicted distance from the truth (Kullback-Leibler)

</li>
<li markdown="1">

BIC: approximation to the Bayesian model evidence

</li>
<li markdown="1">

Lower is better; pick the model that minimises

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Information criteria.** When you have a likelihood and a parameter count, AIC and BIC give a "model fit + complexity penalty" score without holding out data. Cheap to compute; useful for fast model comparison when CV is expensive. AIC tends to pick larger models than BIC (BIC's penalty grows with *n*); neither is a substitute for a held-out test, but they're useful diagnostics.

**Conformal prediction.** Vovk's framework gives prediction *intervals* with finite-sample coverage guarantees under exchangeability — distribution-free, model-agnostic. Pair any base predictor with a conformal calibration step and you get rigorous "95% of new examples will land in this interval". Increasingly popular for deployment.

**Selection bias in repeated tuning.** Even with proper CV, if you tune for long enough you'll find configurations that look great by chance. Reuse of the same validation set ("validation set rot") gives you optimistic numbers. Use a held-out final test set, and resist the urge to "just check" performance early.

**Multiple-comparison adjustments.** When you compare many models on the same dataset, the best score on validation is positively biased — you've selected the winner. Holdout-set adjustments (Recht et al., 2018) and corrected confidence intervals exist; "reuse the test set" is dangerous at scale.

**PAC-Bayes and stability bounds.** Modern theory connects generalization to the stability of the learning algorithm under perturbations of the training set. CV is essentially an empirical stability estimate. Algorithmic stability + small training loss ⇒ small generalization gap.

**When CV lies.** Distribution shift (train and test from different distributions), label noise, group structure missed by random splits, and computational cost are all reasons CV's number can be wrong. Always sanity-check with a final, untouched holdout.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# AIC by hand — useful for non-sklearn models
def aic_gaussian(y_true, y_pred, k):
    n = len(y_true)
    resid = y_true - y_pred
    sigma2 = (resid ** 2).mean()
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return 2 * k - 2 * ll

# Conformal prediction — distribution-free intervals
def split_conformal(model, X_train, y_train, X_calib, y_calib, X_test, alpha=0.1):
    model.fit(X_train, y_train)
    # Residuals on a held-out calibration set
    resid = np.abs(y_calib - model.predict(X_calib))
    q = np.quantile(resid, 1 - alpha, method="higher")
    y_pred = model.predict(X_test)
    return y_pred - q, y_pred + q     # (1 − α) coverage guaranteed
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

[Elements of Statistical Learning — Ch. 7 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">The reference chapter on model selection, AIC/BIC, cross-validation, and the bias-variance trade-off in selection.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Bergstra & Bengio (2012) — Random Search <i class="fas fa-external-link-alt"></i>](https://www.jmlr.org/papers/v13/bergstra12a.html){: target="_blank" }
<span class="annotation">The empirical paper showing random search beats grid search in high-dimensional hyperparameter spaces.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — Cross-Validation Guide <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/cross_validation.html){: target="_blank" }
<span class="annotation">Excellent practical reference for the many CV strategies. Code-first; includes diagrams of each split type.</span>

</li>
<li data-tier="indepth" markdown="1">

[Angelopoulos & Bates (2021) — Gentle Intro to Conformal Prediction <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2107.07511){: target="_blank" }
<span class="annotation">Modern tutorial on conformal prediction — distribution-free prediction intervals with finite-sample coverage. Lots of code.</span>

</li>
</ul>

</div>

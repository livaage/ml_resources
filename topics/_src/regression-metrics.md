---
title: Regression Metrics — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Regression Metrics
lead: MSE, MAE, R², MAPE — what they measure, when each one lies.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**"Average error" hides outliers; "relative error" hides zeros.** Each regression metric makes a different bargain with reality. MSE punishes big mistakes; MAE treats all of them equally; R² compares your model to "always predict the mean"; MAPE answers "what fraction off are we, on average" — but blows up when the truth is zero.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Drag the outlier — watch MSE explode while MAE shrugs; the model line is OLS fit to all the points</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Outlier y
                <input id="viz-rm-outlier" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-rm-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-rm-out-lbl">y = 0.0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-rm-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-rm-caption"></div>
</div>

<script src="{{root}}js/viz/regression-metrics.js"></script>

Drag the orange outlier away from the trend and watch the metric strip below: **MSE** rises quadratically, **MAE** rises linearly, **R²** drops sharply because the variance the model has to explain just grew. A single point can move MSE by an order of magnitude — that's why MSE-trained models often "chase" outliers.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**MSE** (mean squared error). Bayes-optimal if errors are Gaussian; punishes large errors quadratically. Same units as *y²* — usually report √MSE (RMSE) for interpretability.

**MAE** (mean absolute error). Same units as *y*. Robust to outliers — corresponds to predicting the median rather than the mean.

**R²** (coefficient of determination). *1 − SSE/SST*. The fraction of variance the model explains relative to "always predict the mean". 1 = perfect, 0 = no better than the mean, negative = worse than the mean. Pseudo-R² for nonlinear models is similar in spirit but unitless.

**MAPE** (mean absolute *percentage* error). Scale-free, easy to communicate ("our forecast is 12% off on average"). Useless when *y* can be zero or near-zero.

**Pinball loss / quantile loss.** The right metric for quantile regression — measures how well you predicted the τ-th quantile, not the mean.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Reach for

- **RMSE**: default, especially when big errors matter most
- **MAE**: outliers, robust reporting
- **R²**: "how much variance did I explain?"
- **MAPE**: forecasting where y > 0 always
- **Pinball**: prediction intervals

</div>

<div class="no" markdown="1">

### Don't use

- MAPE when y can be zero or close to it
- MSE alone when outliers exist — report MAE too
- R² for non-linear or non-Gaussian models without care
- Any single metric without sanity-checking residual plots

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score,
)

rmse  = mean_squared_error(y_true, y_pred, squared=False)
mae   = mean_absolute_error(y_true, y_pred)
mape  = mean_absolute_percentage_error(y_true, y_pred)
r2    = r2_score(y_true, y_pred)

# Pinball loss for quantile regression
def pinball(y_true, y_pred, tau=0.5):
    e = y_true - y_pred
    return np.maximum(tau * e, (tau - 1) * e).mean()
```

</div>

<div class="level-next">
<span>Want adjusted-R², MSLE, calibration, and proper scoring rules for regression?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Mean squared error decomposed</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{MSE} = \mathbb{E}\!\big[(y - \hat y)^2\big] = (\mathbb{E}[\hat y] - y^*)^2 + \mathbb{V}[\hat y] + \sigma_\text{noise}^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Bias² of the mean prediction

</li>
<li markdown="1">

Variance of the predictions across resampled training sets

</li>
<li markdown="1">

Irreducible noise — the floor no model can beat

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{MSE} \;=\; \text{average of (truth − prediction)}^2 \;=\; \text{bias}^2 \;+\; \text{variance} \;+\; \text{noise}^2 $$</span>

**In words.** Mean squared error is the average of the squared gap between truth and prediction. That average breaks into three pieces that add up: *bias²* (how far your *average* prediction sits from the truth — systematic error), *variance* (how much your predictions wobble across re-fits of the model on fresh data), and *noise²* (the unavoidable scatter in the labels themselves). The last piece is a floor no model can break through. This is exactly the bias-variance decomposition seen from the metric's side.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`MSE`average squared error across the data

</li>
<li markdown="1">

`bias²`squared gap between average prediction and the truth

</li>
<li markdown="1">

`variance`spread of predictions when you re-train on different samples

</li>
<li markdown="1">

`noise²`irreducible randomness in the labels — the floor no model can beat

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Why RMSE is unstable in practice.** One outlier can multiply RMSE by 10×. If you can't avoid them, report MAE alongside, or look at quantiles of the squared residual rather than the mean. Robust alternatives include median absolute error and trimmed RMSE.

**Adjusted R².** Standard R² always rises when you add features (more flexibility, more variance explained). Adjusted R² penalises that: *1 − (1 − R²)(n − 1)/(n − p − 1)*. Use it when comparing models with different numbers of features.

**MSLE (mean squared log error).** Apply MSE to *log(1 + y)*. Symmetric in relative terms, robust to large positive outliers. Useful for skewed targets like prices, counts, or income.

**Predicting distributions.** Don't just minimise mean error — predict quantiles or full distributions. Pinball loss for quantiles; CRPS (continuous ranked probability score) for full predicted distributions. CRPS reduces to MAE when the prediction is a point mass.

**Residual diagnostics.** Plot residuals against predictions, against each feature, against time. Patterns there reveal model failures that aggregate metrics hide — heteroskedasticity, missing interactions, time leakage.

**The "loss vs. metric" mismatch.** Train on MSE if you want to predict means; on MAE if you want medians; on pinball if you want quantiles. Optimising the wrong thing and then evaluating with the right one is a common mistake.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

# Robust RMSE — trim the 5% worst residuals before averaging
def trimmed_rmse(y_true, y_pred, trim=0.05):
    sq = (y_true - y_pred) ** 2
    k = int(len(sq) * (1 - trim))
    return np.sqrt(np.sort(sq)[:k].mean())

# CRPS — proper score for predicted distributions
def crps_gaussian(y, mu, sigma):
    z = (y - mu) / sigma
    from scipy.stats import norm
    return sigma * (z * (2 * norm.cdf(z) - 1) +
                    2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
```

</div>

<div class="level-next">
<span>Want CRPS, log-score, and proper scoring theory?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">CRPS — proper score for distributions</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(z) - \mathbb{1}\{y \leq z\}\big)^2\, dz $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`F`predicted CDF

</li>
<li markdown="1">

Reduces to MAE if *F* is a point mass at the prediction

</li>
<li markdown="1">

Proper — minimised when *F* matches the true distribution

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{CRPS} \;=\; \text{integrated squared gap between the predicted distribution and the observed step function} $$</span>

**In words.** CRPS scores a *distributional* forecast against the single observed value. Think of the predicted distribution as a smooth S-shaped curve (its CDF, `F`), and the actual observation as a step function that jumps from 0 to 1 at the observed value. The CRPS is the area between those two curves, squared and added up across the whole number line (that's what the integral `∫` does). When the predicted distribution collapses to a single point, this reduces to MAE; when it spreads out badly, CRPS punishes both the location and the width.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`predicted distribution F`your full probabilistic forecast (its CDF)

</li>
<li markdown="1">

`observed step`step function that jumps from 0 to 1 at the actual observation

</li>
<li markdown="1">

`integrated gap`area between the two curves, squared and summed across all values

</li>
<li markdown="1">

Reduces to MAE if the prediction is a point mass; proper — minimised when the prediction matches the truth

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Proper scoring rules for regression.** A scoring rule *S(F, y)* is "proper" if *E<sub>Y∼G</sub> S(F, Y)* is minimised at *F = G*. Log score, CRPS, and energy score are all proper for predictive distributions. MAE and squared error are proper for point predictions targeting the median and mean respectively.

**Coverage and calibration of intervals.** If a model says "95% confidence", verify that 95% of the actual observations fall in the predicted interval on a held-out set. Empirical coverage below nominal = under-confident; above = over-confident. Conformal prediction gives exact coverage guarantees under exchangeability.

**Forecast horizons.** For time-series, report errors at multiple horizons separately — error at *t+1* and *t+30* can be very different. SMAPE (symmetric MAPE) is the classical forecasting metric; modern competitions (M4, M5) use a weighted combination of multiple metrics.

**Heteroskedastic targets.** When the noise depends on the input (variance grows with magnitude), uniform metrics can be misleading. Predict and evaluate quantile bands, or use a normalising transformation (log, Box-Cox) before fitting.

**Forecast skill scores.** Compare model performance to a baseline (climatology, persistence, naive forecast) — the skill score is *1 − error/baseline_error*. Important in weather, finance, and epidemiology where the dataset's intrinsic difficulty changes over time.

**Bootstrap CIs for metrics.** Don't just report a single number; bootstrap residuals or resampled test sets to put a confidence interval on RMSE, MAE, or R². Two models with a "10% improvement" can be statistically indistinguishable.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

# Bootstrap CI for any regression metric
def bootstrap_metric(y_true, y_pred, metric, B=1000, alpha=0.05):
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        vals.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return np.mean(vals), lo, hi

# Empirical coverage of prediction intervals
def coverage(y, lower, upper):
    return ((y >= lower) & (y <= upper)).mean()
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

[scikit-learn — Regression Metrics <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics){: target="_blank" }
<span class="annotation">Definitions of every regression metric in the library with mathematical formulae and code.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Hyndman & Athanasopoulos — Forecasting: Principles and Practice <i class="fas fa-external-link-alt"></i>](https://otexts.com/fpp3/accuracy.html){: target="_blank" }
<span class="annotation">The reference book for forecast evaluation. Chapter 5 covers point and probabilistic forecast accuracy in depth.</span>

</li>
<li data-tier="indepth" markdown="1">

[Gneiting & Raftery (2007) — Strictly Proper Scoring Rules <i class="fas fa-external-link-alt"></i>](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437){: target="_blank" }
<span class="annotation">The reference paper on proper scoring rules for both classification and regression. Heavier going but defines the concepts precisely.</span>

</li>
<li data-tier="indepth" markdown="1">

[Angelopoulos & Bates — Conformal Prediction Intro <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2107.07511){: target="_blank" }
<span class="annotation">Distribution-free prediction intervals with finite-sample coverage guarantees — same tutorial linked from the Model Selection page.</span>

</li>
</ul>

</div>

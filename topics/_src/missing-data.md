---
title: Missing Data — ML Resources Hub
eyebrow_text: ← Theory · Data &amp; Features
eyebrow_href: ../theory.html
heading: Missing Data
lead: Drop, impute, or model — handling the NaN problem without lying to yourself about it.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Missing isn't random.** People skip survey questions for reasons; sensors fail in patterns; missingness itself carries information. The question is whether you can ignore the why (drop), pretend it's noise (impute), or model it explicitly (use missingness as a feature).

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Toggle the imputation strategy — see what each one does to a column with 30% missing values</span>
</div>
<div class="viz-classic-controls">
<button id="viz-md-mean" type="button" class="active">Mean</button>
<button id="viz-md-median" type="button">Median</button>
<button id="viz-md-knn" type="button">k-NN</button>
<button id="viz-md-iter" type="button">Iterative (MICE)</button>
<button id="viz-md-flag" type="button">+ missingness flag</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-md-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-md-caption"></div>
</div>

<script src="{{root}}js/viz/missing-data.js"></script>

A 2D dataset with 30% of one column's values missing. **Mean** imputation pulls the missing rows to the column mean — visibly distorts the distribution. **Median** is more robust to outliers. **k-NN** imputes from nearby (in the observed dimensions) rows — preserves local structure. **Iterative** (MICE) regresses each missing column on the others and iterates — captures correlations. Adding a **missingness flag** lets the model learn from the fact something was missing.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Drop rows.** The simplest answer. Fine when missingness is rare (< 1–2% of rows) and unrelated to the target. Wasteful when data is precious.

**Drop columns.** If a column is > ~50% missing, often easier to drop. But beware: a column with a high missing rate *and* strong signal in the observed rows is exactly the kind of column you should keep + flag.

**Mean / median / mode imputation.** Fast, simple, available in every library. Mean preserves the mean; median is more robust; mode for categorical. All three shrink the variance of the imputed column — downstream models think it's less variable than it is.

**k-NN imputation.** Fill each missing cell from the average of its k nearest neighbours (computed on the observed columns). Preserves local structure; can be slow on large datasets.

**Iterative imputation (MICE).** Regress each column on all the others, iterate. Best general-purpose strategy on tabular data. `sklearn.IterativeImputer` implements it.

**Add a missing-flag.** Add a binary column "was column X missing for this row?". Lets the model see the missingness pattern itself — often more useful than the imputed value.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Pick your strategy

- < 1% missing → drop rows
- Moderate missing, tabular → MICE + missingness flag
- Sequential / time-series → forward-fill or interpolate, then flag
- Trees (XGBoost, LightGBM) → handle missing natively, often best

</div>

<div class="no" markdown="1">

### Common mistakes

- Imputing on the full dataset before splitting — leakage
- Mean imputation everywhere — shrinks variance, biases models
- Treating "missing" as a magic value like -999 — trees treat it as just another number
- Forgetting that missingness itself can be predictive

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
import pandas as pd

# Always inside a Pipeline so it fits on TRAIN ONLY
mean   = SimpleImputer(strategy="mean")
median = SimpleImputer(strategy="median")
knn    = KNNImputer(n_neighbors=5, weights="distance")
mice   = IterativeImputer(max_iter=10, random_state=0)

# Indicator columns — add a 0/1 flag for each missing position
indicator = SimpleImputer(add_indicator=True, strategy="median")

# XGBoost / LightGBM handle NaN natively — often the best answer for tabular
import xgboost as xgb
xgb.XGBClassifier().fit(X, y)         # NaNs OK
```

</div>

<div class="level-next">
<span>Want MAR/MCAR/MNAR theory and multiple imputation?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The Little & Rubin taxonomy</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{MCAR: } P(M\mid X, Y) = P(M)\quad\text{MAR: } P(M\mid X, Y) = P(M\mid X)\quad\text{MNAR: depends on } Y $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`M`indicator of missingness

</li>
<li markdown="1">

MCAR — completely random (rare in practice)

</li>
<li markdown="1">

MAR — depends only on observed values (most assumed)

</li>
<li markdown="1">

MNAR — depends on the missing value itself (the hard case)

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \begin{aligned}
\text{MCAR:} &\;\;\text{P(missing)} \;=\; \text{constant} \\
\text{MAR:} &\;\;\text{P(missing)} \;=\; \text{depends on observed features only} \\
\text{MNAR:} &\;\;\text{P(missing)} \;=\; \text{depends on the unobserved value itself}
\end{aligned} $$</span>

**In words.** Three regimes for *why* data is missing. `M` is just a 0/1 indicator that says "this cell is missing"; `P(M | ...)` means "probability of missing given everything else". MCAR (missing completely at random) says the probability doesn't depend on anything — pure coincidence. MAR (missing at random) says it depends only on stuff you *can* see — so smart imputation using the other columns can recover the missing values without bias. MNAR (missing not at random) says it depends on the value you *can't* see (e.g. high earners refuse to report income) — no purely observational fix is unbiased here, you need a model of *why* it's missing.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`P(missing)`chance that a cell is missing

</li>
<li markdown="1">

`observed features`the values you do have for that row

</li>
<li markdown="1">

`unobserved value`the value of the cell that's actually missing

</li>
<li markdown="1">

MCAR rare in practice; MAR is the assumption behind most imputation; MNAR needs domain knowledge

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**MCAR vs MAR vs MNAR.** Missing Completely At Random: the missingness pattern is independent of everything (essentially never true). Missing At Random: depends only on observed variables (the assumption behind most imputation methods). Missing Not At Random: depends on the missing value itself (e.g., people with high income don't disclose income). MNAR is the hard case — needs domain knowledge or modelled missingness mechanism.

**Multiple imputation.** Don't impute once and pretend the gap was filled. Impute multiple times (drawing from the predictive distribution), fit your model on each, pool the results. `statsmodels.MICEData` and the `fancyimpute` / `miceforest` libraries implement this. Critical for valid inference (confidence intervals) on partially missing data.

**Why mean imputation hurts.** Replacing missing values with the mean shrinks the variance of that column by exactly the fraction missing. Linear regressions get attenuated coefficients; SEs are too small; any downstream test will be over-confident. Multiple imputation fixes this; missingness flags partially help.

**Tree-based models handle NaN.** XGBoost, LightGBM, CatBoost, and modern scikit-learn HistGradientBoosting all have built-in NaN handling — they learn a default direction at each split. Often outperforms any imputation strategy on tabular data.

**Time-series patterns.** Forward-fill ("carry the last observed value forward") and linear interpolation are common. Both assume slow-varying signals; for fast-varying or irregular sampling, fit a model (Kalman filter, Gaussian process) and impute from the posterior.

**Sensitivity analysis.** Try multiple imputation methods and compare. Big differences mean missingness is doing real work; choose the method whose assumptions match your domain.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import miceforest as mf
import pandas as pd

# Multiple imputation by chained equations (MICE) with random-forest steps
kernel = mf.ImputationKernel(df, num_datasets=5, save_all_iterations=True)
kernel.mice(iterations=10)
df_imputed_list = [kernel.complete_data(dataset=i) for i in range(5)]

# Fit your model on each imputation and average — Rubin's rules
preds = [model.fit(d, y).predict(X_test) for d in df_imputed_list]
final = np.mean(preds, axis=0)            # variance across runs = uncertainty
```

</div>

<div class="level-next">
<span>Want missingness as a feature, MNAR models, and selection-bias correction?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Imputation with the observed likelihood</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \hat\theta = \arg\max_\theta \int p(X_\text{obs}, X_\text{miss} \mid \theta)\, dX_\text{miss} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

EM is the canonical algorithm for missing-data MLE

</li>
<li markdown="1">

Treats imputation and parameter estimation as joint optimisation

</li>
<li markdown="1">

Requires a (correct) model for both the data and the missingness

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{best parameters} \;=\; \arg\max_\theta \;\; \text{average likelihood of observed data, averaging over possible missing values} $$</span>

**In words.** If you don't see all the data, the right thing to do is pick the model parameters that make the data you *did* see as likely as possible — while averaging over every possible value the missing cells could have been. The `∫ ... dX_miss` is an integral that does exactly that averaging (marginalising out the unknowns). `θ̂` ("theta hat") is the estimate you end up with; `arg max` means "the *θ* that maximises". This integral is hard to compute directly — that's why EM (Expectation-Maximisation) exists: it alternates between filling in the missing values from the current model and re-estimating the model parameters.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`best parameters (θ̂)`the parameter estimates that explain the observed data best

</li>
<li markdown="1">

`arg max`"the value of *θ* that maximises the expression"

</li>
<li markdown="1">

`observed data`the cells you actually have

</li>
<li markdown="1">

`missing values`the cells you don't — averaged over, not pretended to be known

</li>
<li markdown="1">

EM is the standard algorithm for actually computing *θ̂*

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**MNAR modelling.** When missingness depends on the unobserved value (income, drug-use, sensitive variables), no purely observational fix is unbiased. Two general approaches: *selection models* (model *P(missing | X, Y)* explicitly, e.g. Heckman correction) and *pattern-mixture models* (model the distribution conditional on missingness pattern). Both require strong assumptions that you should be transparent about.

**Missingness as a feature.** In some domains the missingness *is* the signal. EMR data: a test wasn't ordered because the doctor didn't think it was necessary — the missingness encodes information. Add binary flags or use models that natively support missingness.

**Generative imputation.** Modern approaches use GAINS (GAN-based imputation), normalising flows, or VAEs to impute. State of the art on heterogeneous tabular data; sometimes overkill when MICE works.

**Selection bias correction.** When the missingness depends on the target, propensity-score weighting or doubly-robust estimation can correct biased estimates — but only if you correctly model the selection mechanism. Domain expertise is irreplaceable here.

**Out-of-distribution missingness.** Train-time and serve-time missingness patterns often differ. Build robustness into both the model and the monitoring — flag prediction-time inputs where the missingness pattern is unusual.

**Right-censoring.** A special case: events that haven't happened yet are "missing" in time-to-event data. Survival analysis (Kaplan-Meier, Cox PH) handles this without imputing.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Iterative imputer with non-linear estimator → robust to non-Gaussian patterns
mice = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=20, random_state=0),
    max_iter=10, random_state=0,
)
X_imp = mice.fit_transform(X)

# Heckman correction for selection bias (income-disclosure example)
# Step 1: model the probability of disclosure as probit
# Step 2: include the inverse Mills ratio as a regressor in the outcome model
import statsmodels.api as sm
sel  = sm.Probit(disclosed, X_sel).fit()
imr  = -sel.pdf(sel.fittedvalues) / sel.cdf(sel.fittedvalues)
out  = sm.OLS(income[disclosed], np.c_[X_out[disclosed], imr[disclosed]]).fit()
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

[van Buuren — Flexible Imputation of Missing Data <i class="fas fa-external-link-alt"></i>](https://stefvanbuuren.name/fimd/){: target="_blank" }
<span class="annotation">Free online textbook by the inventor of MICE. The reference book for both theory and practical recipes.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — Imputation Guide <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/impute.html){: target="_blank" }
<span class="annotation">Practical reference for SimpleImputer, KNNImputer, IterativeImputer with worked examples.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Allison — Missing Data <i class="fas fa-external-link-alt"></i>](https://www.statisticalhorizons.com/wp-content/uploads/MissingDataByML.pdf){: target="_blank" }
<span class="annotation">Concise and well-written intro to MAR / MCAR / MNAR taxonomy with worked statistical examples.</span>

</li>
<li data-tier="indepth" markdown="1">

[Yoon et al. (2018) — GAIN <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1806.02920){: target="_blank" }
<span class="annotation">Generative-adversarial imputation — strong on heterogeneous tabular data, useful when MICE struggles.</span>

</li>
</ul>

</div>

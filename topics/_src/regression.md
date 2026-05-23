---
title: Linear &amp; Logistic Regression — ML Resources Hub
eyebrow_text: ← Theory · Linear Models
eyebrow_href: {{root}}theory.html
heading: Linear &amp; Logistic Regression
lead: The simplest, most interpretable models — and still the right answer surprisingly often.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Predict the output as a weighted sum of the inputs.** Linear regression does this for continuous outputs (predicting a number). Logistic regression squishes the output through an S-curve to predict probabilities (yes/no). Both are interpretable, fast, and surprisingly hard to beat on simple problems.

</div>

<div class="viz-embed">
<div class="viz-embed-header">
<span class="viz-embed-title">Drag points · click empty space to add · <kbd>shift</kbd>+click to remove</span>
<div class="viz-embed-controls">
<select id="viz-regression-degree" aria-label="fit degree"></select>
<button id="viz-regression-noisy" type="button">Noisy</button>
<button id="viz-regression-curvy" type="button">Sine</button>
<button id="viz-regression-clear" type="button">Clear</button>
<button id="viz-regression-reset" type="button">Reset</button>
</div>
</div>
<canvas id="viz-regression-canvas"></canvas>
<div class="viz-embed-stats" id="viz-regression-stats"></div>
</div>

<script src="{{root}}js/viz/regression.js"></script>

The fit minimises the sum of squared vertical distances (the faint orange lines). Try the *Sine* preset, then crank the degree up — degree 9 will pass through almost every point but wiggle wildly in between. That's **overfitting** in two clicks.
{: .viz-intro }

<div class="viz-grid">
<a href="https://setosa.io/ev/ordinary-least-squares-regression/" target="_blank" class="viz-card viz-card-featured">
<span class="viz-source">setosa.io</span>
<h3>Setosa — OLS Regression</h3>
<p>Beautiful interactive explanation of OLS from first principles. Drag points to see how each one influences the fit; explore the geometric interpretation of least squares.</p>
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
<a href="https://mlu-explain.github.io/linear-regression/" target="_blank" class="viz-card">
<span class="viz-source">mlu-explain.github.io</span>
<h3>MLU-Explain — Linear Regression</h3>
<p>Amazon's visual walk-through. Covers residuals, R², and the assumptions behind OLS with clean animated diagrams.</p>
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
<a href="https://seeing-theory.brown.edu/regression-analysis/index.html" target="_blank" class="viz-card">
<span class="viz-source">seeing-theory.brown.edu</span>
<h3>Seeing Theory — Regression</h3>
<p>Brown's interactive primer. Drag the regression line manually and see the loss change in real time. Best for building intuition for what OLS is optimising.</p>
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
</div>

<article class="tldr-body" markdown="1">

Linear regression: *ŷ = β₀ + β₁·x₁ + β₂·x₂ + …*. The coefficients *β<sub>i</sub>* tell you how much each feature changes the prediction. Big positive coefficient on "square footage" means more square footage → higher predicted price. Negative on "age" means older → lower price.

Logistic regression is the same idea but for classification. Take the same weighted sum, then squash it through a sigmoid (S-shaped function) to get a probability between 0 and 1. The coefficients now tell you how much each feature moves the predicted log-odds.

Both are the workhorse baselines you should try before anything fancier. If a linear model gets 90% accuracy, a deep neural net adding 1% probably isn't worth it.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Linear (or roughly linear) relationships
- Interpretability matters — coefficients are readable
- Few hundred to few thousand data points
- You want a fast, hard-to-beat baseline before going complex

</div>

<div class="no" markdown="1">

### Skip it when

- Strongly non-linear relationships (use trees / neural nets)
- Feature interactions matter and you can't engineer them in
- Very high-dim sparse data — needs lots of regularization care
- The output is a sequence, image, or other structured object

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

# Predict a number
reg = LinearRegression().fit(X_train, y_train)
print(f"R²: {reg.score(X_test, y_test):.3f}")
print(f"Coefficients: {dict(zip(X_train.columns, reg.coef_))}")

# Predict yes/no
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
print(f"Probabilities: {clf.predict_proba(X_test[:5])}")
```

</div>

<div class="level-next">
<span>Want OLS, regularization, and the math?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Linear regression — OLS</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \hat{\boldsymbol{\beta}} \;=\; \arg\min_{\boldsymbol{\beta}}\; \|\mathbf{y} - X\boldsymbol{\beta}\|_2^2 \;=\; (X^\top X)^{-1} X^\top \mathbf{y} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`y`target vector (length *N*)

</li>
<li markdown="1">

`X`design matrix (*N × p*; usually include a column of ones for intercept)

</li>
<li markdown="1">

`β`coefficient vector — closed-form solution exists

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{best coefficients} \;=\; \text{coefficients that minimise the sum of squared residuals} $$</span>

**In words.** Each prediction is `X × β` — the inputs combined by the coefficients `β` (beta). The residual at point *i* is `actual − predicted`; square them all and add them up, and you get the total squared error. OLS picks whichever `β` makes that total as small as possible. For squared loss this has a closed-form answer (no iteration needed): the matrix expression on the right is the linear-algebra recipe — invert `XᵀX` and multiply by `Xᵀy`. If *N* ≫ *p* and features aren't co-linear, it gives the unique best linear fit.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`best coefficients`β̂ — the weight on each feature in the fitted model

</li>
<li markdown="1">

`residual`actual − predicted, for each point

</li>
<li markdown="1">

`sum of squared residuals`add up (actual − predicted)² across all training points

</li>
<li markdown="1">

`design matrix`your features stacked into a table, usually with a column of ones for the intercept

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Ordinary Least Squares (OLS).** Linear regression with a squared-error loss has a closed-form solution — no iteration needed. The estimator is unbiased (under classical assumptions) and minimum-variance among linear unbiased estimators (Gauss-Markov).

**Logistic regression.** Replace the squared loss with cross-entropy and the linear output with a sigmoid: *p̂ = σ(Xβ)*. No closed form — solve via gradient descent or Newton's method (IRLS). The loss is convex, so it always converges to the global optimum.

**Regularization.** When features are correlated or you have more features than data points, OLS variance explodes. Add a penalty on the coefficients:

**Ridge (L2):** penalty = *λ‖β‖²*. Shrinks all coefficients toward zero but rarely makes them exactly zero. Closed form: *β̂ = (XᵀX + λI)⁻¹Xᵀy*.

**Lasso (L1):** penalty = *λ‖β‖₁*. Drives many coefficients exactly to zero — performs feature selection. No closed form; solve via coordinate descent.

**Elastic Net:** mix of both. Often the right default when you're not sure.

**Cross-validated *α*.** Sweep over a grid of penalty strengths, pick the one that minimises held-out error. `LassoCV`, `RidgeCV`, `ElasticNetCV` do this automatically.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Feature engineering can capture the structure (polynomials, interactions, splines)
- Many features, modest data — regularization wins
- Sparse models needed (Lasso for feature selection)
- You need calibrated probability estimates (logistic regression is well-calibrated by default)

</div>

<div class="no" markdown="1">

### Skip it when

- Strong feature interactions that aren't engineered in
- Discontinuous or sharply non-linear relationships
- You want hierarchical / multi-output models — switch to Bayesian or GLM frameworks
- The data is overwhelmingly non-tabular

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.linear_model import RidgeCV, LassoCV, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Regression with cross-validated regularization strength
ridge = Pipeline([
    ("scale", StandardScaler()),
    ("reg",   RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)),
]).fit(X_train, y_train)

print(f"Chosen α: {ridge.named_steps['reg'].alpha_:.4f}")
print(f"R² test: {ridge.score(X_test, y_test):.3f}")

# Lasso for sparse coefficients
lasso = Pipeline([
    ("scale", StandardScaler()),
    ("reg",   LassoCV(cv=5, max_iter=10000)),
]).fit(X_train, y_train)

selected = (lasso.named_steps["reg"].coef_ != 0).sum()
print(f"Lasso selected {selected} of {X_train.shape[1]} features")

# Classification — same pattern, with L1 or L2 penalty
clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", max_iter=1000).fit(X_train, y_train)
print(f"Best C: {clf.C_}")
```

</div>

<div class="level-next">
<span>Want the GLM family, MLE derivation, and the assumptions?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Generalized Linear Models</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ g(\mathbb{E}[y \mid x]) \;=\; \boldsymbol{\beta}^\top \mathbf{x}, \qquad y \sim \text{exponential family} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`g`link function — identity (linear), logit (logistic), log (Poisson)

</li>
<li markdown="1">

Linear regression, logistic, Poisson, gamma — all the same recipe with different link & likelihood

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{link function}\big(\text{average of }y\text{ given }x\big) \;=\; \text{dot product of }\beta\text{ and }x, \qquad y \sim \text{exponential family distribution} $$</span>

**In words.** A GLM still has a linear predictor — a weighted sum `β · x` — but instead of being equal to the mean of *y* directly, it equals the mean after passing it through a "link function" `g`. The choice of `g` (and the choice of distribution for *y*) is what specialises the model: identity link with Gaussian *y* recovers ordinary linear regression; logit link with Bernoulli *y* gives logistic regression; log link with Poisson *y* gives Poisson regression for counts. The "exponential family" is a broad class of distributions (Gaussian, Bernoulli, Poisson, gamma, etc.) that share a friendly maximum-likelihood structure.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`link function`g — a function (identity, logit, log…) connecting the linear predictor to the response mean

</li>
<li markdown="1">

`average of y given x`the expected value of the response at input *x*

</li>
<li markdown="1">

`dot product of β and x`weighted sum of features — the linear predictor

</li>
<li markdown="1">

`exponential family`the broad class of distributions (Gaussian, Bernoulli, Poisson…) compatible with GLM fitting

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The GLM viewpoint.** Linear regression assumes *y | x ~ N(βᵀx, σ²)*. Logistic regression assumes *y | x ~ Bernoulli(σ(βᵀx))*. Both fit by maximum likelihood. Generalize: pick any distribution from the exponential family + a "link function" mapping the linear predictor to the distribution's mean. Poisson regression for count data, gamma for positive continuous, multinomial for multi-class.

**OLS assumptions.** Strictly speaking, OLS is unbiased + minimum-variance under: (1) linearity, (2) independent observations, (3) homoscedasticity (constant variance), (4) no perfect multicollinearity. Normality of residuals is needed for inference (CIs, p-values), *not* for the point estimates.

**Multicollinearity.** When features are highly correlated, OLS coefficients become unstable — sign and magnitude can flip with small data changes. Diagnose with VIF (variance inflation factor). Fix with ridge regularization or by dropping / combining correlated features.

**Regularization paths.** As you increase *λ* from 0 to ∞, coefficients shrink (ridge) or hit zero one by one (lasso). Plotting the path is informative — shows you the order in which features matter.

**Coordinate descent & LARS.** The standard fast algorithms for lasso. LARS (Least Angle Regression) builds the full path in one pass — useful when you want to see all *λ* values, not just one.

**Beyond linear.** Generalized Additive Models (GAMs) keep additivity but let each term be a flexible function: *ŷ = f₁(x₁) + f₂(x₂) + …*. Interpretable like GLMs but expressive like neural nets in each dimension. Use `pyGAM` or `statsmodels` for these.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You want interpretable coefficients with confidence intervals — GLM in statsmodels
- Non-Gaussian targets (Poisson counts, gamma durations, ordinal outcomes)
- Bayesian linear regression with priors on coefficients
- GAMs when you want non-linear effects but additive structure

</div>

<div class="no" markdown="1">

### Skip it when

- You need top-tier predictive accuracy with non-additive interactions
- Sequential or spatial structure dominates
- Deep representations are doing most of the work
- The link function and likelihood don't match the data — better to use a flexible regressor

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import statsmodels.api as sm

# GLM with explicit family / link — proper inference for coefficients
X_train_const = sm.add_constant(X_train)

# Poisson regression for count data
model = sm.GLM(y_count, X_train_const, family=sm.families.Poisson()).fit()
print(model.summary())            # coefficients, std errors, p-values, deviance

# Logistic with full inference
logit = sm.GLM(y_binary, X_train_const, family=sm.families.Binomial()).fit()
print(f"Coefs:   {logit.params.round(3).to_dict()}")
print(f"95% CI:  {logit.conf_int().values.round(3)}")
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

[ISLR — chapters 3 & 4 <i class="fas fa-external-link-alt"></i>](https://www.statlearning.com/){: target="_blank" }
<span class="annotation">"Introduction to Statistical Learning" — free PDF. The cleanest intro to linear / logistic regression with worked examples in Python and R.</span>

</li>
<li data-tier="indepth" markdown="1">

[ESL — chapters 3 & 4 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">Same authors, more rigorous treatment. Free PDF. Read after ISLR when you want the math.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — linear models <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/linear_model.html){: target="_blank" }
<span class="annotation">Comprehensive practical reference covering OLS, ridge, lasso, elastic net, logistic, and the lesser-known variants.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[statsmodels <i class="fas fa-external-link-alt"></i>](https://www.statsmodels.org/){: target="_blank" }
<span class="annotation">When you need proper inference (CIs, p-values, GLM diagnostics). The Python library that doesn't reduce statistics to "predict()".</span>

</li>
</ul>

</div>

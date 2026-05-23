---
title: Bias-Variance &amp; Overfitting — ML Resources Hub
eyebrow_text: ← Theory · Generalization
eyebrow_href: ../theory.html
heading: Bias-Variance &amp; Overfitting
lead: The trade-off at the heart of why models fail to generalize.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

There are **two ways** a model can be wrong. **Bias**: the model is too simple to capture the pattern. **Variance**: the model has memorized the training data, including its noise. Most fixes trade one for the other.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide the polynomial degree — 30 fits on different samples overlay; the bias-variance U-curve sits on the right</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Target
                <select id="viz-bv-target"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Degree
                <input id="viz-bv-degree" class="viz-classic-slider" type="range"></input>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                σ
                <input id="viz-bv-noise" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-bv-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-bv-degree-lbl">deg = 4</span>
<span class="viz-classic-badge" id="viz-bv-noise-lbl">σ = 0.18</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-bv-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-bv-caption"></div>
</div>

<script src="{{root}}js/viz/bias-variance.js"></script>

Left: 30 polynomials of the chosen degree, each fit to a different noisy sample of the target. The bold orange line is their pointwise mean. **Bias** is how far that mean misses the target (dashed). **Variance** is the spread of the faded indigo curves around the mean. Right: those two quantities as the degree changes — bias drops, variance rises, and total error makes a U. Sliding the degree slider moves a vertical line through that plot so you can see exactly where you are on the trade-off curve.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Picture trying to fit a straight line through data that's actually curved. The line is *biased* — no matter how much data you give it, it can't bend enough to match. Now picture a wiggly curve that passes through every single training point. Zero training error, but it's memorized the noise — show it new points and they'll be all over the map. That's *variance*.

The art of training a model is finding the sweet spot in between: complex enough to capture the real pattern, simple enough to ignore the noise.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### High bias (underfitting)

- Training error is poor
- Test error is also poor
- The model can't even fit the data it's trained on
- **Fix:** more features, a more complex model, less regularization

</div>

<div class="no" markdown="1">

### High variance (overfitting)

- Training error is tiny
- Test error is much larger
- The model "memorized" the training set
- **Fix:** more data, a simpler model, regularization, ensembling

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Quickest diagnostic: compare training vs test accuracy
print(f"Train: {model.score(X_train, y_train):.3f}")
print(f"Test:  {model.score(X_test,  y_test):.3f}")

# If both are low      -> high bias (underfitting)
# If train >> test     -> high variance (overfitting)
# If both reasonable   -> you're in the sweet spot
```

</div>

<div class="level-next">
<span>Ready for the underlying maths?</span>
<button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The decomposition</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathbb{E}\!\left[(y - \hat{f}(x))^2\right] \;=\; \mathrm{Bias}[\hat{f}(x)]^{\,2} \;+\; \mathrm{Var}[\hat{f}(x)] \;+\; \sigma^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`E[…]`expected (average) value over all possible training sets & inputs

</li>
<li markdown="1">

`f̂(x)`the model's prediction at input *x*

</li>
<li markdown="1">

`Bias`how far the *average* prediction sits from the truth

</li>
<li markdown="1">

`Var`how much predictions wobble as you re-sample the training data

</li>
<li markdown="1">

`σ²`variance of the irreducible label noise

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{average squared error} \;=\; \text{bias}^2 \;+\; \text{variance} \;+\; \text{noise}^2 $$</span>

**In words.** The total error your model makes (averaged over fresh data) breaks cleanly into three pieces that *add up*: a piece from being systematically off (bias), a piece from being inconsistent across re-fits (variance), and a piece from randomness in the labels themselves (noise). The `²` just means "squared" — we square bias and noise before adding so the units match (everything ends up in "error squared"). You can only do something about the first two terms; the third is the floor.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`avg sq err`average of *(true value − prediction)²* across the data

</li>
<li markdown="1">

`bias`how far the average prediction sits from the truth, if you re-trained on many different samples

</li>
<li markdown="1">

`variance`how spread out those predictions are around their average

</li>
<li markdown="1">

`noise`the standard deviation of irreducible randomness in the labels

</li>
</ul>

</div>

Total expected error splits cleanly into three terms. Two of them are *your* problem; one isn't.

</div>

<article class="tldr-body" markdown="1">

**Bias** is error from picking the wrong model family. A linear model has high bias on a quadratic problem no matter how much data you feed it — the family of straight lines simply doesn't contain the right answer.

**Variance** is error from being too sensitive to the specific training sample. A deep, unpruned tree picks up on noise; re-fit it on a slightly different sample and you get a noticeably different model.

**Noise** is the floor: randomness in *y* given *x*. No model can do better.

The trade-off comes from how knobs move bias and variance together:

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Increases bias, reduces variance

- Simpler model class (lower polynomial degree, fewer features)
- Stronger regularization (L1, L2, dropout)
- Smaller trees, more pruning
- Smaller neural networks

</div>

<div class="no" markdown="1">

### Reduces variance, bias unchanged

- More training data
- Ensembling (bagging, random forests)
- Averaging across initial seeds

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

# Convergence behaviour of the two curves diagnoses the regime:
#   gap between them is large and stable -> high variance
#   both plateau at a high error          -> high bias
#   both converge low                     -> you're in the sweet spot
```

</div>

<div class="level-next">
<span>Curious how this breaks down for modern deep nets?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Formal decomposition (squared loss)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \begin{aligned}
\mathbb{E}_{D,\varepsilon}\!\left[(y - \hat{f}_D(x))^2\right]
&\;=\; \underbrace{\left(\bar{f}(x) - f(x)\right)^2}_{\text{bias}^2}
\;+\; \underbrace{\mathbb{E}_D\!\left[\left(\hat{f}_D(x) - \bar{f}(x)\right)^2\right]}_{\text{variance}}
\;+\; \sigma^2
\end{aligned} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`D`random training sample drawn from the data distribution

</li>
<li markdown="1">

`f(x)`true underlying function

</li>
<li markdown="1">

`f̂D(x)`the model fit on a particular dataset *D*

</li>
<li markdown="1">

`f̄(x)`shorthand for *E<sub>D</sub>[f̂<sub>D</sub>(x)]* — the average prediction across training sets

</li>
<li markdown="1">

`ε`noise; *y* = *f*(*x*) + *ε* with *Var*(*ε*) = *σ²*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \begin{aligned}
\text{average squared error}
&\;=\; \underbrace{(\text{avg prediction} - \text{truth})^2}_{\text{bias}^2}
\;+\; \underbrace{\text{average of }(\text{prediction} - \text{avg prediction})^2}_{\text{variance}}
\;+\; \text{noise}^2
\end{aligned} $$</span>

**In words.** Same three-way split as Fundamentals, but now we're explicit about what's being averaged. Imagine re-running training many times — each time on a fresh sample of data, with fresh label noise. Each run gives you a different prediction at any given input *x*. The first term, **bias²**, is the squared gap between the *average* of those predictions and the truth. The second, **variance**, is how much the predictions wobble around that average from run to run. The third, **noise²**, is the unavoidable scatter in the labels themselves. The under-braces ( `⏟` ) are just labels saying which piece is which.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`avg`averaging is over *both* the training sample you happen to draw *and* the noise in the labels

</li>
<li markdown="1">

`truth`the underlying pattern *f(x)* the model is trying to learn

</li>
<li markdown="1">

`prediction`what your model outputs at *x* after fitting on a particular dataset

</li>
<li markdown="1">

`avg prediction`average of the model's predictions across all the training sets you could have drawn

</li>
<li markdown="1">

`noise`random scatter in the labels themselves — what's left after the true pattern

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

The decomposition holds **for squared loss only**. It does not generalize cleanly to 0-1 loss or cross-entropy; the most common attempt at a unified treatment is Domingos (2000). For classification, the practical analogue is the trade-off between training and test error, but "bias" and "variance" become heuristic labels rather than algebraic terms.

**Double descent.** The classical U-shape in test error vs. capacity describes only the under-parameterized regime. For modern interpolating models — wide neural networks, kernel methods at the interpolation threshold — test error often *decreases again* as capacity grows past the point where training error hits zero. Belkin et al. (2019) named this the *double descent* phenomenon.

**Implication.** "Increasing capacity inflates variance" is a sound principle for classical ML but a poor description of deep nets. The implicit bias of the optimizer (SGD with weight decay) does most of the regularization work even when explicit regularization is minimal. Bias-variance is still a useful frame, but only one of several — see also benign overfitting (Bartlett et al.) and norm-based generalization bounds.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Classical regime

- Number of params ≪ training points
- Linear models, trees, classical ML
- Decomposition is quantitatively useful
- Capacity ↑ → variance ↑ holds reliably

</div>

<div class="no" markdown="1">

### Modern overparameterized regime

- Models interpolate the training set (zero training loss)
- Deep nets, wide kernels, transfer-learned models
- Double descent: test error drops past interpolation
- Generalization governed by optimizer's implicit bias

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.utils import resample
from sklearn.base import clone

# Empirical bias-variance estimate via bootstrap
def bias_variance(estimator, X_train, y_train, X_test, y_test, n_boot=200, seed=0):
    rng = np.random.RandomState(seed)
    preds = np.zeros((n_boot, len(X_test)))
    for b in range(n_boot):
        Xb, yb = resample(X_train, y_train, random_state=rng.randint(1e9))
        preds[b] = clone(estimator).fit(Xb, yb).predict(X_test)

    mean_pred = preds.mean(axis=0)
    bias_sq  = ((mean_pred - y_test) ** 2).mean()
    variance =  preds.var(axis=0).mean()
    noise    =  max(0.0, ((y_test - mean_pred) ** 2).mean() - bias_sq)
    return bias_sq, variance, noise
```

</div>

<div class="level-next">
<span>Too dense?</span>
<button data-go-to="fundamentals" type="button">← Back to Fundamentals</button>
</div>

</section>

<!-- TOPIC SIDEBAR -->

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">

[MLU-Explain — Bias-Variance <i class="fas fa-external-link-alt"></i>](https://mlu-explain.github.io/bias-variance/){: target="_blank" }
<span class="annotation">Interactive visualization showing how complexity moves bias and variance. Best intuition builder.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[ESL, chapter 7 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">The canonical treatment of bias-variance and model assessment. Read alongside chapter 2 for context.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Wikipedia <i class="fas fa-external-link-alt"></i>](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff){: target="_blank" }
<span class="annotation">Quick reference for the derivation and connections to regularization.</span>

</li>
<li data-tier="indepth" markdown="1">

[Belkin et al. (2019) <i class="fas fa-external-link-alt"></i>](https://www.pnas.org/doi/10.1073/pnas.1903070116){: target="_blank" }
<span class="annotation">"Reconciling modern ML practice and the classical bias-variance trade-off" — the double-descent paper. Short, accessible.</span>

</li>
<li data-tier="indepth" markdown="1">

[Domingos (2000) <i class="fas fa-external-link-alt"></i>](https://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf){: target="_blank" }
<span class="annotation">Unified bias-variance decomposition for 0-1, squared, and other losses. Read when squared-loss isn't enough.</span>

</li>
</ul>

</div>

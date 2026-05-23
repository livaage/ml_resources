---
title: Calibration — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Calibration
lead: When the model says 70%, is it right 70% of the time? Calibration is whether predicted probabilities mean what they say.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Accuracy says "right or wrong". Calibration says "well-aimed".** A model is calibrated if, among the cases where it predicts 70%, exactly 70% turn out positive. Many models — especially neural nets — are over-confident: they say 99% on cases where the true rate is closer to 85%. The fix is usually *post-hoc* — apply a calibrator to the model's output without retraining.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Toggle the miscalibration · watch the reliability diagram skew · "Temperature scale" fixes it post-hoc</span>
</div>
<div class="viz-classic-controls">
<button id="viz-cal-cal" type="button" class="active">Well calibrated</button>
<button id="viz-cal-over" type="button">Over-confident</button>
<button id="viz-cal-under" type="button">Under-confident</button>
<button id="viz-cal-temp" type="button">Temperature scale</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-cal-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-cal-caption"></div>
</div>

<script src="{{root}}js/viz/calibration.js"></script>

The reliability diagram bins predicted probabilities along the x-axis and plots the observed positive rate on the y-axis. A perfectly calibrated model sits on the diagonal. **Over-confident** models bow below it (say 90%, actually 75% positive); **under-confident** models bow above. "Temperature scaling" divides logits by a learned scalar — a one-parameter post-hoc fix that often nails it.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Modern neural networks are typically *over-confident*. Cross-entropy training, particularly with strong models on easy data, pushes predicted probabilities toward 0 or 1 — past what the data warrants.

**Temperature scaling**: divide logits by a learned T > 0 before softmax, fit T on validation data to minimise log-loss. Single parameter, doesn't change predictions (the argmax is preserved), fixes most cases.

**Platt scaling**: fit a logistic regression on the model's scores. Best for SVMs and other models whose output isn't a probability.

**Isotonic regression**: fit a non-decreasing step function from scores to probabilities. Strictly more flexible than Platt; needs more calibration data.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Calibrate when

- You're going to *use* the probability (cost-sensitive decisions, ensembles, downstream models)
- The model is a neural network with cross-entropy loss
- You're combining multiple classifiers (calibration is a prerequisite for proper averaging)
- Reporting probabilities to a human decision-maker

</div>

<div class="no" markdown="1">

### Skip calibration when

- You only care about ranking, not absolute probabilities (AUC)
- You only need the argmax (just accuracy or top-k)
- You'll already use a downstream decision threshold

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.calibration import CalibratedClassifierCV
import torch, torch.nn.functional as F

# Sklearn — wrap any classifier, fit Platt or isotonic on a held-out fold
cal = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
cal.fit(X_train, y_train)
p_cal = cal.predict_proba(X_test)

# Temperature scaling for a neural net — one learnable parameter
class TempScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.T = torch.nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.T

# Fit T on a validation set with LBFGS, minimising NLL
ts  = TempScale().cuda()
opt = torch.optim.LBFGS([ts.T], lr=0.01, max_iter=50)
def closure():
    opt.zero_grad()
    loss = F.cross_entropy(ts(logits_val), y_val)
    loss.backward()
    return loss
opt.step(closure)
print(f"Learned T = {ts.T.item():.3f}")        # > 1 ⇒ was over-confident
```

</div>

<div class="level-next">
<span>Want ECE, MCE, beta calibration, and conformal alternatives?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Expected calibration error</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right| $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Bm`predictions binned by predicted probability (e.g. 15 equal-width bins)

</li>
<li markdown="1">

`acc(Bm)`observed positive rate in the bin

</li>
<li markdown="1">

`conf(Bm)`average predicted probability in the bin

</li>
<li markdown="1">

Weighted average gap — ECE = 0 is perfect calibration

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{ECE} \;=\; \text{weighted average across bins of} \;\big|\text{observed positive rate} \;-\; \text{average predicted probability}\big| $$</span>

**In words.** Sort all predictions into *M* bins by predicted probability (e.g. bin 1 = "predicted 0–10%", bin 2 = "10–20%", and so on). For each bin, compute two things: the *observed positive rate* (what fraction were actually positive) and the *average predicted probability* (what the model said on average). The gap between those two is the per-bin miscalibration. Average those gaps, weighted by how many points landed in each bin, and you get ECE. Zero means perfect calibration; bigger numbers mean the model's confidence levels don't match reality.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`bin`group of predictions in a probability range (e.g. "predicted 60–70%")

</li>
<li markdown="1">

`observed positive rate`fraction of points in the bin that were actually positive

</li>
<li markdown="1">

`predicted probability`average of the model's confidence in the bin

</li>
<li markdown="1">

Weighted by bin size; ECE = 0 means perfect calibration

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**ECE and its limits.** ECE is the canonical calibration metric, but it depends on the binning scheme — different bin counts can give very different ECEs. MCE (max calibration error) reports the worst bin. Adaptive binning (equal-mass instead of equal-width) is fairer when probabilities are concentrated.

**Temperature scaling intuition.** Dividing logits by *T > 1* softens the softmax — pushes probabilities away from the corners (0/1) toward uniform. Neural networks trained with cross-entropy push toward extremes; *T* learns to dial that back. Doesn't change which class is predicted, just the confidence.

**Beta calibration.** Kull et al. (2017) — a 2-parameter generalisation of Platt scaling with better empirical performance. Useful when isotonic over-fits.

**Conformal prediction.** Instead of fixing the probability, fix the *set*: produce prediction sets that cover the true label with guaranteed probability. Distribution-free; works on top of any base model.

**Class-conditional calibration.** A model can be globally calibrated but mis-calibrated on a specific class. Report per-class calibration in multi-class problems.

**Why this matters in production.** Many ML systems consume probabilities downstream: cost-sensitive decisions, expected-value computations, fraud-detection cascades. Mis-calibration means the downstream code is doing arithmetic on numbers that don't mean what they should.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.calibration import calibration_curve

# Reliability diagram data
prob_true, prob_pred = calibration_curve(y_val, p_pred, n_bins=15, strategy="quantile")

# ECE with adaptive binning
def adaptive_ece(y, p, n_bins=15):
    order = np.argsort(p)
    y, p = y[order], p[order]
    bin_size = len(y) // n_bins
    e = 0
    for i in range(n_bins):
        s = slice(i * bin_size, (i + 1) * bin_size)
        e += abs(y[s].mean() - p[s].mean()) * bin_size / len(y)
    return e
```

</div>

<div class="level-next">
<span>Want proper scoring, multi-class calibration, and OOD calibration?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Brier decomposition</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{Brier} = \underbrace{\text{Reliability}}_{\text{miscalibration}} - \underbrace{\text{Resolution}}_{\text{spread of cond. freq.}} + \underbrace{\text{Uncertainty}}_{\text{base rate}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Brier as a sum of three orthogonal terms

</li>
<li markdown="1">

Reliability ↓ as calibration improves

</li>
<li markdown="1">

Resolution ↑ as the model usefully separates positives from negatives

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{Brier score} \;=\; \text{miscalibration} \;-\; \text{discrimination spread} \;+\; \text{base-rate uncertainty} $$</span>

**In words.** The Brier score (mean squared error between predicted probabilities and labels) decomposes into three orthogonal pieces. *Reliability* is miscalibration — how badly your stated probabilities match reality; lower is better. *Resolution* is how spread out your predictions are across cases — a model that varies its predictions from 5% to 95% has more resolution than one that always says 50%; it's subtracted because more resolution makes Brier smaller. *Uncertainty* is the baseline difficulty (entropy of the label distribution) — unavoidable. The decomposition lets you see *why* a model has bad Brier: miscalibrated or just not very informative?
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Brier score`mean squared error between predicted probabilities and labels

</li>
<li markdown="1">

`miscalibration`how poorly your probabilities match observed frequencies

</li>
<li markdown="1">

`discrimination spread`how varied your predictions are across cases (higher = better)

</li>
<li markdown="1">

`base-rate uncertainty`baseline entropy of the labels — unavoidable

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Multi-class calibration is harder.** Class-conditional calibration, top-label calibration, and full-distribution calibration all have different definitions. Temperature scaling addresses top-label calibration well; *matrix scaling* and *vector scaling* extend it but can over-fit. Dirichlet calibration (Kull et al. 2019) is a principled multi-class generalisation.

**OOD and calibration.** Calibration on the training distribution doesn't imply calibration under shift. Neural networks are dramatically over-confident on out-of-distribution inputs — this is one of the harder open problems in ML safety. Deep ensembles and Bayesian neural nets help; large-margin training and outlier exposure help too.

**Calibrated probabilities ≠ Bayesian.** Calibration is a frequentist consistency property: the long-run frequency in a bin matches the predicted probability. Bayesian uncertainty is something different — it tells you how much you should believe each plausible model given the data. A model can be calibrated without being Bayesian, and a Bayesian model isn't automatically calibrated under prior misspecification.

**Calibration vs. selective prediction.** Sometimes you'd rather abstain than predict an uncertain answer. Selective classification frames this directly: choose a confidence threshold below which you decline. Plumbing this into a deployed system is easier with calibrated probabilities than with raw logits.

**Histogram binning, BBQ, ENIR.** A zoo of post-hoc calibrators beyond Platt/isotonic/temperature. Histogram binning is simple; BBQ averages over multiple binnings; ENIR uses nearly-isotonic regression. All are useful when temperature scaling isn't enough but isotonic over-fits.

**Calibration under deferral.** When the model can defer to a human, the calibrator should know that — joint calibration of model + deferral is an active research area (e.g. learn-to-defer literature).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
import torch, torch.nn.functional as F

# Matrix temperature scaling for multi-class
class MatrixScale(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.W = torch.nn.Parameter(torch.eye(K))
        self.b = torch.nn.Parameter(torch.zeros(K))
    def forward(self, logits): return logits @ self.W + self.b

# Deep-ensemble calibration: average probabilities from multiple seeds
def ensemble_probs(models, x):
    ps = [F.softmax(m(x), dim=-1) for m in models]
    return torch.stack(ps).mean(dim=0)         # better calibrated + better accuracy
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

[Guo et al. (2017) — On Calibration of Modern Neural Networks <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1706.04599){: target="_blank" }
<span class="annotation">The paper that brought "modern nets are miscalibrated" into the mainstream and proposed temperature scaling. Required reading.</span>

</li>
<li data-tier="indepth" markdown="1">

[Minderer et al. (2021) — Revisiting Calibration <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2107.03342){: target="_blank" }
<span class="annotation">Empirical study showing modern vision models (ViT, MLP-Mixer) are actually fairly well-calibrated out of the box, unlike the convnets in Guo et al.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — Calibration Guide <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/calibration.html){: target="_blank" }
<span class="annotation">Practical reference for Platt, isotonic, and reliability diagrams with code.</span>

</li>
<li data-tier="indepth" markdown="1">

[Angelopoulos & Bates — Conformal Prediction Intro <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2107.07511){: target="_blank" }
<span class="annotation">Alternative to calibrating probabilities: produce prediction sets with guaranteed coverage. Increasingly the preferred answer for high-stakes deployment.</span>

</li>
</ul>

</div>

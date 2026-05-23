---
title: Loss Functions — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Loss Functions
lead: What you ask the model to minimise — and why the answer changes the model.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The loss is the question you're asking.** "How wrong was that prediction?" has many possible answers — squared error, absolute error, cross-entropy, hinge. Each answer pulls the model in a different direction. Choosing the loss is choosing what mistakes you care about.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Hover the chart to read the loss at each point · toggle Regression / Classification · slide the outlier to see its weight</span>
</div>
<div class="viz-classic-controls">
<button id="viz-loss-reg" type="button" class="active">Regression</button>
<button id="viz-loss-cls" type="button">Classification</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Huber δ
                <input id="viz-loss-delta" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-loss-delta-lbl">δ = 0.5</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-loss-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-loss-caption"></div>
</div>

<script src="{{root}}js/viz/losses.js"></script>

In **regression** mode the x-axis is the residual *y − ŷ*: MSE punishes big errors quadratically (sensitive to outliers); MAE punishes linearly (robust); Huber blends them, quadratic near zero and linear far away. In **classification** mode the x-axis is the margin *y·f(x)*: cross-entropy keeps penalising even confident-correct predictions a little, hinge stops at margin 1, and the 0/1 loss is what you really want but can't optimise (it's flat almost everywhere).
{: .viz-intro }

<article class="tldr-body" markdown="1">

**For regression.** The default is mean squared error — it's smooth, differentiable everywhere, and Bayes-optimal for predicting the mean under Gaussian noise. Mean absolute error is more robust to outliers (it predicts the median instead). Huber loss splits the difference: quadratic for small errors, linear for big ones.

**For classification.** The default is cross-entropy (also called log-loss) — it pairs with softmax, has well-behaved gradients, and corresponds to maximum-likelihood for a categorical model. Hinge loss is the SVM choice — it stops penalising once you're past the margin. Focal loss down-weights easy examples and is useful for highly imbalanced data.

The loss you pick changes what the model learns to do: minimise MSE and you predict means; minimise MAE and you predict medians; minimise cross-entropy and you predict calibrated probabilities (in the limit).

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Common defaults

- **Regression**: MSE if noise is Gaussian-ish, Huber if there are outliers
- **Binary classification**: binary cross-entropy with sigmoid
- **Multi-class**: categorical cross-entropy with softmax
- **Ranking**: pairwise / listwise (NDCG, ListNet)

</div>

<div class="no" markdown="1">

### Worth a closer look when

- Heavy class imbalance — try focal loss
- Heavy outliers — try Huber or quantile losses
- You care about a custom business metric — write a custom loss
- The mean / median isn't the right summary — try quantile regression

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# Regression
mse   = F.mse_loss(y_pred, y_true)                          # default
mae   = F.l1_loss(y_pred, y_true)                           # robust to outliers
huber = F.smooth_l1_loss(y_pred, y_true, beta=1.0)          # blend

# Binary classification
bce   = F.binary_cross_entropy_with_logits(logits, y_true)   # numerically stable

# Multi-class
ce    = F.cross_entropy(logits, y_true)                      # logits, NOT softmax-ed

# Class-imbalanced — focal loss
def focal(logits, y, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    p   = torch.sigmoid(logits)
    pt  = y * p + (1 - y) * (1 - p)
    return ((1 - pt).pow(gamma) * bce).mean()
```

</div>

<div class="level-next">
<span>Want the probabilistic story and proper scoring rules?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Loss as negative log-likelihood</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(y_i \mid x_i) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`pθ(y | x)`the probability the model puts on the actual label *y* given input *x*

</li>
<li markdown="1">

`N`number of training examples

</li>
<li markdown="1">

For Gaussian `p(y|x)` with fixed variance, this is MSE up to a constant

</li>
<li markdown="1">

For Bernoulli `p(y|x)`, this is binary cross-entropy

</li>
<li markdown="1">

For categorical `p(y|x)`, this is multi-class cross-entropy

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; -\frac{1}{N} \sum_{i=1}^{N} \log (\text{model's predicted probability of the actual label}) $$</span>

**In words.** Average across all *N* training examples of the negative log of how much probability the model assigns to the *correct* answer. When the model is confident and right, the probability is close to 1 and `−log` is close to 0 — small loss. When it's confidently wrong, the probability is close to 0 and `−log` is huge. Different choices of probability model give rise to all the familiar losses: a Gaussian gives MSE, a Bernoulli gives binary cross-entropy, a categorical gives multi-class cross-entropy. So "pick a loss" is really "pick a noise model".
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`predicted probability of the actual label`what the model assigns to the right answer for this example

</li>
<li markdown="1">

`log`natural logarithm — turns multiplied probabilities into added log-probabilities

</li>
<li markdown="1">

`1/N · Σ`average across all *N* training examples

</li>
<li markdown="1">

Choice of *p(y|x)* ↔ choice of loss: Gaussian → MSE, Bernoulli → binary cross-entropy, categorical → multi-class cross-entropy

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Every "standard" loss is really a likelihood.** MSE = Gaussian negative log-likelihood. Binary cross-entropy = Bernoulli NLL. Multi-class cross-entropy = categorical NLL. Picking a loss is implicitly picking a noise model — and that's clarifying. Heavy-tailed errors? Use a Laplace likelihood (gives MAE) or Student-t. Counts? Poisson. Multiple outputs with correlation? A multivariate Gaussian with a learned covariance.

**Proper scoring rules.** A loss is "proper" if its minimum is at the true distribution — log-loss and Brier score are proper for probabilistic classification, accuracy isn't. Using a proper loss is what makes calibrated probabilities come out of training.

**Gradient behaviour matters.** MSE's gradient grows with error (large mistakes pull harder); MAE's gradient is constant (steady push regardless of magnitude); cross-entropy's gradient has the nice "logit − target" form. These shape what training looks like — MAE is harder for gradient descent because the gradient is non-zero at the minimum from one side.

**Loss vs. metric.** The loss is what you optimise (must be differentiable); the metric is what you report. They're often different — you optimise log-loss but evaluate accuracy or F1. Misalignment is fine when the loss is a good proxy, but worth watching: a model with great log-loss can still be poorly calibrated, etc.

**Custom losses.** When the off-the-shelf options don't match your business problem, write one. Common patterns: weighted losses (per-class or per-sample), penalised losses (add a regularizer), quantile losses (predict any quantile, not just the mean).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn.functional as F

# Quantile regression — predict the τ-th quantile, not the mean
def quantile_loss(y_pred, y_true, tau=0.5):
    e = y_true - y_pred
    return torch.maximum(tau * e, (tau - 1) * e).mean()

# Train three models at τ = 0.1, 0.5, 0.9 and you get prediction intervals.

# Per-class weighted cross-entropy for class imbalance
weights = torch.tensor([1.0, 3.0, 1.0])      # class 1 is rare
loss    = F.cross_entropy(logits, y, weight=weights)

# Brier score — proper, well-calibrated, good for early stopping
def brier(probs, y_onehot):
    return ((probs - y_onehot) ** 2).sum(dim=-1).mean()
```

</div>

<div class="level-next">
<span>Want robust losses, MAML inner losses, and adversarial losses?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Influence function</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{I}(z; \theta) = -H_\theta^{-1} \, \nabla_\theta \ell(z; \theta) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`z`a single training point

</li>
<li markdown="1">

`ℓ(z; θ)`per-example loss at parameters *θ*

</li>
<li markdown="1">

`∇θ ℓ`gradient of that loss at the trained parameters

</li>
<li markdown="1">

`H`Hessian of the average loss at the trained parameters

</li>
<li markdown="1">

How much does the trained parameter change if we up-weight point *z*? A measurable consequence of the loss's gradient shape near the optimum

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{influence of point } z \;=\; -(\text{curvature of loss})^{-1} \,\times\, (\text{gradient of point's loss}) $$</span>

**In words.** This formula answers the question: if you nudged the weight of training point *z* a tiny bit, how would the trained parameters change? The right-hand side has two pieces. **Gradient of point's loss** says which direction *z* pulls the parameters. Multiplying by the inverse **curvature** (the Hessian, written `H`) rescales that pull by how sensitive the loss is in each direction — flat directions feel a big change, steep directions a small one. The result is a vector — same shape as the parameters — telling you the parameter-shift per unit upweight. The minus sign just means "moving the loss down" rather than up.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`influence of point z`predicted change in trained parameters per unit upweight of *z*

</li>
<li markdown="1">

`gradient of point's loss`direction this single point pulls the parameters

</li>
<li markdown="1">

`curvature of loss`Hessian — second derivatives, captures how sharp the loss is in each direction

</li>
<li markdown="1">

`(·)−1`inverse curvature — rescales by sensitivity

</li>
<li markdown="1">

Heavy-tailed losses (MSE) → unbounded influence; robust losses (MAE, Huber) → bounded influence

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Robust statistics.** The influence function quantifies how much a single training point can move the trained parameter. MSE's quadratic shape gives *unbounded* influence — one extreme point can swing the fit arbitrarily. MAE's bounded influence is what makes it robust. Huber, Tukey's biweight, and other M-estimators sit on this spectrum, trading efficiency at the Gaussian model for robustness against contamination.

**Focal loss.** Lin et al. (2017) introduced *FL(p) = −(1 − p)<sup>γ</sup> log p* for object detection's extreme class imbalance. The *(1 − p)<sup>γ</sup>* factor down-weights easy examples (where the model is already confident), letting the gradient focus on the hard ones. γ = 2 is a strong default.

**Adversarial losses.** The generator's objective in a GAN is a function of another model's output — the loss landscape becomes a moving target. WGAN, hinge loss, and least-squares GAN are all attempts to give the generator stable gradients. The lesson generalises: any loss whose target depends on another learned thing gets the same difficulty.

**Surrogate losses.** Hinge, log-loss, exponential, etc. are all *surrogates* for the 0/1 loss you actually want for classification. Bartlett, Jordan & McAuliffe (2006) characterised which surrogates are "calibrated" — their minimiser agrees with the Bayes classifier. Bad surrogates can converge to the wrong thing even with infinite data.

**Auxiliary and contrastive losses.** Modern self-supervised learning is built on auxiliary losses — InfoNCE, triplet, supervised contrastive, all encode different ideas of "similar" and "different". The form of the contrastive loss determines what representation emerges (alignment, uniformity, dimensional collapse).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn.functional as F

# InfoNCE — the canonical contrastive loss
def info_nce(query, positives, temperature=0.07):
    # query: (B, d). positives: (B, d) — same indexing.
    # negatives are everyone else in the batch.
    logits = (query @ positives.t()) / temperature       # (B, B)
    labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits, labels)

# Triplet loss — anchor, positive, negative
def triplet(anchor, pos, neg, margin=0.2):
    d_p = (anchor - pos).norm(dim=-1)
    d_n = (anchor - neg).norm(dim=-1)
    return F.relu(d_p - d_n + margin).mean()
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

[Deep Learning Book — Ch. 5 <i class="fas fa-external-link-alt"></i>](https://www.deeplearningbook.org/contents/ml.html){: target="_blank" }
<span class="annotation">Goodfellow, Bengio &amp; Courville's chapter on ML fundamentals lays out losses as negative log-likelihoods cleanly.</span>

</li>
<li data-tier="indepth" markdown="1">

[Lin et al. (2017) — Focal Loss <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1708.02002){: target="_blank" }
<span class="annotation">The focal-loss paper, with a clean motivation for why down-weighting easy examples helps when classes are extremely imbalanced.</span>

</li>
<li data-tier="indepth" markdown="1">

[Bartlett, Jordan, McAuliffe — Convexity, Classification, & Risk Bounds <i class="fas fa-external-link-alt"></i>](https://www.jmlr.org/papers/v18/16-365.html){: target="_blank" }
<span class="annotation">The reference paper on which surrogate losses for classification are "calibrated" — i.e. don't converge to the wrong thing.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — Contrastive Representation Learning <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2021-05-31-contrastive/){: target="_blank" }
<span class="annotation">A tour of contrastive losses (InfoNCE, SimCLR, triplet) with intuition for what each one buys you.</span>

</li>
</ul>

</div>

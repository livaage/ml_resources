---
title: Regularization — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Regularization
lead: How to keep a model from memorising its training data — L1, L2, dropout, early stopping, and friends.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Penalise complexity along with error.** An unregularized model is free to fit noise. Add a penalty on the size of the parameters (L2), on the count of nonzero parameters (L1), or on the network's expressiveness (dropout, weight decay, early stopping) — and the model trades a little training fit for much less overfitting.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide λ — watch L2 shrink every coefficient and L1 zero some out completely</span>
</div>
<div class="viz-classic-controls">
<button id="viz-reg-l2" type="button" class="active">L2 (Ridge)</button>
<button id="viz-reg-l1" type="button">L1 (LASSO)</button>
<button id="viz-reg-en" type="button">Elastic Net</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                λ
                <input id="viz-reg-lambda" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-reg-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-reg-lambda-lbl">λ = 0.10</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-reg-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-reg-caption"></div>
</div>

<script src="{{root}}js/viz/regularization.js"></script>

A linear model is fit to data with 12 features — only 4 are actually informative, the rest are noise. The bars show each coefficient's value. Slide λ from 0 to high under **L1** and watch noise features snap to zero (sparsity). Under **L2** every coefficient shrinks smoothly (no exact zeros, just smaller). Elastic Net does some of both — most useful when correlated features collapse into one under pure L1.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**L2 (Ridge / weight decay).** Add a penalty proportional to the sum of squared weights. Every coefficient gets pulled toward zero, smoothly. No coefficient ever becomes *exactly* zero, but they all get smaller in proportion to the penalty strength. The default for neural networks (often called "weight decay") and the safest general-purpose regularizer.

**L1 (LASSO).** Add a penalty proportional to the sum of absolute weights. This makes some coefficients *exactly* zero — the model does feature selection while it fits. Useful when you suspect most features are irrelevant.

**Dropout.** Randomly drop a fraction of neurons at each training step. Forces the network not to over-rely on any single unit. The original "neural network regularization" trick, still useful for MLPs and some Transformers.

**Early stopping.** Stop training when validation loss stops decreasing. Effectively limits how far the optimizer can wander; equivalent to a particular form of L2 for deep networks under some assumptions.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You see train ≪ test error (overfitting)
- You have many features and suspect most are noise → L1
- You want smooth, stable solutions → L2
- Deep network — start with weight decay 1e-4 to 1e-5

</div>

<div class="no" markdown="1">

### Be careful when

- You're under-fitting — regularization makes it worse
- Features are on different scales — standardise first, or the penalty is uneven
- L1 with correlated features picks arbitrarily — use Elastic Net
- Dropout + BatchNorm interact oddly — check ordering

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import torch

# Classical: pick alpha by cross-validation
ridge   = Ridge(alpha=1.0)                # L2
lasso   = Lasso(alpha=0.1)                # L1 — sparse
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Deep learning: AdamW does decoupled weight decay (correct L2 for adaptive)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Dropout: just a layer
net = torch.nn.Sequential(
    torch.nn.Linear(128, 64), torch.nn.ReLU(),
    torch.nn.Dropout(0.2),                # zeros 20% of activations at train time
    torch.nn.Linear(64,  1),
)
```

</div>

<div class="level-next">
<span>Want the underlying maths?</span>
<button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Regularised loss</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}_\text{reg}(\theta) \;=\; \mathcal{L}(\theta) \;+\; \lambda\, \Omega(\theta) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`L(θ)`the usual training loss (e.g. squared error, cross-entropy)

</li>
<li markdown="1">

`Ω(θ)`penalty on the parameters

</li>
<li markdown="1">

`λ`regularization strength — bigger λ shrinks the parameters harder

</li>
<li markdown="1">

`‖θ‖₂²`L2 / ridge: sum of each parameter squared

</li>
<li markdown="1">

`‖θ‖₁`L1 / LASSO: sum of absolute values — induces sparsity

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{regularised loss} \;=\; \text{training loss} \;+\; \lambda \times \text{penalty}(\text{weights}) $$</span>

**In words.** Instead of just minimising the training loss, you minimise the training loss *plus* a penalty for using big weights. `λ` (lambda, a Greek letter conventionally used for "regularization strength") sets how harshly that penalty bites: at `λ = 0` you ignore the penalty entirely; the bigger λ gets, the more the optimiser is pushed toward small weights even if that hurts the fit a little. The art is tuning λ — usually by trying a few values and picking the one that gives the best validation score.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`training loss`the usual error you're minimising (e.g. squared error)

</li>
<li markdown="1">

`penalty`some measure of how big the weights are

</li>
<li markdown="1">

`λ`regularization strength — bigger λ shrinks the weights harder

</li>
<li markdown="1">

`L2 penalty`add up (each weight)² across all weights — pulls every weight toward zero smoothly

</li>
<li markdown="1">

`L1 penalty`add up |each weight| — drives some weights exactly to zero

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Bayesian interpretation.** L2 regularization is equivalent to assuming the weights are drawn from a bell-curve centred at zero — finding the most-likely weights under that assumption gives you ridge regression. L1 corresponds to a sharper distribution with heavier tails, which is why it produces sparsity. This framing makes "choose λ" the same as "choose how confidently you believe the weights are near zero".

**Why L1 gives exact zeros.** Draw the constraint regions: the L1 ball has corners on the axes; the L2 ball is round. A quadratic loss contour touching the L1 ball is most likely to touch at a corner (where some coordinates are zero). That geometric argument is the reason LASSO does feature selection while ridge doesn't.

**Weight decay in deep learning.** Adding an L2 term to the loss *and* using Adam is subtly wrong — Adam's per-parameter scaling makes the effective decay rate uneven across parameters. AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient: instead of adding λ·w to the gradient, it multiplies w by (1 − η·λ) directly. The fix matters; transformer recipes assume AdamW.

**Dropout as ensembling.** At inference time, dropout is off and the network uses all neurons — but the training behaviour is equivalent to averaging exponentially many sub-networks. This is one of several reasons dropout works.

**Batch / layer norm as implicit regularization.** Normalization layers don't penalise weight magnitude, but they make the loss less sensitive to weight magnitude (because the activations are rescaled). The effect is a form of implicit regularization — and it interacts oddly with explicit weight decay. Modern transformer recipes often use very small or zero weight decay.

**Data augmentation as regularization.** Random crops, flips, mix-ups, and adversarial perturbations all impose smoothness or invariance on the function the model can express — without changing the loss. Often the highest-leverage regularization in practice, especially for vision.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
import torch.nn.functional as F

# Mixup — strong implicit regularizer for classification
def mixup(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    x_m = lam * x + (1 - lam) * x[idx]
    return x_m, y, y[idx], lam

def mixup_loss(logits, y_a, y_b, lam):
    return lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)

# Label smoothing — keeps the model from collapsing on a single class
loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

# Stochastic depth — drop whole residual blocks at random
class StochasticDepth(nn.Module):
    def __init__(self, p): super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p: return x
        return torch.zeros_like(x)
```

</div>

<div class="level-next">
<span>Want the path algorithm, MDL, and PAC-Bayes?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">PAC-Bayes generalization bound</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathbb{E}_{\theta \sim Q}\!\left[\, R(\theta)\right] \;\leq\; \mathbb{E}_{\theta \sim Q}\!\left[\, \hat R(\theta)\right] \;+\; \sqrt{\frac{D_\text{KL}(Q \,\|\, P) + \ln(1/\delta)}{2n}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`R(θ)`true (test) risk of parameters *θ*

</li>
<li markdown="1">

`R̂(θ)`empirical (training) risk

</li>
<li markdown="1">

`P`prior over parameters; `Q` posterior after training

</li>
<li markdown="1">

`D_KL`KL divergence — measures how far *Q* has moved from *P*

</li>
<li markdown="1">

`n`number of training samples; `δ` confidence level

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{avg true error} \;\leq\; \text{avg training error} \;+\; \sqrt{\frac{\text{distance from prior} \;+\; \ln(1/\delta)}{2 \times n}} $$</span>

**In words.** This is an *upper bound* on how bad the test error can be — the `≤` sign says the true error is at most the right-hand side. The first piece is just how well you did on the training set. The second piece — under the `√` (square root) — is a penalty that grows when your learnt weights have moved far from your starting prior, and shrinks as you collect more data (the `n` in the denominator is the number of training samples). `ln` is the natural logarithm, a number that grows very slowly with its input; `δ` is a small confidence-level number you pick.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`avg true error`average error on unseen data, averaged over the distribution of learnt weights

</li>
<li markdown="1">

`avg training error`average error on the training data, same averaging

</li>
<li markdown="1">

`distance from prior`how far the learnt weight distribution has moved from your starting assumption (the prior)

</li>
<li markdown="1">

`n`number of training samples; `δ` a small confidence-level number

</li>
<li markdown="1">

The take-away: staying close to your prior ≈ generalizing well ≈ regularizing

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Regularization path.** The set of solutions as λ varies from 0 to ∞. For LASSO, the path is piecewise linear in λ — you can compute the whole path in one pass (LARS, Efron et al. 2004). For ridge, the solution is a closed-form function of λ — useful for fast cross-validation across many λ values.

**Group LASSO & structured sparsity.** Penalize the L2 norm of *groups* of coefficients, then the L1 norm across groups. Forces entire groups to be zero or active together. Useful when features come in natural blocks (e.g., one-hot encoded categories, time-series lags).

**Spectral norm regularization.** Constrain the largest singular value of each weight matrix — useful for stable adversarial training, GAN discriminators (Miyato et al. 2018 — spectral normalization), and Lipschitz-bounded networks for verified robustness.

**Sharpness-Aware Minimization (SAM).** Foret et al. (2021) explicitly seek flat minima. Equivalent to penalising the gradient norm in a neighbourhood. Reliably improves generalization, especially on smaller models or vision tasks.

**Implicit regularization of SGD.** The noise in stochastic gradient descent acts like a diffusion process — it preferentially picks "flat" minima (low Hessian eigenvalues). This is one explanation for why SGD-trained networks sometimes generalize better than Adam-trained ones, even at higher training loss.

**MDL & information bottleneck.** Minimum Description Length frames regularization as compression — a model is a good fit if (model + residuals) describes the data in fewer bits than the raw data. The Information Bottleneck (Tishby) extends this: compress the input while preserving information about the target.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from torch.nn.utils import spectral_norm

# Spectral normalization — bounds the Lipschitz constant of each layer
discriminator = torch.nn.Sequential(
    spectral_norm(torch.nn.Conv2d(3, 64, 4, 2, 1)),
    torch.nn.LeakyReLU(0.2),
    spectral_norm(torch.nn.Conv2d(64, 128, 4, 2, 1)),
    torch.nn.LeakyReLU(0.2),
)

# Group LASSO via proximal gradient — penalise blocks of weights together
def group_lasso_step(weights, lr, lam, groups):
    # groups is a list of index sets that should shrink together
    with torch.no_grad():
        for g in groups:
            w_g = weights[g]
            norm = w_g.norm()
            shrink = max(1 - lr * lam / (norm + 1e-12), 0.0)
            weights[g] = w_g * shrink
```

</div>

<div class="level-next">
<span>Want the picture instead?</span>
<button data-go-to="intuition" type="button">← Back to Intuition</button>
</div>

</section>

<!-- TOPIC SIDEBAR -->

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="fundamentals" markdown="1">

[Elements of Statistical Learning — Ch. 3 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">The classical reference for L1, L2, elastic net, and the regularization path. Geometric intuition for why L1 produces sparsity.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Srivastava et al. (2014) — Dropout <i class="fas fa-external-link-alt"></i>](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf){: target="_blank" }
<span class="annotation">The original dropout paper. Still readable; gives the ensemble interpretation and many empirical examples.</span>

</li>
<li data-tier="indepth" markdown="1">

[Loshchilov & Hutter (2019) — AdamW <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1711.05101){: target="_blank" }
<span class="annotation">Why weight decay in Adam was wrong, and how AdamW fixes it. Important for any deep-learning regularization decision.</span>

</li>
<li data-tier="indepth" markdown="1">

[Zhang et al. (2017) — Understanding deep learning requires rethinking generalization <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1710.10903){: target="_blank" }
<span class="annotation">The paper showing that deep networks fit random labels perfectly — and yet still generalize well on real data. Forced the field to take implicit regularization seriously.</span>

</li>
</ul>

</div>

---
title: Gradient Descent — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: {{root}}theory.html
heading: Gradient Descent
lead: Iteratively step in the direction that decreases the loss. The algorithm at the heart of training almost every modern model.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Step in the direction that decreases the loss. Keep stepping.** You're standing on a hilly landscape in fog. You can feel which way is downhill at your feet. You step that way. Repeat until the ground is flat. The "landscape" is the loss as a function of model parameters; gradient descent is the foot-feeling-its-way algorithm.

</div>

<div class="viz-embed">
<div class="viz-embed-header">
<span class="viz-embed-title">Click anywhere on the surface to start a new descent · adjust <kbd>η</kbd> to change step size</span>
<div class="viz-embed-controls">
<label style="display: inline-flex; align-items: center; gap: 0.4rem; color: var(--muted); font-size: 0.85rem;">
                    η <input type="range" id="viz-gd-lr" min="0" max="100" value="40" style="width: 100px;"></input>
</label>
<button id="viz-gd-play" type="button">Pause</button>
<button id="viz-gd-reset" type="button">Reset</button>
</div>
</div>
<canvas id="viz-gd-canvas"></canvas>
<div class="viz-embed-stats" id="viz-gd-stats"></div>
</div>

<script src="{{root}}js/viz/gradient-descent.js"></script>

The surface above is a scaled *Rosenbrock* function — minimum at the white marker at (1, 1), inside a curved banana-shaped valley. Try clicking different starting points: the path along the valley floor is slow even at well-chosen step sizes. Crank η up: the trajectory starts oscillating or diverges. This is exactly why momentum and adaptive optimisers exist.
{: .viz-intro }

<div class="viz-grid">
<a href="https://distill.pub/2017/momentum/" target="_blank" class="viz-card viz-card-featured">
<span class="viz-source">distill.pub</span>
<h3>Why Momentum Really Works</h3>
<p>The single best deep dive on momentum, with interactive 2D loss surfaces. Drag points, change β, watch how momentum reshapes the trajectory. Read once and you'll never use vanilla SGD again.</p>
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
<a href="https://www.benfrederickson.com/numerical-optimization/" target="_blank" class="viz-card">
<span class="viz-source">benfrederickson.com</span>
<h3>An Interactive Tutorial on Numerical Optimization</h3>
<p>Side-by-side comparison of gradient descent, momentum, Adagrad, Adam — all running live on the same loss surface. Best for seeing the relative behaviour of different optimisers.</p>
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
<a href="https://ruder.io/optimizing-gradient-descent/" target="_blank" class="viz-card">
<span class="viz-source">ruder.io</span>
<h3>Ruder — Optimizing Gradient Descent</h3>
<p>Comprehensive written overview of every SGD variant — momentum, Adagrad, RMSprop, Adam, AdamW. Static but the clearest single reference for "which optimiser should I use".</p>
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
<a href="https://playground.tensorflow.org/" target="_blank" class="viz-card">
<span class="viz-source">tensorflow.org</span>
<h3>TensorFlow Playground</h3>
<p>Train a small neural network in your browser. Watch the loss curve drop while you tweak learning rate, batch size, and activations. Best for seeing GD work on a real classification problem.</p>
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
</div>

<article class="tldr-body" markdown="1">

Every weight in a neural network — billions of them in modern models — is tuned by gradient descent. So is logistic regression's β, linear regression's β (it has a closed form too, but GD also works), SVM's *w* via SGD, and almost every other "trainable" model.

The recipe is short: compute the gradient (how loss changes if you nudge each parameter), subtract a small fraction of it from each parameter, repeat. The "small fraction" is the **learning rate** — too small and you'll be there forever; too large and you'll bounce around or diverge.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Training any neural network (no real alternative)
- Large datasets where closed-form solutions are too expensive
- Streaming / online learning
- Any model where the loss is differentiable

</div>

<div class="no" markdown="1">

### Skip it when

- The problem has a closed-form solution AND the data fits (e.g. linear regression on a small dataset)
- Loss is not differentiable (use coordinate descent, EM, or specialized methods)
- You need a global guarantee — GD only finds local minima for non-convex problems
- Discrete optimization (combinatorial, integer programming)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Tiny gradient descent for f(x) = (x - 3)^2.  Minimum at x = 3.
x  = 0.0
lr = 0.1
for step in range(50):
    grad = 2 * (x - 3)
    x   -= lr * grad
print(x)   # → ~3.0
```

</div>

<div class="level-next">
<span>Want the formula and the optimizer zoo?</span>
<button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The update rule</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t \;-\; \eta \,\nabla_{\!\boldsymbol{\theta}}\, \mathcal{L}(\boldsymbol{\theta}_t) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`θ`parameters being optimised (weights)

</li>
<li markdown="1">

`η`learning rate — step size

</li>
<li markdown="1">

`∇L`gradient of the loss with respect to *θ*

</li>
<li markdown="1">

`θt+1`parameters after step *t+1*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{new weights} \;=\; \text{old weights} \;-\; \eta \times \text{slope of loss at old weights} $$</span>

**In words.** At each step, you nudge the weights a little bit in the opposite direction of the slope of the loss — "opposite" because the slope points uphill and you want to go downhill. `η` (eta, the *learning rate*) sets how big each nudge is: too small and you crawl, too big and you overshoot or oscillate. The "slope" here is technically a vector of slopes — one per weight — usually written `∇L` (the gradient). You repeat this update until the slope is roughly zero (you're at the bottom of the valley).
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Batch vs SGD vs mini-batch.** Full-batch GD computes the gradient over the entire dataset every step — exact but expensive. *Stochastic gradient descent* (SGD) uses just one example per step — noisy but cheap. *Mini-batch SGD* uses ~32–256 examples — the practical default. The noise from sampling is what makes SGD generalise: it can escape sharp local minima that hurt held-out performance.

**Learning rate.** The single most important hyperparameter. Too small → painfully slow convergence. Too large → divergence or oscillation. Try 1e-3 with Adam, 1e-1 with SGD for a starting point. Use a learning-rate *schedule* (decay, cosine, warmup) for serious training.

**Momentum.** Accumulate a moving average of past gradients: *v ← βv + ∇L*, then *θ ← θ − η·v*. This "rolls through" small oscillations and accelerates along consistent directions. Classical momentum: *β ≈ 0.9*.

**Adam.** Per-parameter adaptive learning rate. Tracks both the first moment (momentum) and second moment (gradient variance) of recent gradients, then normalises. Works well out-of-the-box on a huge range of problems and is the default optimiser in most deep-learning frameworks.

**AdamW.** Adam + decoupled weight decay. Strictly better than Adam in most settings — used by every modern LLM trainer. Reach for AdamW unless you have a specific reason not to.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Training any deep model (AdamW with a warmup-cosine schedule is the modern default)
- Online or streaming learning with very large data
- Fine-tuning pretrained models
- You can afford hyperparameter tuning on at least the learning rate

</div>

<div class="no" markdown="1">

### Skip it when

- Closed form exists and is cheap to compute (OLS regression, ridge regression with small *p*)
- Non-differentiable losses (use coordinate descent or specialised solvers)
- Tiny problems where Newton's method converges in <10 iterations
- You need provable global optimality on non-convex problems

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

model = ...  # your network
opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(opt, T_max=epochs)

for epoch in range(epochs):
    for xb, yb in loader:
        loss = criterion(model(xb), yb)
        opt.zero_grad()
        loss.backward()           # autodiff fills in .grad on each parameter
        opt.step()                # θ ← θ − η · update_rule(grad)
    scheduler.step()
```

</div>

<div class="level-next">
<span>Want convergence rates, conditioning, and second-order methods?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Convergence (convex, L-smooth)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}(\boldsymbol{\theta}_T) - \mathcal{L}(\boldsymbol{\theta}^*) \;\le\; \frac{\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2\eta T} \quad \text{for}\; \eta \le 1/L $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`θ*`the optimal (loss-minimising) parameters

</li>
<li markdown="1">

`θT`parameters after *T* steps; `θ0` the starting point

</li>
<li markdown="1">

`‖·‖²`squared Euclidean distance

</li>
<li markdown="1">

`L`Lipschitz constant of the gradient — bounds curvature

</li>
<li markdown="1">

Convergence rate *O(1/T)* — slow, but better than nothing on non-strongly-convex problems

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ (\text{loss after } T \text{ steps}) - (\text{best possible loss}) \;\le\; \frac{(\text{starting distance from optimum})^2}{2 \times \eta \times T} $$</span>

**In words.** This is an *upper bound* on how far you can still be from the best possible loss after *T* gradient steps. The `≤` means "at most". The right-hand side shrinks as *T* grows — so more steps give a tighter guarantee. The "starting distance from optimum" is just how far apart your initial weights and the optimal weights are (Pythagoras-style, written `‖θ₀ − θ*‖²`). The condition `η ≤ 1/L` says your step size has to be small enough relative to how curvy the loss can get — pick *η* too big and the bound stops holding.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Convex landscapes.** If the loss is convex and *L*-smooth, vanilla GD converges as *O(1/T)*; strongly convex losses converge as *O(c<sup>T</sup>)* for some *c < 1*. Nesterov's accelerated gradient achieves *O(1/T²)* on convex losses — optimal among first-order methods. Practical SGD doesn't quite hit these rates because of noise but tracks them up to a noise floor.

**Conditioning.** The hardness of GD depends on the condition number *κ = λ<sub>max</sub>/λ<sub>min</sub>* of the Hessian — the ratio of biggest to smallest curvature. High *κ* = stretched valley = SGD zigzags. Mitigations: preconditioning (Adam approximates this), batch normalisation (keeps activations on similar scales), proper initialisation.

**Non-convex.** Neural network loss landscapes are non-convex with many local minima and saddle points. Surprisingly, SGD reliably finds *good* minima. Why: (1) overparameterised networks have many global minima; (2) SGD's noise prefers *flat* minima, which generalise better than sharp ones (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017).

**Second-order methods.** Newton's method uses the Hessian: *θ ← θ − H<sup>−1</sup>∇L*. Converges in fewer iterations but each iteration is *O(p³)* — infeasible for deep networks. Approximations: L-BFGS (limited-memory quasi-Newton, good for ≤ millions of params), K-FAC (Kronecker-factored Fisher), Shampoo (block-diagonal preconditioner). Rarely beat AdamW in practice at scale, but seeing some renewed interest.

**Schedules.** Warmup → cosine decay is the modern default for large-batch training. Linear warmup of *η* for ~1000 steps avoids early divergence on large models; cosine decay smoothly anneals toward zero. Cyclic LR (Smith, 2017) and 1cycle schedules also work well in many regimes.

**Per-layer LR.** Different parameter groups often want different learning rates — biases vs. weights, transformer attention vs. MLP, classifier head vs. backbone during fine-tuning. PyTorch makes this trivial with parameter groups; the gains can be substantial.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Large-batch training — proper warmup-cosine schedules are essential
- Distributed training — synchronous SGD with LARS / LAMB for very large batches
- Long-context LLM training — gradient clipping + cosine decay is the recipe
- Fine-tuning — per-parameter-group learning rates matter

</div>

<div class="no" markdown="1">

### Skip it when

- Problem is small enough for closed-form or quasi-Newton
- You need exact gradients of a non-differentiable operator
- Reinforcement learning with very high-variance gradient estimates — use natural gradient or TRPO-style methods
- Combinatorial / discrete optimisation

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Per-parameter-group setup: lower LR for backbone, higher for classifier head
opt = AdamW([
    {"params": model.backbone.parameters(),   "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-4)

# Warmup → cosine decay
import math
def lr_lambda(step, warmup=1000, total=100_000):
    if step < warmup:
        return step / warmup
    p = (step - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))

scheduler = LambdaLR(opt, lr_lambda)

for step, (xb, yb) in enumerate(loader):
    loss = criterion(model(xb), yb)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    scheduler.step()
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
<li data-tier="intuition" markdown="1">

[Distill — Why Momentum Really Works <i class="fas fa-external-link-alt"></i>](https://distill.pub/2017/momentum/){: target="_blank" }
<span class="annotation">Interactive deep dive on momentum specifically. Essential reading once you've understood vanilla SGD.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Ruder — Optimizing Gradient Descent <i class="fas fa-external-link-alt"></i>](https://ruder.io/optimizing-gradient-descent/){: target="_blank" }
<span class="annotation">Comprehensive survey of GD variants — momentum, Adagrad, Adam, etc. The reference for "which optimiser should I use".</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Goodfellow et al. — Deep Learning ch. 8 <i class="fas fa-external-link-alt"></i>](https://www.deeplearningbook.org/contents/optimization.html){: target="_blank" }
<span class="annotation">Textbook treatment of optimisation in the deep-learning context. Free HTML. Read after Ruder for the full picture.</span>

</li>
<li data-tier="indepth" markdown="1">

[Loshchilov & Hutter (2017) — AdamW <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1711.05101){: target="_blank" }
<span class="annotation">The paper introducing decoupled weight decay. Short, important — modern LLM training all uses AdamW.</span>

</li>
</ul>

</div>

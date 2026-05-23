---
title: Optimization Algorithms — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Optimization Algorithms
lead: How gradient descent grew up — momentum, adaptive learning rates, and the modern toolkit.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Vanilla SGD is rarely the fastest.** Real loss surfaces have narrow ravines, saddle points, and curvature that changes from one direction to another. Modern optimizers solve these in different ways: *momentum* remembers past direction, *RMSprop / Adagrad* scale each parameter by its own gradient history, *Adam* combines both. The viz below races them on the same start.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Same start, same step budget — watch four optimizers race across a curved loss surface</span>
</div>
<div class="viz-classic-controls">
<button id="viz-opt-step" type="button">Step</button>
<button id="viz-opt-play" type="button">Play</button>
<button id="viz-opt-reset" type="button">Reset</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Landscape
                <select id="viz-opt-land"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                lr
                <input id="viz-opt-lr" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-opt-lr-lbl">lr = 0.05</span>
<span class="viz-classic-badge" id="viz-opt-step-lbl">step 0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-opt-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-opt-caption"></div>
</div>

<script src="{{root}}js/viz/optimizers.js"></script>

All four optimizers see the same loss surface from the same start. **SGD** moves straight down the local gradient and zig-zags in narrow valleys. **Momentum** accumulates velocity and "rolls through" valleys. **RMSprop** shrinks per-parameter steps where gradients are big, freeing it to take bigger steps where they're small. **Adam** combines momentum with RMSprop's adaptive scaling — usually the safest first pick.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**SGD.** Update is θ ← θ − η·∇L. Cheap, well-understood, often the best generalization in deep learning — but slow to converge on ill-conditioned problems where one direction needs huge steps and another tiny ones.

**Momentum.** Keep a running velocity that smooths gradient noise and accelerates along consistent directions. Nesterov momentum looks one step ahead before computing the gradient, which gives a small but real speedup.

**Adam.** Maintain per-parameter running estimates of the gradient mean and squared mean; scale each step by mean / √(second moment). Essentially momentum × RMSprop. The default for most deep learning today, though SGD-with-momentum sometimes generalizes better on vision tasks.

Practical truth: the learning rate matters more than the optimizer choice. A well-tuned SGD often beats a poorly-tuned Adam. Always sweep the learning rate first.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Adam (and friends)

- Default for transformer / NLP training
- You don't have time to tune lr per layer / per phase
- Sparse gradients (embeddings, NLP) — Adam handles them well
- RNN / LSTM training is much friendlier with Adam

</div>

<div class="no" markdown="1">

### SGD-with-momentum

- Image classification, where it generalizes a little better
- You've tuned the learning rate schedule carefully
- Reproducing classical results (most ResNet papers use it)
- You want the simplest possible thing in the loop

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# SGD with momentum — the deep-learning workhorse
opt_sgd  = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

# Adam — the safe default everywhere else
opt_adam = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate matters more than optimizer. Always have a schedule:
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=100)
```

</div>

<div class="level-next">
<span>Want the math: momentum, Adam, second-order, line search?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Adam update</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \begin{aligned} m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\ v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\ \hat m_t &= m_t / (1 - \beta_1^t), \quad \hat v_t = v_t / (1 - \beta_2^t) \\ \theta_{t+1} &= \theta_t - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon) \end{aligned} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`gt`gradient at step *t*

</li>
<li markdown="1">

`m`EMA of the gradient (momentum)

</li>
<li markdown="1">

`v`EMA of squared gradients (per-parameter scale)

</li>
<li markdown="1">

`β1, β2`decay rates for the two EMAs (typically 0.9 and 0.999)

</li>
<li markdown="1">

`η, ε`learning rate and a tiny constant to avoid division by zero

</li>
<li markdown="1">

Bias correction *1 − β<sup>t</sup>* matters early in training when both EMAs start at zero

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \begin{aligned} \text{velocity} &= \beta_1 \times \text{velocity} + (1-\beta_1) \times \text{gradient} \\ \text{scale}  &= \beta_2 \times \text{scale} + (1-\beta_2) \times (\text{gradient})^2 \\ \text{new weights} &= \text{old weights} - \eta \,\times\, \frac{\text{bias-corrected velocity}}{\sqrt{\text{bias-corrected scale}} + \epsilon} \end{aligned} $$</span>

**In words.** Adam keeps two running averages of recent gradients. The first, **velocity**, smooths the gradient itself — that's the momentum piece. The second, **scale**, averages the *squared* gradient — telling you how big the gradient has typically been on each parameter. Each step divides the velocity by the square root of the scale, so parameters with consistently large gradients take small steps and parameters with tiny gradients take big ones. The `β` values control how much history each average retains (commonly 0.9 and 0.999). The "bias correction" division by `1 − βt` fixes a small bias from initialising both averages at zero — it matters most in the first few hundred steps.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`gradient`current gradient of the loss

</li>
<li markdown="1">

`velocity`running average of recent gradients (momentum)

</li>
<li markdown="1">

`scale`running average of recent *squared* gradients (per-parameter typical magnitude)

</li>
<li markdown="1">

`β1, β2`how much memory each running average keeps (commonly 0.9 and 0.999)

</li>
<li markdown="1">

`η`learning rate; `ε` tiny constant that prevents divide-by-zero

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Momentum, properly.** The velocity update *v ← β v + g* is an exponentially-weighted average of past gradients with time-constant ≈ 1/(1−β). Past gradients in directions that point the same way reinforce; gradients that cancel out are smoothed. This is exactly why momentum helps in narrow ravines — the consistent ravine-floor direction gets amplified, while the bouncing perpendicular component cancels.

**Adagrad → RMSprop → Adam.** Adagrad scales each parameter's step by 1/√(Σ g²) — but the sum grows monotonically, killing learning eventually. RMSprop replaces the sum with an EMA, fixing the decay-to-zero problem. Adam adds momentum on top. AdamW decouples weight decay from the gradient (Loshchilov & Hutter, 2019) — for transformers this is a meaningful improvement.

**Learning rate schedules.** The single highest-leverage knob is the lr schedule, not the optimizer. Warm-up (linear ramp from 0 to lr over a few hundred steps) is essential for transformers — Adam's bias correction misbehaves at step 1. Cosine decay is the modern standard for the rest. One-cycle and triangular schedules (Smith 2017) are useful when you want a quick convergence.

**Second-order methods.** Newton's method scales steps by the inverse Hessian — works wonderfully in low dimensions, intractable in high. Practical approximations include K-FAC (block-diagonal Fisher), Shampoo (Kronecker factors of the Hessian), and L-BFGS (history-based quasi-Newton). Useful for fine-tuning or small models; rarely beat tuned Adam at scale.

**Implicit regularization.** Different optimizers reach different solutions on the same loss surface. SGD's gradient noise has been argued to act as a regularizer that finds "flatter" minima — this is part of why it sometimes generalizes better than Adam, despite Adam reaching a lower training loss.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# Linear warm-up + cosine decay — the modern transformer recipe
def warm_cos_lr(step, warmup, total, lr_max):
    if step < warmup:
        return lr_max * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * lr_max * (1 + math.cos(math.pi * p))

# Gradient clipping — almost always worth it for transformers / RNNs
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# AdamW: weight decay should NOT pass through Adam's denominator
opt = torch.optim.AdamW(model.parameters(), lr=lr,
                        betas=(0.9, 0.95), weight_decay=0.1)
```

</div>

<div class="level-next">
<span>Want K-FAC, Shampoo, sharpness-aware minimization, and natural gradients?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Natural gradient</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \theta_{t+1} = \theta_t - \eta\, F(\theta_t)^{-1} \nabla \mathcal{L}(\theta_t) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`θ`model parameters

</li>
<li markdown="1">

`∇L`gradient of the loss

</li>
<li markdown="1">

`F`Fisher information matrix — the Hessian of the KL between distributions

</li>
<li markdown="1">

`F−1`inverse Fisher; rescales the gradient by local curvature in distribution space

</li>
<li markdown="1">

Steps in *distribution* space — reparameterization-invariant

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{new weights} \;=\; \text{old weights} \;-\; \eta \,\times\, (\text{curvature in distribution space})^{-1} \,\times\, \text{gradient of loss} $$</span>

**In words.** The natural gradient is "regular gradient descent, but measure distance the right way". Ordinary SGD treats every parameter as if a unit step in parameter space were equally meaningful. The **Fisher information matrix** `F` measures how much the model's output distribution actually *changes* when each parameter moves. Multiplying the gradient by its inverse rescales the step so the same step always produces a similar amount of change in the output distribution. The result is reparameterization-invariant: if you rewrote your weights in different units, the trajectory would be the same. In practice `F` is huge, so methods like K-FAC and Shampoo approximate it cheaply.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`old weights, new weights`parameters before and after the step

</li>
<li markdown="1">

`gradient of loss`same gradient SGD would use

</li>
<li markdown="1">

`curvature in distribution space`Fisher information — how much the output distribution changes when each parameter moves

</li>
<li markdown="1">

`(·)−1`inverse — undoes that curvature, so each step produces a controlled change in output

</li>
<li markdown="1">

Reparameterization-invariant: rescaling the parameter axes doesn't change the trajectory

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**K-FAC.** Kronecker-Factored Approximate Curvature (Martens & Grosse, 2015) approximates the Fisher as a block-diagonal Kronecker product per layer. Practical for medium-sized networks; offers real wall-clock speedup for some workloads but adds substantial implementation complexity.

**Shampoo.** Anil et al. (2020) — second-order preconditioning that maintains Kronecker factors of the gradient covariance and applies their inverse 1/4 root. Slower per step than Adam but converges in fewer steps; recent variants are competitive on real-world training.

**SAM — Sharpness-Aware Minimization.** Foret et al. (2021) — explicitly look for parameters that minimize loss *and* have small gradient norm in their neighbourhood (flat minima). Adds a small inner-loop "find the worst direction nearby" step. Reliably improves generalization on vision tasks; ~2× the compute cost.

**Lion / Sophia / Schedule-Free.** Three recent optimizers from Chen et al. (2023), Liu et al. (2024), and Defazio et al. (2024) respectively. Lion uses sign-of-momentum updates; Sophia uses a clipped second-order term; Schedule-Free does away with the lr schedule entirely. Each shows wins on some benchmarks but hasn't displaced AdamW as the default.

**Why SGD generalizes better.** Empirically, SGD-with-momentum often finds "wider" minima than Adam — measured by Hessian eigenvalues at the solution. There are theoretical arguments (gradient noise as injected SDE) and counter-arguments. The practical implication is unchanged: try AdamW first, and for vision benchmarks try SGD-with-momentum as a comparison.

**Trust region methods.** TRPO and PPO (in RL) use a KL constraint instead of a learning rate — they implicitly act like a natural gradient step. Lessons from those have leaked back into supervised learning (e.g., the "trust region for gradient steps" view of Adam's denominator).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# Sharpness-Aware Minimization in 10 lines
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_opt, rho=0.05, **kwargs):
        self.base = base_opt(params, **kwargs)
        defaults = dict(rho=rho, **self.base.defaults)
        super().__init__(self.base.param_groups, defaults)

    @torch.no_grad()
    def first_step(self):
        norm = torch.norm(torch.stack([p.grad.norm() for g in self.param_groups
                                       for p in g["params"] if p.grad is not None]))
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w); self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if "e_w" in self.state[p]: p.sub_(self.state[p]["e_w"])
        self.base.step()

# Usage:
# loss.backward(); opt.first_step()
# loss = compute_loss(); loss.backward(); opt.second_step()
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
<li data-tier="intuition" markdown="1">

[Distill — Why Momentum Really Works <i class="fas fa-external-link-alt"></i>](https://distill.pub/2017/momentum/){: target="_blank" }
<span class="annotation">Goh's beautifully visual essay on momentum as a damped oscillator. Easily the best introduction to the geometry of optimization.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Kingma & Ba (2015) — Adam <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1412.6980){: target="_blank" }
<span class="annotation">The original Adam paper. Short and readable; sections 2–4 are essential.</span>

</li>
<li data-tier="indepth" markdown="1">

[Loshchilov & Hutter (2019) — AdamW <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1711.05101){: target="_blank" }
<span class="annotation">"Decoupled Weight Decay Regularization". The paper that pointed out Adam's weight-decay implementation was wrong and gave us AdamW.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Deep Learning Book — Optimization Ch. 8 <i class="fas fa-external-link-alt"></i>](https://www.deeplearningbook.org/contents/optimization.html){: target="_blank" }
<span class="annotation">Comprehensive chapter on practical optimization including learning rate schedules, batch normalization's interaction with optimizers, and more.</span>

</li>
</ul>

</div>

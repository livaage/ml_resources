---
title: Training Debugging — ML Resources Hub
eyebrow_text: ← Engineering · Debugging &amp; Profiling
eyebrow_href: {{root}}engineering.html
heading: Training Debugging
lead: NaNs, exploding gradients, divergent loss — and the small checklist that catches 95% of training pathologies.
active_nav: engineering
prev_href: model-registries.html
prev_title: Model Registries
next_href: loss-curve-forensics.html
next_title: Loss Curve Forensics
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Most training failures fall into a small set of diagnosable categories.** Loss is NaN → numerical instability. Loss diverges → lr too high or bad init. Loss plateaus → gradient flow stopped. Loss looks fine but val is bad → overfitting / leakage / domain shift. Each has a small set of likely causes and a standard checklist.

</div>

<article class="tldr-body" markdown="1">

**The 5-minute checklist for any training run that's misbehaving.** (1) Drop lr by 10× — does the divergence stop? (2) Add gradient clipping at 1.0 — does it stabilise? (3) Try fp32 (off mixed-precision) — does the NaN disappear? (4) Take 2 examples and try to overfit — does loss go to ~0? (5) Check the data with print statements — does anything look wrong (NaNs, weird scales, wrong labels)?

**Most common causes by symptom.** NaN in step 1: bad init or bad data. NaN after step N: explosion (use clipping) or numerical issue in a specific layer. Loss won't decrease at all: gradients aren't flowing (detached graph, frozen layer, wrong optimizer setup). Loss decreases but val is awful: overfitting, leakage, or distribution shift.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### The standard checks

- Drop lr 10× — fixes 80% of divergence
- Add `clip_grad_norm_(1.0)` — fixes the rest
- Try fp32 to rule out mixed-precision issues
- Overfit a tiny batch — confirms the loop is correct
- Inspect 5 random batches — does the data look sane?
- Check parameter / activation stats — anything saturated?

</div>

<div class="no" markdown="1">

### What to NOT do first

- Switch architecture — fix the loop bugs first
- Add more regularization to fix divergence
- Tune hyperparameters before confirming the baseline works
- Conclude the model "doesn't work" without finishing the checklist

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

def diagnose_training(model, x, y, loss_fn, opt, n_steps=200):
    """Five-minute sanity check on a tiny batch."""
    x, y = x[:2], y[:2]                          # 2 examples
    loss_history = []
    for step in range(n_steps):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        if torch.isnan(loss):
            print(f"NaN at step {step}")
            return loss_history
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_history.append(loss.item())
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f} gnorm={gnorm:.4f}")
    if loss_history[-1] > 0.1:
        print("Failed to overfit 2 examples — check the loop, loss, or model")
    return loss_history
```

</div>

<div class="level-next">
<span>Want NaN forensics, gradient hooks, & mixed-precision pitfalls?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Why training diverges</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \Delta\theta = -\eta \nabla L \;\Rightarrow\; |\nabla L| \gg \tfrac{1}{\eta} \;\Rightarrow\; \text{overshoot} \;\Rightarrow\; \text{larger gradient} \;\Rightarrow\; \text{NaN} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Positive feedback loop between gradient magnitude and loss

</li>
<li markdown="1">

Cutoff (clipping) breaks the loop; lower lr makes it less likely

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{step} = -\eta \times \text{slope} \;\Rightarrow\; \text{huge slope} \;\Rightarrow\; \text{overshoot} \;\Rightarrow\; \text{even huger slope} \;\Rightarrow\; \text{NaN} $$</span>

**In words.** A divergence chain reaction. Each `⇒` reads as "leads to". The weight change `Δθ` ("delta theta") is just the learning rate `η` (eta) times the negative slope of the loss `∇L` ("nabla L" = gradient of L). When the slope gets so big that `η × slope` overshoots the valley, the new position is worse, which has an even bigger slope, and so on — gradients explode and eventually overflow to NaN (not-a-number). Two ways out: cap the gradient magnitude (clipping), or shrink `η` so each step is smaller.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`step`change in weights this iteration (`Δθ`)

</li>
<li markdown="1">

`η`learning rate — how big each step is

</li>
<li markdown="1">

`slope`gradient of the loss with respect to the weights (`∇L`)

</li>
<li markdown="1">

Positive feedback loop between gradient magnitude and loss

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Anomaly detection.** `torch.autograd.set_detect_anomaly(True)` tracks where in the backward pass a NaN originated. Slow — use for one debug run.

**Layer-by-layer hooks.** Attach a forward hook to every layer; check for NaN / Inf in the output. The first layer that produces NaN is the culprit. Common: division by zero in normalisation, log(0), exp(huge) in softmax, fp16 underflow in attention.

**Mixed-precision pitfalls.** Softmax over very large or very small logits → fp16 NaN. Loss scaling helps but isn't a silver bullet. `torch.amp.autocast` + `GradScaler` are the standard pattern; bf16 is more robust than fp16 if your hardware supports it.

**Gradient explosion.** Symptom: loss spikes then NaNs. Cause: a large activation produces a large gradient, which makes a large weight update, which makes a larger activation. Fix: gradient clipping (almost always < 1.0 for transformers / RNNs).

**Gradient vanishing.** Symptom: loss plateaus, gradient norms decay toward zero. Cause: deep networks with bad init, saturating activations (sigmoid, tanh), or accumulated multiplicative effects. Fix: residual connections, Layer Norm, He / Xavier init.

**Bad initialisation.** Weights too large → activations explode → gradients explode. Weights too small → activations vanish → no learning. PyTorch's defaults are usually OK; if you're rolling your own layers, use `nn.init.kaiming_normal_` or similar.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# Hook every layer; report the first layer to produce NaN
def install_nan_hooks(model):
    def make_hook(name):
        def hook(module, inp, out):
            tensors = out if isinstance(out, (list, tuple)) else (out,)
            for t in tensors:
                if isinstance(t, torch.Tensor) and not t.isfinite().all():
                    print(f"⚠  non-finite in {name}: "
                          f"NaN={t.isnan().any().item()} Inf={t.isinf().any().item()}")
                    return
        return hook
    for name, mod in model.named_modules():
        if name: mod.register_forward_hook(make_hook(name))

# Mixed precision the right way
scaler = torch.amp.GradScaler()
for x, y in loader:
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
```

</div>

<div class="level-next">
<span>Want distributed debugging, OOM forensics, & subtle correctness bugs?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Distributed mismatch</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{rank}_i.\theta_{t+1} \neq \text{rank}_j.\theta_{t+1} \;\Rightarrow\; \text{drift, divergence, or NaN} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Parameter sync is what keeps ranks consistent

</li>
<li markdown="1">

Any source of asymmetry — different seeds, different batch ordering, different num steps — causes drift

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{(weights on GPU } i\text{)} \neq \text{(weights on GPU } j\text{)} \;\Rightarrow\; \text{ranks drift, training breaks} $$</span>

**In words.** In distributed training, every GPU (called a "rank") is supposed to hold the same weights after each step. The subscripts `i` and `j` refer to two different ranks; `θt+1` is "weights after step t+1". The `≠` means "not equal" — when ranks diverge, you get noisy gradients, slow convergence, or eventually NaN. The cure is the all-reduce sync after every backward pass; the failure modes come from anything that breaks symmetry (different per-rank seeds, dropped DDP wrapper, hand-rolled gradient hooks that don't all-reduce).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`weights on GPU i / j`parameters held by different ranks in the distributed cluster

</li>
<li markdown="1">

`⇒`leads to (the divergence in weights causes the symptoms)

</li>
<li markdown="1">

Parameter sync is what keeps ranks consistent

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Distributed debugging.** First rule: reproduce on 1 GPU. ~80% of "distributed bugs" are actually single-GPU bugs that happen to surface in the distributed setting. When the bug genuinely needs > 1 GPU: log per-rank metrics, look for the rank where something diverges, then bisect on what's different.

**Gradient sync bugs.** Symptom: all ranks converge slowly or to a worse minimum than single-GPU. Causes: missing `DDP` wrapper on a model that gets cloned; ranks computing different losses (e.g., different normalisation across the batch); gradient hooks that don't allreduce.

**OOM forensics.** `torch.cuda.memory_summary()` shows allocations. Memory growing over time → leak (tensors held alive by some Python reference, often inside a logging dictionary). Memory high but stable → just need a smaller batch or more aggressive activation checkpointing.

**Subtle correctness bugs.** Detached graphs (gradient doesn't flow). Frozen layers (parameters that should train but don't). Off-by-one in masks. Wrong dimension for softmax. Reduction over the wrong axis. The 1-batch overfit catches most of these.

**Profiler + debugger.** When training is slow, profile. When it's wrong, debug. Mixing the two wastes time. PyTorch profiler with the `profile_memory=True` flag is the canonical tool for both.

**Catastrophic-forgetting variant.** Fine-tuning destroys earlier capability. Symptom: a model that was great on one task is now bad on it after training on another. Mitigations: lower lr, fewer epochs, freeze earlier layers, EWC-style regularization.

**The "git bisect" trick.** If a model used to train fine and now diverges, bisect through git history to find the commit that broke it. Faster than reading every diff.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.distributed as dist

# Per-rank sanity log
def log_per_rank(name, tensor):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank{rank}] {name}: mean={tensor.mean():.4f} std={tensor.std():.4f}")

# Detect: are all ranks computing the same loss?
def check_loss_equal(loss):
    if not dist.is_initialized(): return
    losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
    dist.all_gather(losses, loss)
    stacked = torch.stack(losses)
    if (stacked.max() - stacked.min()).abs() > 1e-3:
        print(f"⚠  losses differ across ranks: {[l.item() for l in losses]}")
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

[Karpathy — A Recipe for Training Neural Networks <i class="fas fa-external-link-alt"></i>](https://karpathy.github.io/2019/04/25/recipe/){: target="_blank" }
<span class="annotation">Still the best single source on debugging training. Read once a year.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyTorch — Numerical Accuracy Notes <i class="fas fa-external-link-alt"></i>](https://pytorch.org/docs/stable/notes/numerical_accuracy.html){: target="_blank" }
<span class="annotation">Official guide on fp16 / bf16 / fp32 trade-offs and the deterministic-ops flags.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyTorch — Autograd Anomaly Detection <i class="fas fa-external-link-alt"></i>](https://pytorch.org/docs/stable/notes/autograd.html#anomaly-detection){: target="_blank" }
<span class="annotation">The reference docs for <code>set_detect_anomaly</code> — your friend when chasing NaNs in backward.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyTorch Common Pitfalls <i class="fas fa-external-link-alt"></i>](https://github.com/pytorch/pytorch/wiki/PyTorch-Common-Pitfalls){: target="_blank" }
<span class="annotation">Community-maintained list of the bugs that bite everyone. Long and worth skimming.</span>

</li>
</ul>

</div>

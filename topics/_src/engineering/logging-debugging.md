---
title: Logging &amp; Debugging — ML Resources Hub
eyebrow_text: ← Engineering · Development Loop
eyebrow_href: {{root}}engineering.html
heading: Logging &amp; Debugging
lead: Structured logs, the right print statements, and which debugger tricks work on tensors.
active_nav: engineering
prev_href: testing-ml-code.html
prev_title: Testing ML Code
next_href: ci-for-ml.html
next_title: CI for ML
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**"NaN" is a clue, not a stack trace.** ML bugs usually present as silent degradation — bad metrics, divergent loss, slow training — not as crashes. The trick is making each level of the stack legible: structured logs you can search, useful prints at the right verbosity, a debugger that handles tensors.

</div>

<article class="tldr-body" markdown="1">

**Use real logging, not print.** Python's `logging` module — or better, `loguru` — gives you levels, timestamps, structured records, and rotation for free. The five-minute investment pays off forever.

**Log what matters.** Loss, learning rate, gradient norms, data shapes, the configuration object on startup. Not every batch — sample every N batches. If you're using a tracker (W&B / MLflow), it's also your log.

**Most ML bugs.** NaN / Inf propagation, shape mismatches (sometimes silently broadcasting), gradient explosion or vanishing, learning-rate too high, data corruption upstream of training, label noise, distribution shift between train and val.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from loguru import logger
import torch

# Structured logging with levels
logger.add("train.log", rotation="100 MB", level="INFO",
           format="{time} {level} {message}")

logger.info("config: {}", cfg)

for step in range(num_steps):
    loss, metrics = train_step()

    if torch.isnan(loss):
        logger.error("NaN at step {}", step)
        debug_dump(step)
        raise RuntimeError("NaN loss")

    if step % 100 == 0:
        gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        logger.info("step={} loss={:.4f} gnorm={:.4f}", step, loss.item(), gnorm)
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### What to watch in training

- **Loss**: smooth decrease; NaN spike = something exploded
- **Gradient norm**: stable; growing = lr too high; collapsing = vanishing
- **Activation stats**: most activations should be neither 0 nor saturated
- **Learning rate**: log it explicitly; schedule bugs are common
- **GPU utilization**: low util usually = data-loading bottleneck

</div>

<div class="no" markdown="1">

### Common silent failures

- Wrong data type (fp16 underflow, int truncation)
- Broadcasting where you didn't expect (e.g. (B,) vs (B, 1))
- Detached graph — gradients don't flow
- Frozen layers you forgot to unfreeze
- Wrong device — silent half-CPU half-GPU

</div>

</div>

<div class="level-next">
<span>Want anomaly detection, NaN sources, & debugging recipes?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The first-five-things-to-check checklist</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{loss} \xrightarrow{?} \text{NaN/diverge} \;\Rightarrow\; \text{lr}, \;\text{init}, \;\text{numerical stability}, \;\text{data}, \;\text{precision} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Drop the lr by 10×: does the divergence go away?

</li>
<li markdown="1">

Re-init: bad initialisation? deeper-net pathology?

</li>
<li markdown="1">

Add eps to denominators, clip gradients

</li>
<li markdown="1">

Check the data: outliers, label encoding, NaNs upstream

</li>
<li markdown="1">

Try fp32: an fp16 underflow / overflow is masking the problem

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss goes NaN or diverges} \;\Rightarrow\; \text{check: learning rate, initialisation, numerical stability, data, precision} $$</span>

**In words.** When training blows up — loss becomes NaN, or shoots to infinity — there are five usual suspects, in roughly this order of frequency. The `⇒` ("implies") in the math means "if the left side happens, investigate the right". **Learning rate** too high is by far the most common; try dropping it 10×. **Initialisation** issues cause deep networks to explode at step 0. **Numerical stability** covers things like log(0) or division by very small numbers. **Data** issues (NaN in inputs, outliers, wrong label encoding) are subtle. **Precision** (fp16 underflow) is increasingly common with mixed-precision training.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`NaN/diverge`loss becomes "not a number" or grows unboundedly

</li>
<li markdown="1">

`lr`learning rate — try cutting by 10×

</li>
<li markdown="1">

`init`weight initialisation scheme

</li>
<li markdown="1">

`numerical stability`add ε to denominators, log(x+ε), clip gradients

</li>
<li markdown="1">

`data`NaN / outliers / bad labels upstream

</li>
<li markdown="1">

`precision`fp16 / bf16 underflow or overflow — try fp32

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Anomaly detection during training.** `torch.autograd.set_detect_anomaly(True)` — slow, but tracks where a NaN originated. Use for one debug run; turn off for production.

**Gradient clipping.** `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. Almost always worth it for transformers and RNNs. The clip-then-NaN heuristic: if clipping fixes the divergence, you have an exploding gradient problem.

**NaN forensics.** Hook every layer's forward/backward to check for NaN; first layer that produces them is the culprit. Common sources: 1/0 in normalisation, log(0), exp(huge), softmax over identical logits with fp16.

**The 1-batch overfit.** Take 2–8 examples. The model should overfit perfectly in under 1000 steps. If it can't, something is fundamentally wrong (architecture, loss, data pipeline).

**Tensor-aware debugger.** pdb works fine; for tensor inspection use `display(tensor.shape, tensor.dtype, tensor.device, tensor.requires_grad)`. `ipdb` is nicer. PyCharm and VSCode have visual tensor inspectors.

**The pickle vs JSON rule.** Save configs, hyperparameters, and metrics as JSON or YAML — diff-friendly, language-agnostic. Save model weights as pickle / safetensors. Don't mix the two purposes.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn

# NaN-source finder: hook every module
def attach_nan_hooks(model):
    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                print(f"NaN in forward output of {name}")
        return fn
    for name, mod in model.named_modules():
        mod.register_forward_hook(hook(name))

# Print per-layer activation stats to spot saturating layers
def activation_summary(model, x):
    activations = {}
    def hook(name):
        return lambda m, i, o: activations.update({name: o.detach()})
    for name, mod in model.named_modules():
        mod.register_forward_hook(hook(name))
    model(x)
    for name, a in activations.items():
        print(f"{name:30s} mean={a.mean():.3f} std={a.std():.3f} "
              f"max={a.max():.3f} sat={(a.abs() > 5).float().mean():.2%}")
```

</div>

<div class="level-next">
<span>Want anomaly detection at scale, distributed debugging, & OOM forensics?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Loss curve patterns</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{divergence}, \;\text{plateau}, \;\text{cliff}, \;\text{oscillation}, \;\text{step decay} \;\to\; \text{each implies a specific bug class} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Each pattern has a small set of likely causes

</li>
<li markdown="1">

Documented troubleshooting tree → much faster than guessing

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \{\text{divergence},\; \text{plateau},\; \text{cliff},\; \text{oscillation},\; \text{step decay}\} \;\to\; \text{each pattern maps to a specific class of bug} $$</span>

**In words.** Loss curves come in a small zoo of recognisable shapes, and each shape narrows the search for what's wrong. **Divergence** (shooting up) usually means lr is too high or there's a numerical issue. **Plateau** from step 0 means gradients aren't flowing — something's detached or frozen. **Cliff** (sudden drop) is usually a schedule kick or finally finding the right answer. **Oscillation** means lr is too high or batch size is too small. **Step decay** is normal — just confirms your scheduler is working. The arrow `→` means "implies"; recognising the pattern compresses hours of debugging into minutes.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`divergence`loss shoots upward — usually lr or numerical issue

</li>
<li markdown="1">

`plateau`flat from start — gradient flow broken

</li>
<li markdown="1">

`cliff`sudden drop — schedule change or breakthrough

</li>
<li markdown="1">

`oscillation`noisy ups and downs — lr too high or batch too small

</li>
<li markdown="1">

`step decay`discrete drops matching the LR schedule — usually fine

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Loss curve forensics.** Diverges sharply: lr too high, bad init, or numerical issue. Plateaus immediately: gradient flow stopped (detach, frozen). Oscillates: lr too high or batch too small. Cliff (sudden drop): a learning rate schedule kick, or the model finally found the right answer.

**Distributed debugging.** Two ranks producing different losses on the same data → a sync or seeding issue. Use `torch.distributed.barrier()` + per-rank logging to find where they diverge. Always reproduce on a single GPU before debugging distributed; you'd be amazed how often a 1-GPU debug fixes the cluster bug.

**OOM (Out-of-Memory) forensics.** Memory usage growing over time → leak (something not getting freed; common with detached graphs kept alive). Plateaus high but stable → just need a smaller batch or more aggressive checkpointing. `torch.cuda.memory_summary()` shows allocation breakdown.

**Profiler-led debugging.** If training is slow, profile first. Common culprits: data loading (CPU bottleneck — increase `num_workers`), small ops (kernel launch overhead — fuse them), single-host sync (`torch.distributed.gather` blocking on the slowest rank).

**The "minimal reproducer" discipline.** Reproduce any non-trivial bug in < 30 lines of code with explicit seeds and data. Most ML bugs become 10× easier once isolated; many disappear on re-creation.

**Logging hygiene at scale.** Per-rank log files. JSON-lines format for parsing. Trace IDs across services. Sentry / Datadog / similar for alerting on real production failures. Cardinality matters — don't log per-example fields you'll have a billion of.

**Reproducible bug reports.** Save the config, the data hash, the git sha, and the exact CUDA / driver / torch version. Most "this used to work" bugs are environment drift.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, gc

def memory_audit(label=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{label}] alloc {alloc:.2f} GB  reserved {reserved:.2f} GB")

# Find the biggest tensors alive — useful for leak hunting
def biggest_tensors(top_k=10):
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensors.append((obj.numel() * obj.element_size(), tuple(obj.shape), obj.dtype))
        except: pass
    tensors.sort(reverse=True)
    for size, shape, dtype in tensors[:top_k]:
        print(f"  {size / 1e6:6.1f} MB  {shape}  {dtype}")
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
<span class="annotation">The canonical reference for debugging neural net training. Every section is worth re-reading.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[loguru <i class="fas fa-external-link-alt"></i>](https://github.com/Delgan/loguru){: target="_blank" }
<span class="annotation">Modern Python logging that "just works". Zero-config, structured, rotating, colourful by default.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyTorch Profiler Recipe <i class="fas fa-external-link-alt"></i>](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html){: target="_blank" }
<span class="annotation">Official walkthrough on the PyTorch profiler — first stop when training is slow.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[SimpleParsing — Troubleshooting <i class="fas fa-external-link-alt"></i>](https://github.com/lebrice/SimpleParsing/blob/master/docs/troubleshooting.md){: target="_blank" }
<span class="annotation">Many of the same patterns transfer to ML codebases — including configuration debugging.</span>

</li>
</ul>

</div>

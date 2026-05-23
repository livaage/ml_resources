---
title: Training Scaffolding — ML Resources Hub
eyebrow_text: ← Engineering · Time-Savers
eyebrow_href: {{root}}engineering.html
heading: Training Scaffolding
lead: Reusable training-loop boilerplate — Lightning, Accelerate, custom — so you spend time on the model, not the wrapper.
active_nav: engineering
prev_href: notebook-to-script.html
prev_title: Notebook → Script
next_href: cli-patterns.html
next_title: CLI Patterns
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The training loop is mostly the same across projects.** Forward pass, backward, step, scheduler, log, eval, save. Don't rewrite it every time. Use Lightning, Accelerate, or a small in-house scaffold. The model logic is what's project-specific; the loop is plumbing.

</div>

<article class="tldr-body" markdown="1">

**Three options.** (1) Vanilla PyTorch — the most control, the most boilerplate. (2) HuggingFace Accelerate — minimal changes to your existing loop, handles devices + distributed + mixed precision. (3) PyTorch Lightning — opinionated scaffold; you implement `training_step`, it handles everything else.

Lightning is great for typical supervised training; Accelerate is great when you want to keep your custom loop but get free distributed + mixed-precision; vanilla is right for unusual training patterns (RL, GANs, custom adversarial setups).

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Picking a framework

- **Lightning**: typical supervised training, fast iteration, good defaults
- **Accelerate**: existing custom loop you don't want to throw away
- **Vanilla PyTorch**: unusual training (RL, multi-network adversarial)
- **Trainer (HF)**: transformer fine-tuning specifically

</div>

<div class="no" markdown="1">

### Common mistakes

- Rewriting the loop from scratch in every project
- Wrapping a framework so tightly you can't use its features
- Using Lightning + bypassing its abstractions ("Lightning + custom hooks")
- Reinventing logging, checkpointing, mixed-precision yourself

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import lightning as L
import torch
import torch.nn.functional as F

class MyModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_backbone()
        self.head     = torch.nn.Linear(512, 10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.head(self.backbone(x))
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.head(self.backbone(x))
        loss = F.cross_entropy(logits, y)
        acc  = (logits.argmax(-1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

trainer = L.Trainer(
    max_epochs=20, devices=4, strategy="ddp", precision="bf16-mixed",
    callbacks=[L.callbacks.ModelCheckpoint(monitor="val/loss")],
    logger=L.loggers.WandbLogger(project="my-project"),
)
trainer.fit(MyModel(), train_loader, val_loader)
```

</div>

<div class="level-next">
<span>Want callbacks, custom loops, & the Accelerate pattern?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The standard training loop, abstracted</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{prep} \to (\text{step} \to \text{log} \to \text{eval} \to \text{checkpoint})^* \to \text{save} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Every framework's loop is some specialisation of this

</li>
<li markdown="1">

The hook points are where you customise — callbacks, on_step_end, etc.

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{setup} \;\to\; \big(\text{train step} \to \text{log} \to \text{evaluate} \to \text{checkpoint}\big)\text{ repeated} \;\to\; \text{final save} $$</span>

**In words.** Almost every training loop has the same skeleton. `prep` happens once at startup (build model, optimizer, data loader). Then a block of four stages — step the model, log metrics, run eval periodically, save a checkpoint — runs many times in a loop; the `*` is regex-style notation for "repeat zero or more times". At the end, save the final model. Frameworks differ only in *how* you customise each stage (callbacks, hooks, mixins) — the shape is the same.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`setup`one-time work: model, optimizer, data, devices

</li>
<li markdown="1">

`train step`forward + backward + optimizer step on one batch

</li>
<li markdown="1">

`log / evaluate / checkpoint`periodic side-effects inside the loop

</li>
<li markdown="1">

Every framework's loop is some specialisation of this skeleton

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Lightning callbacks.** Lightning's extensibility model. `EarlyStopping`, `ModelCheckpoint`, `LearningRateMonitor`, custom callbacks. The right place to inject behaviour without bloating the LightningModule.

**Accelerate.** Wraps your existing loop. `accelerator.prepare(model, optimizer, dataloader)` handles device placement, mixed precision, distributed. Your loop stays mostly the same but gains all the framework features.

**Mixin patterns.** Reusable pieces: `EMAMixin` for exponential moving average of weights, `GradAccumMixin` for accumulation. Drop into any model.

**Custom Lightning DataModules.** Encapsulate your data setup: `prepare_data` (download), `setup` (split, transform), `train_dataloader` / `val_dataloader`. Lightning's separation of model and data scaffolding pays off across multiple projects.

**Trainer flags.** `fast_dev_run=True` for a smoke test, `overfit_batches=2` for a 1-batch sanity check, `limit_train_batches=0.1` for a quick partial run. Lightning has most of these built in; Accelerate gets them via your own flags.

**Save the LightningModule, not the model.** Lightning saves the module with hyperparameters; load with `MyModel.load_from_checkpoint(path)` and you get the same object with the same configuration. Don't extract the bare PyTorch model unless you need to deploy to non-Lightning code.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from accelerate import Accelerator

# Keep your own training loop; let Accelerate handle the rest
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)

model, opt, loader = accelerator.prepare(model, opt, loader)

for batch in loader:
    with accelerator.accumulate(model):
        out  = model(batch["x"])
        loss = loss_fn(out, batch["y"])
        accelerator.backward(loss)
        opt.step(); opt.zero_grad()

    if accelerator.is_main_process and step % 100 == 0:
        accelerator.print(f"step={step} loss={loss.item():.4f}")

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.save(accelerator.unwrap_model(model).state_dict(), "ckpt.pt")
```

</div>

<div class="level-next">
<span>Want custom callbacks, EMA / SWA, & the unconventional loops?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Exponential Moving Average</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \theta_{\mathrm{EMA}}^{(t)} = \mu \cdot \theta_{\mathrm{EMA}}^{(t-1)} + (1 - \mu) \cdot \theta^{(t)} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Stash a smoothed copy of your weights

</li>
<li markdown="1">

Use the EMA weights at eval / inference — often more accurate, more stable

</li>
<li markdown="1">

Standard for diffusion training, self-supervised methods, and some supervised setups

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{smoothed weights now} = \mu \cdot \text{smoothed weights before} + (1 - \mu) \cdot \text{current weights} $$</span>

**In words.** EMA (exponential moving average) keeps a slowly-updating shadow copy of your weights. At each step, the smoothed weights are mostly the previous smoothed weights, mixed with a tiny bit of the latest live weights. `μ` (mu, a Greek letter) is the decay — typically `0.999` or `0.9999`; bigger means slower-changing (more smoothing). The superscript `(t)` just labels the step number. Use the smoothed copy at evaluation / inference time — it's almost always a slightly better, more stable predictor than the latest weights.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`smoothed weights now`the EMA shadow copy after the current step

</li>
<li markdown="1">

`μ`decay factor (close to 1) — how much of the previous EMA to keep

</li>
<li markdown="1">

`current weights`the live trained parameters at this step

</li>
<li markdown="1">

Standard for diffusion training, self-supervised methods, and some supervised setups

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**EMA / SWA.** Maintain a running average of weights during training; use it at inference. EMA decay typically 0.999 or 0.9999. SWA (Stochastic Weight Averaging) is the larger-step variant. Both reliably improve test accuracy by 0.1-1pp.

**Gradient accumulation correctly.** Forward + backward N micro-batches without stepping; then step once. Effective batch size is N × micro_batch. Watch out for BatchNorm — it uses per-rank stats, not effective batch stats.

**Custom unusual loops.** GAN training (alternating G and D updates), reinforcement learning (rollouts + updates), curriculum learning (data shifting over time). Lightning supports most via callbacks; pure PyTorch is sometimes cleaner.

**Checkpoint hygiene.** Save every N steps (resumable mid-epoch). Keep last K and best M checkpoints. Compress with safetensors. Decoupled from "the final model" — that's a registry artefact.

**Resume on failure.** `trainer.fit(ckpt_path="last.ckpt")` in Lightning. With Accelerate, save and restore both model + optimizer + scheduler + RNG state. Essential for jobs longer than a few hours.

**Profile inside the training loop.** `torch.profiler` can be attached as a Lightning callback. Look at the chrome trace; spot data-loader gaps, kernel-launch overhead, slow communications.

**Test the training loop.** Unit test the LightningModule's `training_step` with a hand-built batch. Asserts: loss is non-negative, gradients are non-zero on every parameter, output shape matches. Catches refactor bugs without launching a full run.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import lightning as L
import torch
import copy

class EMACallback(L.Callback):
    def __init__(self, decay=0.999):
        self.decay = decay; self.ema = None

    def on_train_start(self, trainer, pl_module):
        self.ema = copy.deepcopy(pl_module).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *_):
        for p_e, p in zip(self.ema.parameters(), pl_module.parameters()):
            p_e.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._swap(pl_module)
    def on_validation_epoch_end(self, trainer, pl_module):
        self._swap(pl_module)

    def _swap(self, pl_module):
        for p, p_e in zip(pl_module.parameters(), self.ema.parameters()):
            p.data, p_e.data = p_e.data.clone(), p.data.clone()
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

[PyTorch Lightning Documentation <i class="fas fa-external-link-alt"></i>](https://lightning.ai/docs/pytorch/){: target="_blank" }
<span class="annotation">The reference. Trainer flags, callbacks, DataModules, and Lightning Studio for hosted training.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[HuggingFace Accelerate <i class="fas fa-external-link-alt"></i>](https://huggingface.co/docs/accelerate){: target="_blank" }
<span class="annotation">Minimal-wrapper alternative — wraps your existing loop instead of replacing it. Works particularly well with HF transformers.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[karpathy/nanoGPT <i class="fas fa-external-link-alt"></i>](https://github.com/karpathy/nanoGPT){: target="_blank" }
<span class="annotation">The minimalist counter-example — full GPT training in a few hundred lines of pure PyTorch. Worth reading for the "what does the framework do for you" perspective.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[timm — PyTorch Image Models <i class="fas fa-external-link-alt"></i>](https://timm.fast.ai/){: target="_blank" }
<span class="annotation">The reference library for vision backbones, with a built-in training script. Useful even if you don't use timm; the training script is a good template.</span>

</li>
</ul>

</div>

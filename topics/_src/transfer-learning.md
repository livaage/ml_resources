---
title: Transfer Learning &amp; Fine-Tuning — ML Resources Hub
eyebrow_text: ← Theory · Learning Paradigms
eyebrow_href: ../theory.html
heading: Transfer Learning &amp; Fine-Tuning
lead: Don't start from scratch — take a pre-trained model and specialise it. The trick that made modern ML scale.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Most of what a model learns is generally useful.** Edges, textures, syntax, word relationships — these aren't specific to your downstream task; they're general structure of images or language. Pre-train a model on a huge corpus, then specialise the last bit to your task with whatever labels you have. You get a head-start equivalent to thousands of labelled examples.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Compare three strategies as the labelled set shrinks — from scratch vs frozen-features vs full fine-tune</span>
</div>
<div class="viz-classic-controls">
<button id="viz-tl-step" type="button">Re-sample</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                # labels
                <input id="viz-tl-n" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-tl-n-lbl">n = 20</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-tl-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-tl-caption"></div>
</div>

<script src="{{root}}js/viz/transfer-learning.js"></script>

The chart shows test accuracy as a function of labelled-training-set size for three strategies. **From scratch** trains a fresh model on just the labels. **Linear probe** uses a frozen pre-trained encoder, training only a linear head. **Fine-tune** initialises with the pre-trained weights and updates everything. As labels become plentiful the curves converge — but at the low end, transfer learning provides a 10-30 point head-start.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Why transfer works.** The first few layers of a CNN are essentially edge detectors regardless of the task. The first few layers of a transformer encode syntactic relationships. These are reusable. The last layers are task-specific. Replace those; keep the rest.

**Linear probe / feature extraction.** Freeze the encoder. Train only a linear classifier on top. Fast, robust to small datasets, won't overfit. Best when your task is similar to the pre-training task.

**Fine-tuning.** Initialise with pre-trained weights and update everything (or just the later layers). More flexible, sometimes more accurate, more sensitive to overfitting on small labelled sets.

**Adapter-based fine-tuning.** Insert small trainable modules into a frozen backbone (LoRA, adapters). Cheap to train, cheap to switch tasks. The dominant fine-tuning recipe for LLMs.

**Domain adaptation.** Train on labelled source domain; deploy on unlabelled target domain. Use adversarial losses, importance weighting, or feature alignment to bridge the gap. Useful when labels exist for one domain (medical imaging from hospital A) but not another (hospital B).

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You have a small labelled dataset for your task
- A pre-trained model exists in your domain (vision, NLP, audio)
- You can't afford to train from scratch (compute or data)
- The pre-training task is similar to your downstream task

</div>

<div class="no" markdown="1">

### Watch out

- Domain mismatch — features from natural images may not transfer to medical X-rays
- Fine-tuning with too few labels can catastrophically forget useful pre-training
- Different pre-training objectives → different features → not all encoders transfer equally
- Beware "leaderboard gains" from pre-training on test-set lookalikes

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
import torchvision.models as models

# Linear probe — freeze backbone, train head
def linear_probe(num_classes):
    m = models.resnet50(weights="DEFAULT")
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # only this trains
    return m

# Full fine-tune — train everything, smaller learning rate on the backbone
def fine_tune(num_classes):
    m = models.resnet50(weights="DEFAULT")
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
# Use param groups with different lrs:
opt = torch.optim.AdamW([
    {"params": m.fc.parameters(),       "lr": 1e-3},   # new head
    {"params": [p for n, p in m.named_parameters() if not n.startswith("fc")],
                                          "lr": 1e-5},  # pre-trained backbone
])
```

</div>

<div class="level-next">
<span>Want LoRA, adapters, prompt tuning, and catastrophic forgetting?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">LoRA update</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ W_{\text{new}} = W_{\text{pretrained}} + \frac{\alpha}{r} \cdot B A, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}, \; B \in \mathbb{R}^{d_{\text{out}} \times r} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Wpretrained`frozen pre-trained weight matrix

</li>
<li markdown="1">

`A, B`small trainable matrices; *A* projects down to rank *r*, *B* projects back up

</li>
<li markdown="1">

`r`rank of the update (typically 4–64)

</li>
<li markdown="1">

`α`scaling factor; the `α/r` ratio keeps the update magnitude controlled

</li>
<li markdown="1">

*r* << *d* ⇒ ~100× fewer trainable parameters; LoRA adapters are composable

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{new weights} \;=\; \text{frozen pre-trained weights} \;+\; \tfrac{\alpha}{r} \,\times\, (\text{down-projection} \times \text{up-projection}) $$</span>

**In words.** Keep the original weight matrix *frozen* — don't touch it. Add a small, trainable update on top, structured as the product of two low-rank matrices: one that projects the input down to a tiny dimension *r*, and another that projects back up to the original output size. Because *r* is small (often 8 or 16), this update has roughly 100× fewer parameters than the original weight matrix. The `α/r` scaling keeps the update from being too big — `α` is a hyperparameter you tune. At inference you can either keep the LoRA modules separate (and switch them per task) or fold them back into the original weights with no overhead.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`frozen pre-trained weights`the original matrix from the pre-trained model; never updated

</li>
<li markdown="1">

`down-projection`small matrix that shrinks the input to a tiny dimension *r*

</li>
<li markdown="1">

`up-projection`small matrix that brings it back up to the output size

</li>
<li markdown="1">

`r`rank — how small the bottleneck is; controls parameter count

</li>
<li markdown="1">

`α`scaling factor; `α/r` keeps the magnitude of the update sensible

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Parameter-efficient fine-tuning (PEFT).** Don't update all the weights — just a small fraction. *Adapters* (Houlsby et al. 2019): insert small MLPs after each layer. *LoRA* (Hu et al. 2021): add low-rank updates to frozen weights. *Prefix tuning* / *prompt tuning*: train a small "prompt" embedding while keeping the model frozen. All are cheap to train, cheap to store, easy to switch.

**Catastrophic forgetting.** Naive fine-tuning destroys useful pre-training. Mitigations: lower learning rate, fewer epochs, freezing earlier layers, regularising weights toward initialisation (EWC), replaying pre-training data.

**Discriminative fine-tuning.** Different layers, different learning rates. The standard recipe: small lr for early (general) layers; larger lr for later (task-specific) layers. Also called "layer-wise lr decay".

**Distribution shift in transfer.** The pre-training distribution rarely matches the downstream one exactly. Domain adaptation (adversarial alignment, importance weighting, test-time adaptation) bridges the gap when labels for the target domain are scarce.

**Foundation models as the new default.** CLIP, GPT-3/4, BERT, ViT, DINOv2. Pre-train once at enormous cost; reuse forever. Fine-tune for cheap. The economics of ML have changed: most teams now start from a foundation model and fine-tune, rarely training from scratch.

**Zero-shot and few-shot.** Modern foundation models can do many tasks without any task-specific training — prompt them appropriately. Few-shot uses a handful of demonstrations in the prompt (in-context learning). Often beats traditional fine-tuning when labels are very scarce.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# LoRA fine-tuning a transformer — only adapter params are trainable
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(base_model, config)
peft_model.print_trainable_parameters()
# trainable params: 1,572,864 || all params: 175,000,000,000 || trainable%: 0.0009

# Discriminative fine-tuning — layer-wise lr decay
def layer_lrs(model, base_lr=1e-3, decay=0.7):
    groups = []
    for i, layer in enumerate(reversed(list(model.layers))):
        groups.append({"params": layer.parameters(),
                       "lr": base_lr * (decay ** i)})
    return groups
```

</div>

<div class="level-next">
<span>Want continual learning, prompt tuning, RLHF, and meta-learning?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Elastic Weight Consolidation</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \sum_i \lambda \, F_i \, (\theta_i - \theta^*_i)^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Lnew`loss on the new task

</li>
<li markdown="1">

`θ*i`pre-trained value of parameter *i*

</li>
<li markdown="1">

`Fi`Fisher information per parameter — how "important" that parameter was for the old task

</li>
<li markdown="1">

`λ`regularization strength

</li>
<li markdown="1">

Penalise moving important pre-trained parameters far from their original values

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; \text{new-task loss} \;+\; \lambda \sum_i (\text{importance}_i) \,\times\, (\text{new param}_i - \text{old param}_i)^2 $$</span>

**In words.** The total loss has two pieces. The first is the usual training loss on the new task. The second is a penalty: for each parameter, it asks "how far has this parameter moved from its pre-trained value, weighted by how *important* that parameter was for the old task?" The `Σ` sums the penalty across every parameter. "Importance" here is the **Fisher information** — large for parameters that mattered most for the old task. `λ` (lambda) controls how strongly to hold on to the old behaviour. This lets the model learn the new task while preserving the parts of pre-training it most needs to keep.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`new-task loss`standard supervised loss on the new task's labelled data

</li>
<li markdown="1">

`importance`Fisher information — how much each parameter mattered for the old task

</li>
<li markdown="1">

`new param − old param`how far the parameter has moved from its pre-trained value

</li>
<li markdown="1">

`λ`regularization strength — bigger λ means stronger anti-forgetting pressure

</li>
<li markdown="1">

Classical antidote to catastrophic forgetting

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Continual / lifelong learning.** Train sequentially on a stream of tasks; don't forget the old ones. Classical methods: EWC, Synaptic Intelligence, MAS — penalise moving important parameters. Modern: rehearsal buffers, progressive networks. Hard in general; trade-offs between plasticity and stability are inescapable.

**Prompt tuning & prefix tuning.** Keep the base model frozen; learn a small embedding ("soft prompt") prepended to inputs. Lester et al. (2021), Li & Liang (2021). Competitive with full fine-tuning at modest scale; doesn't catch up at the largest scales.

**In-context learning.** The strangest discovery of LLM-era ML: large enough models can learn from examples in their prompt without any gradient updates. Few-shot prompting is the canonical case; reasoning chains (chain-of-thought) extend it. The mechanism is still actively researched.

**Multi-task learning.** Train a single model on many tasks simultaneously. Share early layers; specialise heads. Risks: tasks interfere with each other (negative transfer); the gradient of the joint objective is messy. Helped by careful loss weighting (uncertainty weighting, GradNorm).

**Meta-learning ("learn to learn").** Train on a distribution over tasks; the model learns to adapt quickly to a new task with few examples. MAML (Finn et al. 2017), ProtoNets (Snell et al. 2017), Reptile. Mostly superseded by foundation-model fine-tuning at scale, but still relevant for low-data regimes.

**RLHF as transfer.** Pre-trained LLMs are "supervised fine-tuned" on instructions, then RLHF-tuned against human preferences. This is transfer learning with multiple objectives and a learned reward. The pre-train → SFT → RLHF pipeline is now standard for assistant-grade models.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# Elastic Weight Consolidation — penalise moving important pre-trained params
class EWC:
    def __init__(self, model, fisher_info, theta_star, lam=1000.0):
        self.fisher, self.theta_star, self.lam = fisher_info, theta_star, lam

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.theta_star[n]) ** 2).sum()
        return self.lam * loss

# In a training step:
loss = task_loss + ewc.penalty(model)
loss.backward()
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

[Hu et al. (2021) — LoRA <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2106.09685){: target="_blank" }
<span class="annotation">Low-rank adaptation for LLMs. Short, well-written; section 4 is the canonical reference for the architecture.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[HuggingFace PEFT <i class="fas fa-external-link-alt"></i>](https://github.com/huggingface/peft){: target="_blank" }
<span class="annotation">The reference parameter-efficient fine-tuning library. LoRA, prefix tuning, IA3, AdaLoRA — all in one place with HF Transformers integration.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — Learning with not enough data Pt II <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2022-02-20-active-learning/){: target="_blank" }
<span class="annotation">Companion to the SSL piece — covers active learning, transfer learning, and fine-tuning recipes.</span>

</li>
<li data-tier="indepth" markdown="1">

[Kirkpatrick et al. (2017) — EWC <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1612.00796){: target="_blank" }
<span class="annotation">The Elastic Weight Consolidation paper — the canonical continual-learning anti-forgetting method.</span>

</li>
</ul>

</div>

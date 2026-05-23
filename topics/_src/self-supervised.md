---
title: Self-Supervised &amp; Contrastive Learning — ML Resources Hub
eyebrow_text: ← Theory · Learning Paradigms
eyebrow_href: ../theory.html
heading: Self-Supervised &amp; Contrastive Learning
lead: Make up the labels — pre-train huge models from huge unlabelled data, then fine-tune on the little labelled data you have.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The model invents labels from the input itself.** Hide half a sentence and predict the rest. Crop two pieces of one image and learn to recognise they're related. Mask 15% of an image's patches and rebuild them. The supervision is free — the structure is in the data — and the resulting representations transfer beautifully.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Watch a contrastive loss pull positive pairs together and push negative pairs apart in the embedding space</span>
</div>
<div class="viz-classic-controls">
<button id="viz-ssl-step" type="button">Step</button>
<button id="viz-ssl-play" type="button">Play</button>
<button id="viz-ssl-reset" type="button">Reset</button>
<span class="viz-classic-badge" id="viz-ssl-step-lbl">step 0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-ssl-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-ssl-caption"></div>
</div>

<script src="{{root}}js/viz/self-supervised.js"></script>

8 "images" each get 2 augmented views (16 points total). Same-image views are *positive pairs* — the contrastive loss wants their embeddings close together. Different-image views are *negative pairs* — push apart. Step a few times and watch the 8 pairs cluster while the clusters spread away from each other. The dashed lines connect positives; same-coloured pairs are augmentations of the same source.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Masked prediction.** Hide part of the input, predict it back. BERT masks ~15% of tokens. Masked autoencoders (MAE) mask ~75% of image patches. The model learns rich representations because it has to *understand* the input to fill in the gaps.

**Next-token prediction.** Predict the next token given the previous ones. Trains GPT and friends. Simpler than masked prediction; scales spectacularly.

**Contrastive learning.** Take two augmented "views" of the same input. Pull their embeddings together; push apart from views of different inputs. SimCLR (Chen et al. 2020), MoCo (He et al. 2020), DINO (Caron et al. 2021) are landmark methods.

**Non-contrastive methods.** BYOL (Grill et al. 2020) and SimSiam (Chen & He 2021) — just predict one view's embedding from the other, with a stop-gradient. Surprisingly, no negatives needed.

**The product is the encoder.** The pretext task is a means to an end. After pre-training, you keep the encoder and discard the head. The encoder is then either frozen (linear probe) or fine-tuned for downstream tasks.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You have huge unlabelled data and small labelled data
- You want general-purpose representations (foundation models)
- The supervised baseline is data-bottlenecked
- You need to share an encoder across many tasks

</div>

<div class="no" markdown="1">

### Limits

- Pre-training is computationally expensive — millions of GPU-hours at scale
- Designing the pretext task is its own research problem
- Linear-probe vs fine-tune trade-off is task-dependent
- Some downstream tasks need very different features than the pretext task selected for

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# SimCLR — info-NCE contrastive loss
def info_nce(z1, z2, temperature=0.07):
    # z1, z2: (B, d) — paired views of the same images
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    sim = z @ z.t() / temperature                    # (2B, 2B)
    sim.fill_diagonal_(-1e9)                         # mask self-similarity
    # Positives: index i and i+B (and vice versa)
    labels = torch.cat([torch.arange(B, 2 * B),
                         torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)

# Typical training step
view1, view2 = augment(x), augment(x)
z1 = encoder(view1); z2 = encoder(view2)
loss = info_nce(z1, z2)
loss.backward(); opt.step()
```

</div>

<div class="level-next">
<span>Want the contrastive math, alignment / uniformity, and BYOL's trick?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">InfoNCE loss</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathrm{sim}(z_i, z_i^+)/\tau)}{\sum_{j} \exp(\mathrm{sim}(z_i, z_j)/\tau)} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`zi`embedding of the anchor view

</li>
<li markdown="1">

`zi+`positive pair of *z<sub>i</sub>* (different view of same image)

</li>
<li markdown="1">

`zj`every other embedding in the batch (positive plus all negatives)

</li>
<li markdown="1">

`sim(·, ·)`similarity (cosine, after L2-normalising)

</li>
<li markdown="1">

`τ`temperature — controls how peaked the similarity is

</li>
<li markdown="1">

Lower bound on mutual information between views

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; -\log \frac{\exp(\text{similarity to positive} / \tau)}{\sum_{j} \exp(\text{similarity to view}_j / \tau)} $$</span>

**In words.** For each anchor view, the loss is a softmax cross-entropy that asks: "out of all the other views in the batch, which one is your positive pair?" The numerator is the exponentiated similarity between the anchor and its known positive; the denominator adds up that quantity across every view in the batch (positive plus all the negatives). Dividing makes a probability between 0 and 1 — the probability the model assigns to picking the right partner. Taking `−log` turns that into a loss that's small when the positive is highest-similarity. `τ` (tau, the temperature) sharpens or smooths how confidently the softmax peaks: small τ → very peaked, large τ → flatter.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`similarity to positive`cosine similarity between anchor and its known positive pair

</li>
<li markdown="1">

`similarity to viewj`cosine similarity between anchor and every view in the batch

</li>
<li markdown="1">

`τ`temperature — small τ means the softmax is more peaked

</li>
<li markdown="1">

Minimising this loss = "make the positive pair clearly the most similar"

</li>
<li markdown="1">

Equivalent to a softmax classifier where the positive is the correct label and negatives are wrong labels

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Alignment and uniformity.** Wang & Isola (2020) showed contrastive learning optimises two terms: *alignment* (positives close) and *uniformity* (representations spread out on the hypersphere). A well-trained encoder has both — collapse to a constant is the failure mode.

**The role of negatives.** Negatives prevent collapse. The bigger the batch (more negatives), the harder the pretext task — SimCLR uses huge batch sizes. MoCo decouples the negative pool from the batch via a memory queue. Both work.

**Non-contrastive methods (BYOL, SimSiam).** Surprisingly, you can avoid negatives by using a momentum encoder + stop-gradient + predictor head. The asymmetry breaks the collapse equilibrium. The community still doesn't fully agree on *why* this works.

**Augmentation matters.** Strong augmentations (random crops, colour jitter, blur, cutout) are essential. The encoder learns to be invariant to whatever you augment over — choose augmentations that preserve what you care about and destroy what you don't.

**Masked image modelling.** MAE (He et al. 2022): mask 75% of an image's patches, reconstruct from the remaining 25%. Asymmetric encoder (sees only visible)/decoder (rebuilds) is key for compute efficiency. State of the art for vision representation learning at scale.

**Multi-modal SSL.** CLIP (Radford et al. 2021) pairs an image and its caption — text and image encoders trained jointly with contrastive loss. The resulting embeddings align modalities, enabling zero-shot classification by similarity to text prompts.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

# Masked Autoencoder (MAE) — encoder sees only unmasked patches
class MAE(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.mask_ratio = mask_ratio

    def forward(self, x_patches):
        N = x_patches.size(1)
        keep = int(N * (1 - self.mask_ratio))
        perm = torch.randperm(N)
        ids_keep, ids_mask = perm[:keep], perm[keep:]

        z = self.encoder(x_patches[:, ids_keep])     # only visible patches
        x_hat = self.decoder(z, ids_keep, ids_mask)  # reconstruct everything
        return F.mse_loss(x_hat[:, ids_mask], x_patches[:, ids_mask])
```

</div>

<div class="level-next">
<span>Want JEPA, DINO, multi-modal, and scaling laws?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Joint-embedding predictive architecture (JEPA)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L} = \big\lVert \mathrm{pred}\big(s_x(\text{context})\big) - s_y(\text{target}) \big\rVert^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`sx, sy`context and target encoders (often weight-shared or EMA-related)

</li>
<li markdown="1">

`pred(·)`predictor mapping context embedding → guessed target embedding

</li>
<li markdown="1">

`‖·‖²`squared Euclidean distance between predicted and actual target embedding

</li>
<li markdown="1">

Predict the target's *embedding*, not the target itself

</li>
<li markdown="1">

No pixel-level reconstruction → no wasted capacity on irrelevant detail; LeCun's preferred frame for SSL at scale

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; (\text{predicted target embedding} \;-\; \text{actual target embedding})^2 $$</span>

**In words.** Encode the context (e.g. visible patches of an image) into an embedding, run a small **predictor** network on it to guess what the target patch's embedding would look like, then compare that guess to the actual embedding of the target patch (produced by a separate target encoder). The loss is the squared distance between the prediction and the real embedding. Unlike masked autoencoders, you never reconstruct raw pixels — the model only has to get the *features* right, so it doesn't waste capacity on irrelevant detail like noise or texture.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`context`visible part of the input (e.g. unmasked image patches)

</li>
<li markdown="1">

`target`hidden part of the input the model has to guess

</li>
<li markdown="1">

`predicted embedding`predictor's guess at what the target's encoded features should be

</li>
<li markdown="1">

`actual embedding`the target encoder's actual output on the hidden region

</li>
<li markdown="1">

Operates in feature space, not pixel space — no wasted capacity on irrelevant detail

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Distillation-based SSL (DINO).** Caron et al. (2021). Teacher + student encoders see different crops; the student matches the teacher's prediction; the teacher is an EMA of the student. No negatives, no labels, surprising stability. The features it learns are remarkably semantic — segment objects without supervision.

**JEPA family.** Predict embeddings, not raw inputs. I-JEPA (Assran et al. 2023) extends MAE-style masking to embedding space. Avoids the "pixel perfection" trap where the model wastes capacity on details that don't matter.

**Scaling laws.** SSL pretraining benefits enormously from scale. Empirical evidence (Goyal et al. 2022, OpenCLIP 2022): doubling data ⇒ predictable improvement on downstream linear probe. Foundation models exploit this directly.

**The semantic richness of CLIP.** Contrastive image-text pre-training produces embeddings where similarity ~ semantic alignment. The zero-shot recipe ("classify by similarity to text prompts") works surprisingly well even without any fine-tuning.

**SSL evaluation.** Linear probe (freeze encoder, train logistic regression head) is the standard. k-NN probe is faster and often correlates. Fine-tuning is the more practically-relevant test but slower; reflects the operating mode you'd deploy in.

**Why does SSL work so well?** The bet is that pretext tasks like "fill in the blank" force the model to learn the actual structure of the data — what's likely and unlikely. With enough data, this structure is exactly what downstream tasks need. It's not magic; it's information-rich supervision.

**Limits.** SSL features encode what the pretext task incentivises — fine-grained semantics, but rarely numeric reasoning, planning, or counterfactual abilities. The choice of pretext task is itself an inductive bias.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# DINO loss — student predicts teacher's distribution (no negatives)
def dino_loss(student_logits, teacher_logits, temp_s=0.1, temp_t=0.04):
    teacher_probs = F.softmax(teacher_logits / temp_t, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp_s, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()

# Teacher EMA update
def update_teacher(student, teacher, m=0.996):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(m).add_((1 - m) * p_s.data)
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

[Lilian Weng — Contrastive Representation Learning <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2021-05-31-contrastive/){: target="_blank" }
<span class="annotation">Tour of every major contrastive method: InfoNCE, SimCLR, MoCo, BYOL, SwAV, DINO. The single best learning-tour blog post on the topic.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Chen et al. (2020) — SimCLR <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2002.05709){: target="_blank" }
<span class="annotation">The paper that made contrastive learning click for vision. Section 2's recipe is the canonical reference.</span>

</li>
<li data-tier="indepth" markdown="1">

[He et al. (2022) — Masked Autoencoders <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2111.06377){: target="_blank" }
<span class="annotation">The MAE paper. Asymmetric encoder/decoder, 75% mask ratio, simple and effective.</span>

</li>
<li data-tier="intuition" markdown="1">

[LeCun & Misra — Self-Supervised: The Dark Matter <i class="fas fa-external-link-alt"></i>](https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/){: target="_blank" }
<span class="annotation">Manifesto-style argument for SSL as the right frame for foundation models. Worth reading even if you disagree with the framing.</span>

</li>
</ul>

</div>

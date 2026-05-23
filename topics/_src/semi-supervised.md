---
title: Semi-Supervised Learning — ML Resources Hub
eyebrow_text: ← Theory · Learning Paradigms
eyebrow_href: ../theory.html
heading: Semi-Supervised Learning
lead: A handful of labels, a mountain of unlabelled data — how to use both.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The unlabelled data tells you about the input distribution.** Even without labels, you can see that the data has clusters, a manifold, low-density boundaries. Those structural signals tell a model "the decision boundary probably runs *here*, not *there*." Pair that with a small labelled set and you can do real work.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Same dataset, four strategies — only a few points are labelled; watch how each method uses the rest</span>
</div>
<div class="viz-classic-controls">
<button id="viz-ssl2-sup" type="button" class="active">Supervised only</button>
<button id="viz-ssl2-pseudo" type="button">Pseudo-label</button>
<button id="viz-ssl2-graph" type="button">Graph propagation</button>
<button id="viz-ssl2-self" type="button">Self-training (iterated)</button>
<button id="viz-ssl2-reset" type="button">Re-sample</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-ssl2-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-ssl2-caption"></div>
</div>

<script src="{{root}}js/viz/semi-supervised.js"></script>

8 labelled points (filled, indigo or orange), many unlabelled (grey). **Supervised only** draws a boundary from just the labelled — often crooked. **Pseudo-label** classifies the unlabelled with the supervised model, retrains on everything. **Graph propagation** spreads labels through k-NN edges based on similarity. **Self-training** iterates — confidence-thresholded pseudo-labels added each round.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Self-training / pseudo-labels.** Train on the labelled; predict on unlabelled; add the confident predictions to the training set; repeat. Simple, works well when the supervised model is already decent.

**Consistency regularization.** Augment an unlabelled example two ways; punish the model for predicting different labels. Forces invariances; underpins FixMatch (Sohn et al. 2020) and friends.

**Graph-based methods.** Build a k-NN graph; let labels propagate along high-weight edges; absorb to a smooth function on the graph. Classical (Zhu et al. 2003); still useful on structured data.

**Pre-training + fine-tuning.** The dominant modern recipe — see [Self-Supervised Learning](self-supervised.html). Pre-train on all the unlabelled data; fine-tune on the labelled. Conceptually a different family but functionally the same goal: do more with little label data.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Labelled data is expensive but unlabelled is plentiful
- The data has cluster / manifold structure you can exploit
- Active learning has been tried and you still need more
- You can iterate cheaply on a pseudo-labelling pipeline

</div>

<div class="no" markdown="1">

### Watch out

- Pseudo-label errors compound — start with confident predictions only
- The cluster assumption can be false — clusters may not align with classes
- Consistency regularization needs careful augmentation choices
- Graph methods scale badly to huge data

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.semi_supervised import (
    SelfTrainingClassifier, LabelPropagation, LabelSpreading,
)
from sklearn.linear_model import LogisticRegression

# Self-training: -1 means "no label". The classifier predicts and adds
# confident predictions to its own training set, iteratively.
clf = SelfTrainingClassifier(LogisticRegression(), threshold=0.9, max_iter=10)
clf.fit(X_mixed, y_mixed)             # y has -1 for unlabelled

# Label propagation: k-NN graph, labels spread along edges
lp = LabelPropagation(kernel="knn", n_neighbors=7)
lp.fit(X_mixed, y_mixed)              # -1 unlabelled
predicted = lp.transduction_          # labels assigned to each unlabelled
```

</div>

<div class="level-next">
<span>Want FixMatch, MixMatch, & the cluster assumption?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Consistency loss</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}_{\text{cons}} = \mathbb{E}_{x_u}\,\big\lVert f_\theta(\alpha_1(x_u)) - f_\theta(\alpha_2(x_u)) \big\rVert^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`xu`an unlabelled example

</li>
<li markdown="1">

`α1, α2`two random augmentations of *x<sub>u</sub>*

</li>
<li markdown="1">

`fθ(·)`model's prediction (logits or probability)

</li>
<li markdown="1">

`‖·‖²`squared difference between the two predictions

</li>
<li markdown="1">

Encourage the same prediction under augmentation — the cluster assumption operationalised

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; \text{avg over unlabelled}\big(\,\text{prediction on view 1} \;-\; \text{prediction on view 2}\,\big)^2 $$</span>

**In words.** Take an unlabelled example, augment it two different ways (crop differently, jitter colour, etc.), and ask the model to make the *same prediction* on both versions. The loss is the squared difference between the two predictions, averaged over all unlabelled examples. You don't need a label to compute this — you only need the model to be consistent with itself under augmentation. This bakes in the assumption that small perturbations shouldn't change the class — equivalent to the cluster / smoothness assumption operationalised through random crops, flips, colour jitter, etc.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`unlabelled example`a data point with no known label

</li>
<li markdown="1">

`view 1, view 2`two random augmentations of the same example

</li>
<li markdown="1">

`prediction`the model's output (logits or probabilities) for that view

</li>
<li markdown="1">

Loss is small when the model gives the same answer on both views

</li>
<li markdown="1">

No label needed — the supervision is self-consistency

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The cluster assumption.** Examples in the same cluster have the same label. Pseudo-labelling, graph propagation, and consistency methods all bet on this. When it fails — clusters that span multiple classes — semi-supervised methods can hurt rather than help.

**The smoothness assumption.** The decision function should be locally constant in regions of high data density. Manifold / graph-based methods optimise exactly this.

**FixMatch.** Sohn et al. (2020). Apply a weak augmentation, get a pseudo-label; apply a strong augmentation, predict that pseudo-label as ground truth. Threshold by confidence. Sets the modern standard for vision semi-supervised learning.

**MixMatch / ReMixMatch / FlexMatch.** Variants that smooth pseudo-labels by averaging across augmentations, mix-up labelled and unlabelled examples, or dynamically adjust the confidence threshold per class. Marginally better than FixMatch on standard benchmarks.

**Mean teacher.** Maintain a moving-average copy of the model (the "teacher") and use it to generate pseudo-labels. The student matches the teacher; the teacher is an EMA of the student. Less noisy than vanilla pseudo-labelling.

**Connection to self-supervised.** Pretrain self-supervised, fine-tune on the labels. With foundation models, this is often the best semi-supervised method by a wide margin — the encoder already encodes the data manifold, the labels just specialise.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn.functional as F

# FixMatch — weak and strong augmentation
def fixmatch_step(model, x_l, y_l, x_u, weak, strong, tau=0.95, lam=1.0):
    # Supervised loss on the labelled set
    l_sup = F.cross_entropy(model(x_l), y_l)
    # Confident pseudo-labels from the weakly-augmented version
    with torch.no_grad():
        probs = F.softmax(model(weak(x_u)), dim=-1)
        max_p, pl = probs.max(dim=-1)
        mask = (max_p >= tau).float()
    # Consistency loss: prediction on strong-augmented matches pseudo-label
    l_cons = (F.cross_entropy(model(strong(x_u)), pl, reduction="none") * mask).mean()
    return l_sup + lam * l_cons

# Mean Teacher
class MeanTeacher:
    def __init__(self, model, m=0.999):
        self.student = model
        self.teacher = type(model)().load_state_dict(model.state_dict())
        self.m = m
    @torch.no_grad()
    def update(self):
        for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
            p_t.data.mul_(self.m).add_((1 - self.m) * p_s.data)
```

</div>

<div class="level-next">
<span>Want noisy student, MPL, and theoretical bounds?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Risk bound with unlabelled data</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ R(\hat f) \leq \hat R_l(\hat f) + \Omega(\hat f; X_u) + O\!\left(\sqrt{\tfrac{\mathrm{VC}(\mathcal{H})}{n_l}}\right) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`R(f̂)`true (test) risk of the learnt model

</li>
<li markdown="1">

`R̂l`labelled empirical risk

</li>
<li markdown="1">

`Ω(f̂; Xu)`complexity penalty informed by unlabelled data (cluster / smoothness)

</li>
<li markdown="1">

`VC(H)`VC dimension — capacity of the hypothesis class

</li>
<li markdown="1">

`nl`number of labelled examples

</li>
<li markdown="1">

Unlabelled data effectively shrinks the hypothesis class

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{true error} \;\leq\; \text{labelled training error} \;+\; \text{penalty from unlabelled} \;+\; O\!\left(\sqrt{\tfrac{\text{capacity}}{\text{number of labels}}}\right) $$</span>

**In words.** This is an *upper bound* on how bad the test error can be (the `≤` means "at most"). It has three pieces: the training error on the labelled subset; a regularization term `Ω` that uses the *unlabelled* data to penalise functions that violate the cluster or smoothness assumption; and the usual statistical-learning term involving the model's **capacity** (how flexible the function class is) and the number of **labels**. The `O(√·)` just means "this term scales like the square root of the bracketed quantity". The take-away: unlabelled data narrows the effective set of plausible models, which tightens the bound.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`true error`expected loss on new, unseen labelled data

</li>
<li markdown="1">

`labelled training error`error you actually measured on your tiny labelled set

</li>
<li markdown="1">

`penalty from unlabelled`complexity penalty informed by unlabelled data (cluster / smoothness)

</li>
<li markdown="1">

`capacity`VC dimension or similar measure of how flexible the model family is

</li>
<li markdown="1">

`number of labels`how many labelled examples you have — more labels = tighter bound

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Noisy Student (Xie et al. 2020).** Train a teacher on labelled; pseudo-label the unlabelled; train a larger student with noise injection (dropout, augmentation, stochastic depth) on both. Iterate. State of the art on ImageNet for a long time; the recipe extends to many vision tasks.

**Meta Pseudo Labels (Pham et al. 2021).** The teacher is also learned — adjust its pseudo-labelling based on how it affects the student's val loss. Closes a feedback loop that vanilla self-training lacks.

**Co-training.** Two models trained on different feature views of the data; each labels examples for the other. Works when the views are conditionally independent given the label.

**Transductive vs inductive SSL.** Transductive: predict labels for the specific unlabelled set you have (label propagation does this). Inductive: produce a function that generalises to new examples (most modern methods).

**Theoretical guarantees.** Several frameworks (Niyogi 2008, Ben-David et al. 2008) give conditions under which unlabelled data provably helps. The assumptions are strong; in practice the empirical record is more important than the bounds.

**When SSL hurts.** If the cluster assumption fails (e.g. classes overlap, or clusters in input space don't align with label boundaries), self-training amplifies the supervised model's errors. Always keep a labelled-only baseline.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Noisy Student-style iterative self-training
def noisy_student(X_l, y_l, X_u, n_iter=5):
    teacher = LogisticRegression().fit(X_l, y_l)
    for _ in range(n_iter):
        probs = teacher.predict_proba(X_u)
        conf  = probs.max(axis=1)
        pl    = probs.argmax(axis=1)
        mask  = conf > 0.9
        # Train a (potentially larger / noisier) student on union
        X_all = np.r_[X_l, X_u[mask]]
        y_all = np.r_[y_l, pl[mask]]
        teacher = LogisticRegression(C=1.0).fit(X_all, y_all)
    return teacher
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

[Sohn et al. (2020) — FixMatch <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2001.07685){: target="_blank" }
<span class="annotation">The modern reference for consistency-based semi-supervised learning. Simple recipe, strong empirical results.</span>

</li>
<li data-tier="indepth" markdown="1">

[Xie et al. (2020) — Noisy Student <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1911.04252){: target="_blank" }
<span class="annotation">Iterative self-training with noise injection. Held ImageNet SOTA for a long time. Beautiful empirical study.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — Semi-Supervised Methods <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/semi_supervised.html){: target="_blank" }
<span class="annotation">Self-training, label propagation, label spreading — all the classical methods with clean APIs.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — Learning with not enough data Pt I <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/){: target="_blank" }
<span class="annotation">Excellent survey of semi-supervised methods circa 2022 — pseudo-labelling, consistency, mean teacher, FixMatch, and more.</span>

</li>
</ul>

</div>

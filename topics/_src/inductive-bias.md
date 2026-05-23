---
title: Inductive Bias &amp; Hypothesis Spaces — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Inductive Bias &amp; Hypothesis Spaces
lead: The assumptions baked into a model — what it can express, and what it can't.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Every model is a guess about how the world is shaped.** A linear model guesses "straight lines"; a tree guesses "axis-aligned splits"; a neural network with ReLUs guesses "piecewise-linear regions". You can't learn without making these guesses — that's the *no free lunch* theorem. The art is choosing assumptions that match the world you're modelling.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Same data, four model families — each draws a wildly different boundary because of its built-in assumptions</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Data
                <select id="viz-ib-data"></select>
</label>
<button id="viz-ib-reset" type="button">Re-sample</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-ib-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-ib-caption"></div>
</div>

<script src="{{root}}js/viz/inductive-bias.js"></script>

The same dataset shown to four different model families. Each draws its decision boundary in a characteristic shape — a **linear** model can only draw straight lines; a **tree** only axis-aligned rectangles; **k-NN** draws jagged regions reflecting local neighbours; **RBF / kernel** methods draw smooth curves. None of these is "correct" — each has assumed something different about how the world is shaped. Try the XOR dataset and watch which models can express it.
{: .viz-intro }

<article class="tldr-body" markdown="1">

A model's **hypothesis space** is the set of functions it can possibly represent. Linear regression's hypothesis space is "all straight lines"; a polynomial of degree 5's is "all degree-5 polynomials"; a deep network's is "all piecewise-linear functions of a certain depth and width". The hypothesis space is the model's *capacity*.

Within a hypothesis space, the **inductive bias** is the preferences the learning algorithm uses to pick a specific function when many fit the data. Linear regression prefers the line minimising squared error. k-NN prefers the function that matches its neighbours. Without an inductive bias, infinitely many functions fit any finite dataset — you need a tie-breaker.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Choose a strong inductive bias when

- You have little data — strong assumptions compensate
- You know the underlying structure (linearity, smoothness, periodicity)
- You need interpretability
- You need fast inference

</div>

<div class="no" markdown="1">

### Choose a weak inductive bias when

- You have lots of data — the data can lead
- The underlying structure is unknown or highly nonlinear
- Highest accuracy matters more than understanding
- You can afford the compute

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Same data, different inductive biases — wildly different generalization
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC

models = {
    "linear":    LogisticRegression(),
    "tree":      DecisionTreeClassifier(max_depth=4),
    "k-NN":      KNeighborsClassifier(n_neighbors=5),
    "RBF kernel": SVC(kernel="rbf", gamma=1.5),
}
for name, m in models.items():
    m.fit(X, y)
    print(f"{name:12s} train={m.score(X, y):.3f}  test={m.score(X_t, y_t):.3f}")
```

</div>

<div class="level-next">
<span>Want the no-free-lunch theorem and capacity bounds?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">PAC learning bound</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ P\!\left[\, \text{err}_{\text{true}}(h) \leq \text{err}_{\text{train}}(h) + \sqrt{\tfrac{\ln |\mathcal{H}| + \ln(1/\delta)}{2n}}\, \right] \geq 1 - \delta $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`|ℋ|`size of the hypothesis space (its *complexity*)

</li>
<li markdown="1">

`n`training-set size

</li>
<li markdown="1">

Larger ℋ ⇒ more flexibility, but a worse generalization bound

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{Probability}\!\left[\, \text{true error} \leq \text{training error} + \sqrt{\tfrac{\ln(\text{hypothesis count}) + \ln(1/\delta)}{2 \times n}}\, \right] \geq 1 - \delta $$</span>

**In words.** With probability at least `1 − δ` (where `δ` is a small confidence-level you pick — e.g. 0.05 for "95% confident"), the true error of your chosen model *h* is at most its training error plus a penalty term. That penalty under the `√` (square root) grows when there are more possible hypotheses to choose from (the `ln|ℋ|` piece — `ln` is the natural logarithm, which grows slowly), and shrinks as you collect more data (the `n` in the denominator is your training-set size). The take-away: bigger hypothesis spaces need more data to generalize.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`hypothesis count`how many functions your model could potentially express

</li>
<li markdown="1">

`n`training-set size

</li>
<li markdown="1">

`δ`a small confidence-level number you pick

</li>
<li markdown="1">

Larger hypothesis space ⇒ more flexibility, but a worse generalization bound

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**No free lunch.** Wolpert (1996) showed that averaged over all possible target functions, every learning algorithm has the same expected performance. A model only beats random because its inductive bias matches the structure of the problems we actually care about. Don't expect a single model to win everywhere.

**Common inductive biases in ML.**
*Smoothness* (nearby inputs → nearby outputs) is assumed by k-NN, kernel methods, and most regularized models.
            *Compositionality* (complex functions built from simple ones) is assumed by deep networks.
            *Translation equivariance* is assumed by CNNs.
            *Permutation invariance* is assumed by set / graph networks.
            *Sparsity* (few active features) is assumed by L1-regularized models.

**Capacity, complexity, generalization.** Larger hypothesis spaces fit the training set better but generalize worse — that's the bias-variance trade-off seen through the lens of capacity. Classical bounds (VC dimension, Rademacher complexity) make this rigorous; the strange thing is that modern over-parameterized deep nets seem to violate them, which has become an active research area (double descent, benign overfitting).

**Choosing inductive biases is feature engineering.** Coordinates, kernels, network architectures, augmentations, and even loss functions all encode assumptions. Choosing them well is the under-appreciated half of doing ML.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Inductive biases as feature engineering
import numpy as np

# Periodic signal: encode it as sin/cos features → linear model can fit
def periodic_features(t, periods=(1, 7, 30)):
    feats = []
    for p in periods:
        feats.append(np.sin(2 * np.pi * t / p))
        feats.append(np.cos(2 * np.pi * t / p))
    return np.stack(feats, axis=-1)

# Translation equivariance: a CNN encodes "the same pattern at any location"
# Without it, you'd need exponentially more data to see every shift.

# Permutation invariance: for sets, use a model like  Σ φ(x_i)
# A vanilla MLP would have to learn the same thing for every permutation.
```

</div>

<div class="level-next">
<span>Want PAC-learnability, VC dim, and double descent?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">VC dimension</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathrm{VC}(\mathcal{H}) = \max\{\,n : \exists\, \text{points } x_1, \ldots, x_n \text{ shattered by } \mathcal{H}\,\} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

"Shattered" = ℋ can realise every possible labelling of those *n* points

</li>
<li markdown="1">

The largest such *n* is the VC dimension — a measure of effective capacity

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{VC dim}(\text{hypothesis space}) \;=\; \text{largest } n \text{ such that some set of } n \text{ points can be shattered} $$</span>

**In words.** The VC dimension is the largest number of points your model family can label every possible way. If there exist *n* points for which your model can produce every possible labelling (all 2<sup>n</sup> combinations of yes/no), the model is said to *shatter* those points. The biggest such *n* is the VC dimension — a single number that summarises how flexible your hypothesis space really is. A straight-line classifier in 2D has VC dimension 3 (can shatter 3 points, but no set of 4); deeper or wider models have bigger numbers.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`hypothesis space`the set of all functions your model can express

</li>
<li markdown="1">

`n`number of points

</li>
<li markdown="1">

`shatter`produce every possible labelling of those points

</li>
<li markdown="1">

The biggest *n* the model can shatter is its VC dimension — its effective capacity

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**PAC learnability.** A hypothesis class is PAC-learnable if there's an algorithm that, for any ε and δ, produces a hypothesis with error ≤ ε with probability ≥ 1 − δ from polynomially many samples in 1/ε, 1/δ, and the problem's complexity. Finite VC dimension is sufficient for PAC learnability under realizability assumptions; the modern view is broader (e.g., margin-based bounds, online learning).

**VC and Rademacher complexity.** VC dimension bounds how complex a binary classifier can be: linear classifiers in *d* dimensions have VC dim *d* + 1; a depth-*L* neural net with *p* parameters has VC dim roughly *O(pL)*. Rademacher complexity is data-dependent and tighter in practice — it measures how well ℋ correlates with random labels on your specific dataset.

**Double descent.** Belkin et al. (2019) showed test error in modern models has a *second descent* beyond the classical interpolation threshold: as you increase capacity past the point where the training set is fit exactly, test error *drops again*. Inductive biases of SGD and over-parameterization (the network implicitly chooses the "simplest" of the many zero-loss solutions) seem to drive this.

**Implicit regularization.** SGD on an over-parameterized model doesn't pick any zero-training-loss solution — it picks one with specific structure (often, the minimum-norm one). Architectures, optimizers, and even data augmentations all bias the search through hypothesis space; the explicit regularizer in the loss is only part of the story.

**Neural Tangent Kernel.** In the infinite-width limit, neural networks trained by gradient descent behave like kernel regression with a specific kernel (the NTK). This connects deep learning to classical generalization theory — but the connection isn't tight at finite width, where feature learning kicks in.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Implicit regularization: SGD on a deeply over-parameterized network often
# picks a minimum-norm-ish solution. Empirically: increase width and watch
# test error drop *past* the interpolation point — that's double descent.

import torch, torch.nn as nn

def make_net(width):
    return nn.Sequential(
        nn.Linear(d_in, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, d_out),
    )

# Sweep widths past the "param count = data count" line
for w in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    net = make_net(w)
    # train to (near-)zero training loss …
    print(w, evaluate(net, X_test, y_test))
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

[Schapire — Notes on No Free Lunch <i class="fas fa-external-link-alt"></i>](https://www.cs.princeton.edu/courses/archive/spring08/cos511/scribe_notes/0204.pdf){: target="_blank" }
<span class="annotation">Short, readable proof of the no-free-lunch theorem with examples. The original Wolpert paper is also classic but harder going.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Shalev-Shwartz & Ben-David — Understanding ML <i class="fas fa-external-link-alt"></i>](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/){: target="_blank" }
<span class="annotation">Free textbook with the cleanest modern presentation of PAC learning, VC dim, and generalization bounds. Chapters 2–6 are essential.</span>

</li>
<li data-tier="indepth" markdown="1">

[Belkin et al. (2019) — Double Descent <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1812.11118){: target="_blank" }
<span class="annotation">"Reconciling modern ML practice and the bias-variance trade-off". The paper that brought the double-descent picture into the mainstream.</span>

</li>
<li data-tier="indepth" markdown="1">

[Jacot, Gabriel & Hongler (2018) — NTK <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1806.07572){: target="_blank" }
<span class="annotation">The Neural Tangent Kernel paper. Shows how infinite-width networks become kernel methods — connects deep learning to classical generalization theory.</span>

</li>
</ul>

</div>

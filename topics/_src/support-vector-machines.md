---
title: Support Vector Machines — ML Resources Hub
eyebrow_text: ← Theory · Kernel Methods
eyebrow_href: {{root}}theory.html
heading: Support Vector Machines
lead: Find the line (or curve) that splits two classes with the widest margin possible.
next_href: gaussian-processes.html
next_title: Gaussian Processes
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Find the boundary with the widest possible gap.** If you can draw multiple straight lines separating two classes, the SVM picks the one that leaves the most "elbow room" on either side. The points right on the edge of the gap are the *support vectors* — they alone determine the boundary.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Toggle the kernel · slide C (linear) or γ (RBF) · watch the boundary bend and the support vectors light up</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Kernel
                <select id="viz-svm-kernel"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Data
                <select id="viz-svm-data"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                C
                <input id="viz-svm-c" class="viz-classic-slider" type="range"></input>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                γ
                <input id="viz-svm-gamma" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-svm-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-svm-c-lbl">C = 1.00</span>
<span class="viz-classic-badge" id="viz-svm-gamma-lbl">γ = 3.0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-svm-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-svm-caption"></div>
</div>

<script src="{{root}}js/viz/svm.js"></script>

In **linear** mode, dashed lines mark the margin at f = ±1; outlined points are the support vectors. Drop the slider for *C* and watch the margin widen — the SVM tolerates more violations in exchange for a fatter buffer. Switch to **RBF** on the *Concentric* dataset and the boundary bends into a circle; raise γ to make it tighter (and more prone to overfit), lower it for a smoother fit.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Imagine two clusters of points on a 2D plane, one class on the left, one on the right. A regular classifier might draw any line between them. An SVM picks the line that's *maximally far* from both — the "fattest" possible separator. The intuition: leaving a wide margin should generalize better to new points than squeezing a line right up against the data.

When the classes aren't linearly separable in the original feature space, the **kernel trick** lets SVMs find a separator in an implicit higher-dimensional space without ever explicitly computing the coordinates. That's how SVMs handle non-linear problems.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Small to medium data (≲ 50k points)
- High-dimensional features (text, gene expression)
- You want a strong out-of-the-box classifier with sensible defaults
- Clean binary classification where the boundary is the interesting object

</div>

<div class="no" markdown="1">

### Skip it when

- Large data — SVMs scale poorly (≳ O(N²))
- You need calibrated probabilities (SVM scores aren't probabilities)
- You don't have a good kernel guess and don't want to grid-search
- Highly imbalanced classes — needs careful class-weight tuning

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scale", StandardScaler()),    # SVMs are scale-sensitive
    ("svm",   SVC(kernel="rbf", C=1.0, gamma="scale")),
]).fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

</div>

<div class="level-next">
<span>Want the max-margin math and the kernel trick?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Soft-margin SVM</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \min_{\mathbf{w}, b} \;\tfrac{1}{2}\|\mathbf{w}\|^2 \;+\; C \sum_{i=1}^{N} \max\!\left(0,\, 1 - y_i\,(\mathbf{w}^\top \mathbf{x}_i + b)\right) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`w, b`normal vector and offset of the separating hyperplane

</li>
<li markdown="1">

`yi ∈ {−1, +1}`class label

</li>
<li markdown="1">

`max(0, 1 − …)`hinge loss — penalises points inside the margin or misclassified

</li>
<li markdown="1">

`C`regularization knob — large *C* = hard margin, small = wide margin tolerating mistakes

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{minimise}\;\; \tfrac{1}{2}\,(\text{size of weights})^2 \;+\; C \times \text{sum of hinge penalties across points} $$</span>

**In words.** An SVM looks for a separating line (or hyperplane) defined by weights `w` and an offset `b`. The first term keeps `w` small — and crucially, a smaller `w` means a *wider* margin between the line and the nearest points. The second term sums the hinge penalty: for each training point, you score `1 − label × (dot product of w and x + b)`; if the point is correctly classified and outside the margin, that score is negative and gets clipped to zero (no penalty), otherwise the point pays a linear cost. `C` dials the trade-off: large `C` punishes any margin violation hard (hard margin), small `C` tolerates violations in exchange for a wider buffer.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`size of weights`length of the weight vector — smaller weights mean a wider margin

</li>
<li markdown="1">

`hinge penalty`1 − label × score for each point, clipped at zero — costs nothing if you're correctly outside the margin

</li>
<li markdown="1">

`label`class identity, written as +1 or −1 so the sign trick works in the penalty

</li>
<li markdown="1">

`C`regularization knob — large C punishes violations hard, small C lets the margin widen

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The max-margin idea.** For a linearly separable problem, find *w* and *b* such that *y<sub>i</sub>(wᵀx<sub>i</sub> + b) ≥ 1* for all training points, and minimise *‖w‖*. Minimising ‖w‖ is equivalent to maximising the margin width 2/‖w‖. Only the points exactly on the margin (the "support vectors") matter; the rest could be deleted without changing the answer.

**Soft margin.** Real data isn't perfectly separable. The *hinge loss* allows some points inside the margin or misclassified, paying a linear penalty. The *C* hyperparameter trades margin width against training error.

**The kernel trick.** SVMs depend on the data only through inner products *x<sub>i</sub>ᵀx<sub>j</sub>*. Replace these with a kernel function *K(x<sub>i</sub>, x<sub>j</sub>) = φ(x<sub>i</sub>)ᵀφ(x<sub>j</sub>)*, where *φ* maps to a higher-dimensional space — without ever computing *φ* explicitly. Common kernels:

• **Linear:** *K = xᵀx'*. Fast baseline. Use for high-dim sparse data (text).<br/>
           • **Polynomial:** *K = (xᵀx' + c)<sup>d</sup>*. Captures feature interactions up to degree *d*.<br/>
           • **RBF (Gaussian):** *K = exp(−γ‖x − x'‖²)*. The default. Infinite-dim feature space.<br/>
           • **Sigmoid:** historical; rarely used today.

**Scaling matters.** All distance-based kernels (RBF, polynomial) are sensitive to feature scales. Always standardize first.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Small or medium dataset where O(N²) training is acceptable
- You want a strong baseline with limited hyperparameter tuning
- High-dimensional sparse features (text classification was SVMs' golden era)
- You want non-linear decision boundaries without engineering features

</div>

<div class="no" markdown="1">

### Skip it when

- Data is large — SGD-trained linear models scale much better
- You need probability outputs (Platt scaling exists but is post-hoc)
- Multi-class with many classes — one-vs-rest gets expensive
- You want feature importance / explanations — SVMs are opaque

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Standard recipe: scale → SVM with grid-searched C, gamma
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svm",   SVC()),
])

grid = GridSearchCV(
    pipe,
    {
        "svm__kernel": ["rbf"],
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.1, 1.0],
    },
    cv=5, scoring="f1", n_jobs=-1,
).fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Test accuracy: {grid.score(X_test, y_test):.3f}")
```

</div>

<div class="level-next">
<span>Want the dual form, KKT conditions, and the scaling story?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Dual formulation</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \max_{\boldsymbol{\alpha}} \;\sum_{i} \alpha_i \;-\; \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j \, y_i y_j \, K(\mathbf{x}_i, \mathbf{x}_j) \qquad \text{s.t.}\; 0 \le \alpha_i \le C,\; \sum_i \alpha_i y_i = 0 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Dual variables `αi` — one per training point

</li>
<li markdown="1">

Non-zero `αi` ⟺ point *i* is a support vector

</li>
<li markdown="1">

Decision: `f(x) = sign(Σ αi yi K(xi, x) + b)` — only kernels with support vectors

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{maximise}\;\; \text{(sum of point weights)} \;-\; \tfrac{1}{2} \times \text{(sum over pairs of weighted kernel similarities)} \quad \text{subject to:}\; 0 \le \text{point weight} \le C,\; \text{labelled weights sum to zero} $$</span>

**In words.** Re-parameterise the problem so that instead of optimising over `w`, you optimise over one number `αi` (alpha sub i) per training point. Each `α` says how much that point matters; in the end most are zero, and only the handful with non-zero `α` — the *support vectors* — actually shape the boundary. The objective rewards spreading the `α`'s out (the sum) while penalising similar same-label points stacking up (the pairwise term, which uses the kernel `K` to measure similarity). The constraint `0 ≤ α ≤ C` caps any one point's influence; the sum-to-zero condition keeps the two classes balanced. New predictions only require kernel evaluations against the support vectors — that's the whole point of the dual.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`point weight`α — how much each training point pulls on the boundary; most are zero

</li>
<li markdown="1">

`support vector`a training point whose α is non-zero — only these shape the decision boundary

</li>
<li markdown="1">

`kernel similarity`K(xᵢ, xⱼ) — a measure of how similar two points are, possibly in a higher-dim space

</li>
<li markdown="1">

`C`cap on each α — same C as the soft-margin penalty in the primal

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Primal vs. dual.** The primal optimises over *w* (size = number of features). The dual optimises over *α* (size = number of training points). Use the primal when features ≪ samples; use the dual when features ≫ samples (where the kernel trick lives). The KKT conditions give the explicit relationship: *w = Σ α<sub>i</sub>y<sub>i</sub>φ(x<sub>i</sub>)*, summing only over support vectors.

**Kernel choice as inductive bias.** A kernel encodes "what does similar mean?". RBF treats all features symmetrically; if some matter more, scale them differently. Polynomial encodes specific interaction orders. String kernels, graph kernels, and learned kernels (deep kernels, NTK) extend SVMs to structured data — though for most of these, modern neural approaches have won.

**Probability calibration.** SVM decision scores aren't probabilities. *Platt scaling* fits a logistic regression on the SVM outputs using a held-out set: *P(y=1|x) ≈ σ(A·f(x) + B)*. sklearn does this when you pass `probability=True` (which slows training noticeably). For honest probabilities, prefer logistic regression directly.

**Scaling SVMs.** Standard SVMs are O(N²)–O(N³). For larger data: (1) *LinearSVC* uses a different solver and scales to millions; (2) approximate kernels (Nyström, random Fourier features) project data into a finite-dim space, then train a linear model; (3) *SGDClassifier* with hinge loss = linear SVM via stochastic gradient descent, scales to streams.

**Why SVMs lost.** Once feature engineering was replaced by representation learning (deep nets) and tabular wins went to gradient boosting, SVMs lost their main strongholds. They're still relevant for: small-data classification with strong kernel priors, one-class anomaly detection (one-class SVM), and as a theoretical object — VC-dimension and margin bounds came from SVM theory.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Small-data classification with domain-knowledge kernels (e.g. string / graph kernels)
- One-class SVMs for novelty / anomaly detection
- Linear SVMs at scale via `LinearSVC` or SGD-hinge
- Theoretical work where margin bounds matter

</div>

<div class="no" markdown="1">

### Skip it when

- You're in modern deep-learning territory — representations + simple heads dominate
- Probabilistic outputs are first-class — go logistic / GP / Bayesian
- Big tabular — boosted trees almost always win
- You don't want to tune *C* and *γ* — RF is more forgiving

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Approximate RBF kernel + linear model — scales to millions of points
big_svm = Pipeline([
    ("scale", StandardScaler()),
    ("rbf",   RBFSampler(gamma="scale", n_components=500, random_state=0)),
    ("clf",   SGDClassifier(loss="hinge", alpha=1e-4, max_iter=20)),
]).fit(X_train, y_train)

print(f"Test acc: {big_svm.score(X_test, y_test):.3f}")
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

[scikit-learn — SVM <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/svm.html){: target="_blank" }
<span class="annotation">Practical reference with the kernel zoo, scaling tips, and the difference between SVC / NuSVC / LinearSVC.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[ESL — chapter 12 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">The statistical-learning treatment: derivation, dual, kernels, and connections to penalised regression. The clearest single source.</span>

</li>
<li data-tier="indepth" markdown="1">

[Joachims (1998) — Text Categorization with SVMs <i class="fas fa-external-link-alt"></i>](https://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf){: target="_blank" }
<span class="annotation">The paper that made SVMs the default for text classification. Useful for understanding why high-dim sparse SVMs work so well.</span>

</li>
<li data-tier="indepth" markdown="1">

[Wikipedia — Sequential Minimal Optimization <i class="fas fa-external-link-alt"></i>](https://en.wikipedia.org/wiki/Sequential_minimal_optimization){: target="_blank" }
<span class="annotation">SMO — the algorithm at the heart of fast SVM training (used by libsvm and sklearn). Concise overview.</span>

</li>
</ul>

</div>

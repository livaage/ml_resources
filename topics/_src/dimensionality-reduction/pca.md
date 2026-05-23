---
title: Principal Component Analysis (PCA) — ML Resources Hub
eyebrow_text: ← Theory · Dimensionality Reduction
eyebrow_href: {{root}}theory.html
heading: Principal Component Analysis (PCA)
lead: The classical linear projection — find the directions of greatest variance.
prev_href: ../dimensionality-reduction.html
prev_title: Dimensionality Reduction
next_href: t-sne.html
next_title: t-SNE
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Find the axes that capture the most variation, project everything onto them.** PCA rotates your coordinate system so the first axis is the one with the most spread, the second is the next-most while being orthogonal to the first, and so on. Throw away the low-variance axes; you've reduced dimensions with the smallest possible loss.

</div>

<article class="tldr-body" markdown="1">

The parent [Dimensionality Reduction](../dimensionality-reduction.html) page has the interactive viz — drag the projection axis and watch the histogram width tell you how much variance lives along that direction. PCA, analytically, picks the axis that maximises that width.

**What you get.** A new orthogonal basis (the principal components, PCs) ordered by variance. The "explained variance ratio" tells you how much information each PC carries. Usually the first 2–10 PCs are enough to capture most of the signal in real datasets.

**Why it works.** The PCs are eigenvectors of the covariance matrix; eigenvalues are the variances. Equivalently, the top PCs are the left singular vectors of the data matrix. The math is exact, the result is unique (up to sign), the algorithm runs in seconds.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You need linear dimensionality reduction (compression, denoising)
- You want to visualise high-D data quickly
- Decorrelating features before another model
- Whitening data for stable training
- Quick "is there structure?" sanity check

</div>

<div class="no" markdown="1">

### Skip it when

- Structure is genuinely non-linear (use t-SNE / UMAP / autoencoder)
- You need interpretable individual dimensions (PCs are mixtures)
- Variance ≠ information for your problem (e.g., sparse + high-cardinality)
- Features are on wildly different scales — standardise first, or it gets dominated

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always standardise first
X_std = StandardScaler().fit_transform(X)

# Project to 2D for visualisation
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_std)
print(f"Explained variance: {pca.explained_variance_ratio_}")
# array([0.43, 0.21])  — first two PCs carry 64% of the variance

# Pick enough components to keep 95% of variance
pca_95 = PCA(n_components=0.95)
X_red = pca_95.fit_transform(X_std)
print(f"Kept {pca_95.n_components_} components out of {X.shape[1]}")
```

</div>

<div class="level-next">
<span>Want the SVD connection, kernel PCA, and pitfalls?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">PCA via SVD</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ X = U \Sigma V^\top, \quad \text{PCs} = V, \quad \text{scores} = U\Sigma $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`V`columns are principal components (right singular vectors)

</li>
<li markdown="1">

`UΣ`each row is one example's coordinates in the new basis

</li>
<li markdown="1">

`σ_i²/(n-1)`variance along the *i*-th component

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{data} \;=\; \text{rotation}_1 \times \text{stretch} \times \text{rotation}_2, \quad \text{principal components} \;=\; \text{rotation}_2, \quad \text{coordinates} \;=\; \text{rotation}_1 \times \text{stretch} $$</span>

**In words.** SVD splits any matrix into three pieces: two rotations (`U` and `V`) with a diagonal stretch (`Σ`, sigma) in between. For PCA we care about `V` — its columns are the principal directions, ordered from most spread to least. The `UΣ` product gives the new coordinates of each row in that rotated basis. The `σ_i² / (n − 1)` formula just converts a singular value into the variance it captures along that direction. Truncating `V` to the top *k* columns gives the best *k*-dim linear summary of the data.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`data`centered feature matrix (one row per example)

</li>
<li markdown="1">

`rotation`orthonormal matrix — turns axes without changing distances

</li>
<li markdown="1">

`stretch`diagonal matrix of singular values, biggest first

</li>
<li markdown="1">

`principal components`columns of *V* — directions ranked by spread

</li>
<li markdown="1">

`coordinates`each row's new position in PC space (the "scores")

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Eigendecomposition view.** Compute the covariance matrix *C = X<sup>T</sup>X / (n−1)*, find its eigenvectors and eigenvalues. The eigenvectors are the PCs; eigenvalues are the variances along them. Numerically less stable than SVD on a tall matrix, so libraries use SVD.

**SVD view.** Decompose *X = UΣV<sup>T</sup>* directly. The right singular vectors *V* are the PCs; *σ<sub>i</sub>² / (n − 1)* are the variances. Works on any matrix (not just square), more stable than computing *X<sup>T</sup>X*.

**Probabilistic PCA.** Treat PCA as a latent-variable model: *x = Wz + μ + ε* with *z ~ N(0, I)*, *ε ~ N(0, σ²I)*. Recovers standard PCA in the limit σ → 0; gives a proper likelihood for things like model selection.

**Kernel PCA.** Apply PCA in a higher-dimensional space implicitly via a kernel. Captures non-linear structure (e.g., concentric circles unfold). Doesn't give an inverse projection back to input space; less useful for compression, more useful for visualisation / features.

**Sparse PCA.** Force most loadings in each PC to be zero — yields interpretable components (each PC is a small set of original features). Trades off some variance captured for interpretability.

**Scaling pitfalls.** Without standardisation, PCA is dominated by whichever feature has the largest range. Always `StandardScaler` first unless your features are already on the same scale and you have a reason to keep them that way.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, SparsePCA

# Numerically stable PCA via SVD (the sklearn default)
pca = PCA(n_components=10, svd_solver="auto")
X_red = pca.fit_transform(X_std)

# Kernel PCA — non-linear projection via RBF kernel
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
X_kpca = kpca.fit_transform(X_std)

# Sparse PCA — interpretable components (each PC uses few features)
spca = SparsePCA(n_components=5, alpha=1.0)
X_spca = spca.fit_transform(X_std)
print("Sparsity:", (spca.components_ != 0).mean(axis=1))   # fraction non-zero per PC
```

</div>

<div class="level-next">
<span>Want randomised SVD, online PCA, and connections to other methods?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Randomised SVD</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ X \approx Q (Q^\top X), \quad Q \in \mathbb{R}^{n \times k}, \;\; Q^\top Q = I_k $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Project *X* onto a random low-dim subspace *Q*

</li>
<li markdown="1">

Compute SVD on the projection — much cheaper

</li>
<li markdown="1">

Halko, Martinsson & Tropp (2011) — standard for large-scale PCA

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{data} \;\approx\; \text{random sketch} \times (\text{random sketch}^\top \times \text{data}) $$</span>

**In words.** Exact SVD on a huge matrix is expensive. The randomised trick is to pick a small random "sketch" matrix `Q` with just *k* columns (the number of components you actually want, plus a small buffer), and project the data onto it. The columns of `Q` are orthonormal — that's what `Q⊤Q = I` says — meaning they form a nice basis for a random *k*-dim subspace. SVD on the much smaller projected matrix is cheap, and the result captures the top *k* components of `X` almost as well as exact SVD would.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`random sketch (Q)`a tall, thin matrix with orthonormal columns drawn from a random subspace

</li>
<li markdown="1">

`k`target number of components (plus a small over-sampling buffer)

</li>
<li markdown="1">

`Q⊤Q = I`columns of *Q* are orthonormal — a clean low-dim basis

</li>
<li markdown="1">

SVD on the projection: cheap; result: near-exact top-*k* components

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Randomised SVD.** For very large matrices, exact SVD is prohibitive. Randomised methods project onto a small random subspace, then do SVD on the projection. `sklearn.decomposition.PCA(svd_solver="randomized")`. *O(n²k)* vs *O(n³)*; competitive accuracy when only top-*k* components matter.

**Online / incremental PCA.** Stream the data; update the PCA components without seeing all of it at once. `IncrementalPCA`. Useful when the dataset doesn't fit in memory or arrives over time.

**PCA vs Linear Discriminant Analysis (LDA).** PCA maximises variance, unsupervised. LDA maximises between-class variance relative to within-class, supervised. For classification, LDA preserves more class-discriminative information; for visualisation or compression, PCA is more general.

**PCA vs autoencoders.** A linear autoencoder with bottleneck dim *k* trained on MSE recovers (the span of) the top-*k* PCs. Adding non-linearities makes the autoencoder strictly more expressive — non-linear PCA, essentially. See the [Autoencoder](../neural-networks/autoencoder.html) page.

**Robust PCA.** Decompose *X = L + S* where *L* is low-rank and *S* is sparse (outliers). Candès et al. (2011). Useful when a small fraction of entries is corrupted; standard PCA is sensitive to outliers, robust PCA isn't.

**The variance ≠ signal issue.** PCA preserves variance, which isn't always the same as preserving the signal you care about. Counterexample: a small class-discriminative direction can be dwarfed by a large class-irrelevant direction. Standardise; consider supervised methods (LDA, supervised UMAP) if relevant.

**Numerical considerations.** Centering matters: PCA assumes mean-zero data. `X - X.mean(axis=0)` first. SVD is stable; eigendecomposition on the covariance matrix can lose precision when condition number is high.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.decomposition import IncrementalPCA

# Streaming / incremental PCA — fit in batches
ipca = IncrementalPCA(n_components=20, batch_size=1000)
for batch in load_in_batches(X, size=1000):
    ipca.partial_fit(batch)
X_red = ipca.transform(X)

# Randomised SVD for very wide / tall matrices
pca = PCA(n_components=50, svd_solver="randomized", random_state=0)
pca.fit(X)

# Robust PCA — sparse outliers separated from low-rank structure
# pip install pyrpca
from pyrpca import RobustPCA
rpca = RobustPCA(max_iter=200)
L, S = rpca.fit_transform(X)         # L is low-rank, S is sparse outliers
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

[Setosa.io — PCA <i class="fas fa-external-link-alt"></i>](https://setosa.io/ev/principal-component-analysis/){: target="_blank" }
<span class="annotation">Beautiful interactive primer. Drag the projection axis and watch what the model picks. Still the best single intuition source.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Elements of Statistical Learning — Ch. 14.5 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">Hastie, Tibshirani &amp; Friedman's chapter — the reference for PCA, ICA, and related projection methods.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — Decomposing Signals in Components <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/decomposition.html){: target="_blank" }
<span class="annotation">Practical reference: PCA, IncrementalPCA, SparsePCA, KernelPCA, TruncatedSVD — APIs and worked examples.</span>

</li>
<li data-tier="indepth" markdown="1">

[Halko, Martinsson & Tropp (2011) — Randomized Algorithms <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/0909.4061){: target="_blank" }
<span class="annotation">The reference paper for randomised SVD. Foundational for any PCA at scale.</span>

</li>
</ul>

</div>

---
title: Dimensionality Reduction — ML Resources Hub
eyebrow_text: ← Theory · Model Families
eyebrow_href: ../theory.html
heading: Dimensionality Reduction
lead: Compress high-dimensional data to a smaller representation that preserves what matters.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Most high-dim data lives on a lower-dim "shape".** A photo has millions of pixels, but only a few hundred numbers can describe most of what's in it. Dimensionality reduction finds those few numbers.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide θ to spin the projection axis — watch the 1D histogram widen and narrow; "Find PC1" snaps to the max-variance angle</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Data
                <select id="viz-dr-data"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                θ
                <input id="viz-dr-angle" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-dr-find" type="button">Find PC1</button>
<button id="viz-dr-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-dr-angle-lbl">θ = 0°</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-dr-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-dr-caption"></div>
</div>

<script src="{{root}}js/viz/dimred.js"></script>

PCA is exactly this game played analytically: find the axis that maximises projected variance. The faint indigo cross-hairs are the principal axes the model found (PC1 and PC2); the orange axis is yours to spin. When yours matches PC1, the histogram is at its widest. The number under it is what people mean by "captured variance" — for the tilted dataset, one component recovers ~90% of the structure.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Two reasons to do this. First, **visualization**: you can't plot 50 dimensions, but you can plot 2. Algorithms like t-SNE and UMAP project high-dim data into 2D in a way that preserves which points are near each other.

Second, **compression and denoising**: removing noise dimensions can make downstream models faster and sometimes more accurate. **PCA** is the classical workhorse here — it finds the directions of greatest variance and projects onto them.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Visualizing high-dim data in 2D / 3D
- Speeding up downstream models on high-dim features
- Removing redundant correlated features
- Feature engineering for tabular ML

</div>

<div class="no" markdown="1">

### Skip it when

- You need every feature to be interpretable — projected dimensions are mixes
- Each feature already carries unique signal (no redundancy)
- You're modelling images / text — let the model learn the embedding
- You need to reconstruct the original space exactly

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize first — PCA is variance-based, scale-sensitive
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2).fit(X_scaled)
X_2d = pca.transform(X_scaled)

print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Cumulative:          {pca.explained_variance_ratio_.cumsum()}")
```

</div>

<div class="level-next">
<span>Want PCA's math and the t-SNE / UMAP story?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">PCA via SVD</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ X_{\text{centered}} \;=\; U \Sigma V^\top, \qquad \text{principal components} = V_{:,1{:}k} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`V`right singular vectors — the principal directions in feature space

</li>
<li markdown="1">

`Σ`singular values — magnitudes; squared ∝ variance along each direction

</li>
<li markdown="1">

Top *k* columns of *V* give the best *k*-dim linear approximation in the *L*<sub>2</sub> sense

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{centered data} \;=\; \text{rotation}_1 \times \text{stretch} \times \text{rotation}_2, \qquad \text{principal directions} \;=\; \text{first } k \text{ columns of rotation}_2 $$</span>

**In words.** Singular Value Decomposition splits any matrix into three pieces: two rotations and a stretch in between. `U` and `V` are the rotations, `Σ` (sigma — diagonal matrix) holds the stretch factors in decreasing order. The columns of `V` point along the directions of biggest spread in your data; the diagonal entries of `Σ` tell you how much spread there is along each. Keep just the first *k* directions and you've got the best linear summary of the data in *k* dimensions, where "best" means smallest reconstruction error in the squared-distance sense.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`centered data`features with the mean subtracted off each column

</li>
<li markdown="1">

`rotation`orthonormal matrix — turns axes without changing distances

</li>
<li markdown="1">

`stretch`diagonal matrix of singular values, biggest first

</li>
<li markdown="1">

`principal directions`the columns of *V* — top *k* are the most informative axes

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**PCA** finds the linear projection that preserves the most variance. It's optimal under squared-error loss, fast (closed form via SVD), interpretable (each component is a weighted combination of original features), and computes uncertainty / reconstruction errors. Downside: linear only — can't unfold curved manifolds.

**t-SNE** models pairwise similarities in high-dim space as Gaussian, in low-dim space as Student-t (heavier tails). Minimizes KL divergence between the two. Preserves *local* neighborhood structure beautifully but distorts global distances. Stochastic — different runs give different layouts. Don't use for downstream modelling, only visualization.

**UMAP** is t-SNE's faster successor. Builds a fuzzy topological graph, then optimizes a low-dim embedding to match it. Faster than t-SNE, better at preserving global structure, deterministic-ish with a seed. Has become the default visualization method for embeddings.

**ICA** (Independent Component Analysis) finds projections that are statistically *independent*, not just uncorrelated. Recovers true source signals when they're non-Gaussian — classic application is blind source separation (the "cocktail party problem").

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- **PCA:** denoising, decorrelating features for downstream models
- **t-SNE / UMAP:** 2D visualization of high-dim embeddings
- **ICA:** recovering independent sources (EEG, audio separation)
- You need to compress before clustering or kNN

</div>

<div class="no" markdown="1">

### Skip it when

- You need to preserve global distances exactly (t-SNE / UMAP distort them)
- Non-linear structure with no global trend (PCA fails)
- Features are sparse — PCA densifies and destroys sparsity benefits
- Embeddings will be used as features downstream — t-SNE / UMAP outputs aren't designed for that

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Reduce to 50 dims with PCA first — t-SNE / UMAP are slow on raw high-dim data
X_pca = PCA(n_components=50, random_state=0).fit_transform(X)

# Then non-linear embedding to 2D
X_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X_pca)
X_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X_pca)

# PCA -> t-SNE / UMAP is the standard recipe for visualizing image / text embeddings
```

</div>

<div class="level-next">
<span>Want the manifold view, autoencoders, and pitfalls?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Probabilistic PCA</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathbf{x} \;=\; W\,\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\varepsilon}, \quad \mathbf{z} \sim \mathcal{N}(0, I),\; \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 I) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

PCA = max-likelihood limit (*σ*² → 0) of this latent-variable model

</li>
<li markdown="1">

Generalizes to factor analysis, mixture-of-PPCA, and variational autoencoders

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{observed point} \;=\; \text{loading matrix} \times \text{hidden code} \;+\; \text{mean} \;+\; \text{noise} $$</span>

**In words.** Probabilistic PCA tells a generative story: each high-dim observation *x* comes from a low-dim hidden code *z*, multiplied by a learned matrix *W*, shifted by a mean *μ*, plus a bit of Gaussian noise *ε* (epsilon). The hidden code *z* is itself drawn from a standard Gaussian (`N(0, I)` means "mean zero, unit covariance"). When the noise variance *σ²* shrinks to zero, the maximum-likelihood solution for *W* coincides with classical PCA — so PCA is a special, noise-free case of this richer model. Keeping the noise term lets you do model selection, mixture-of-PCAs, and connects directly to variational autoencoders.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`observed point (x)`the data point in original (high-dim) space

</li>
<li markdown="1">

`hidden code (z)`the low-dim latent vector, sampled from a unit-Gaussian prior

</li>
<li markdown="1">

`loading matrix (W)`maps latent to observed

</li>
<li markdown="1">

`mean (μ)`shift

</li>
<li markdown="1">

`noise (ε)`Gaussian residual with variance *σ²*

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**PCA pitfalls.** (1) Scale matters — components are dominated by high-variance features unless you standardize. (2) Linear only — fails on curved manifolds like the Swiss roll. (3) Outliers can hijack the top components — use robust PCA (decomposition into low-rank + sparse) if outliers are a concern.

**Choosing *k*.** Cumulative explained variance to a threshold (e.g. 95%). Scree-plot elbow. Cross-validated reconstruction error. For downstream classification, choose *k* via held-out task performance — the variance criterion isn't task-aligned.

**Kernel PCA** uses the kernel trick to do PCA in a feature space implicitly defined by a kernel. Solves the linearity limitation but loses the interpretability of axes.

**Manifold learning.** Isomap (geodesic distances + classical MDS), LLE (local linear embedding), Laplacian eigenmaps. Each makes a different assumption about local structure. t-SNE and UMAP dominate in practice but the older methods have stronger theoretical guarantees for specific manifold types.

**Autoencoders** are the deep-learning generalization. Encoder maps *x → z*, decoder maps *z → x̂*; train to minimize reconstruction error. Bottleneck dimension = embedding dim. VAEs add a probabilistic prior on *z*, giving you a generative model.

**t-SNE caveats.** Distances in the t-SNE plot are *not* meaningful. Cluster sizes are *not* meaningful. The number of clusters is dictated by `perplexity` as much as by the data. Always re-run with several perplexity values to check stability — see Wattenberg et al. "How to Use t-SNE Effectively".

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Robust PCA — large outliers expected
- Kernel PCA — known kernel inductive bias and small / moderate data
- Variational autoencoders — generative model + embedding in one
- Topological data analysis — UMAP's preserved structure feeds persistent homology

</div>

<div class="no" markdown="1">

### Skip it when

- Embedding must be stable under small data changes (t-SNE / UMAP are sensitive)
- You need to invert the embedding back to original space (manifold methods can't)
- Out-of-sample extension matters — t-SNE doesn't natively transform new points
- You care about preserved geodesic distances — only Isomap claims that

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, d_latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_latent, 256), nn.ReLU(),
            nn.Linear(256, d_in),
        )
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

model = Autoencoder(d_in=X.shape[1], d_latent=16)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    x_hat, _ = model(X)
    loss = loss_fn(x_hat, X)
    opt.zero_grad(); loss.backward(); opt.step()

# Embedding is model.enc(X)
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

[Wattenberg et al. — How to Use t-SNE Effectively <i class="fas fa-external-link-alt"></i>](https://distill.pub/2016/misread-tsne/){: target="_blank" }
<span class="annotation">Interactive demonstration of common t-SNE failure modes (cluster sizes, distances, ghost clusters). Essential before publishing any t-SNE plot.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[UMAP documentation <i class="fas fa-external-link-alt"></i>](https://umap-learn.readthedocs.io/){: target="_blank" }
<span class="annotation">Excellent tutorials with intuition for each hyperparameter. The maintained reference implementation.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Shlens — A Tutorial on PCA <i class="fas fa-external-link-alt"></i>](https://www.cs.cmu.edu/~elaw/papers/pca.pdf){: target="_blank" }
<span class="annotation">Best concise PCA derivation. Two pages of math, no fluff. Read this before anything else.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — Decomposition <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/decomposition.html){: target="_blank" }
<span class="annotation">Comparison of PCA, kernel PCA, ICA, NMF, dictionary learning. Each has a worked example.</span>

</li>
<li data-tier="indepth" markdown="1">

[McInnes et al. (2018) — UMAP paper <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1802.03426){: target="_blank" }
<span class="annotation">The UMAP algorithm in mathematical detail — topological foundations, fuzzy simplicial sets, the cost function.</span>

</li>
</ul>

</div>

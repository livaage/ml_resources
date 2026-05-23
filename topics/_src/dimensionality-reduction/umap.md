---
title: UMAP — ML Resources Hub
eyebrow_text: ← Theory · Dimensionality Reduction
eyebrow_href: {{root}}theory.html
heading: UMAP
lead: Like t-SNE but faster, more global structure preserved, and the embedding works on new points.
prev_href: t-sne.html
prev_title: t-SNE
next_href: ica.html
next_title: ICA
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Build a weighted k-NN graph in the original space; lay it out in 2D so the graph topology is preserved.** Same goal as t-SNE — visualise clusters — but a different objective and a different optimisation. Faster, more reproducible, and the resulting embedding model can transform new points without re-fitting.

</div>

<article class="tldr-body" markdown="1">

The interactive on the [t-SNE page](t-sne.html) shows the broadly-similar dynamics — local attractions, long-range repulsions, clusters coalescing. UMAP's force balance is slightly different (it preserves *more* global structure) and the algorithm is faster, but the visual story is the same.

**Practical differences.** UMAP usually runs 2–10× faster than t-SNE. The embeddings often have more meaningful global structure — cluster *positions* sometimes mean something (unlike t-SNE). UMAP can `transform` new points after fitting; t-SNE can't (without parametric variants).

**The two main knobs.** `n_neighbors` (default 15) — small for tight local structure, large for global structure. `min_dist` (default 0.1) — small for tight clusters, large for spread.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Pick UMAP over t-SNE when

- You have > 50k examples — it'll be much faster
- You need to embed new examples after fitting
- Global cluster positions might matter for your interpretation
- You want a more reproducible embedding

</div>

<div class="no" markdown="1">

### Stick with t-SNE when

- You're following a paper / convention that uses t-SNE
- The dataset is small (< 10k); the speed difference is negligible
- You want to compare against published t-SNE plots

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import umap
import matplotlib.pyplot as plt

# Standard 2D embedding for visualisation
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,     # local ↔ global structure trade-off
    min_dist=0.1,       # tight clusters ↔ spread points
    metric="euclidean", # cosine often better for embeddings
    random_state=0,
)
X_2d = reducer.fit_transform(X_high_d)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=4)

# Embed new points without re-fitting
X_test_2d = reducer.transform(X_test_high_d)
```

</div>

<div class="level-next">
<span>Want the topological story, the cross-entropy objective, & parametric UMAP?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">UMAP objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \min_y \;\; \mathrm{CE}\!\left(\,\nu_{\text{high}}\,\|\, \nu_{\text{low}}\right) = -\sum_{ij} \nu_{ij}^h \log \nu_{ij}^l + (1 - \nu_{ij}^h) \log (1 - \nu_{ij}^l) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`νh`fuzzy-set membership in the high-D k-NN graph

</li>
<li markdown="1">

`νl`fuzzy-set membership in the 2D embedding

</li>
<li markdown="1">

Cross-entropy instead of t-SNE's KL — keeps both attractive and repulsive terms balanced

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \min_y \;\; -\!\sum_{ij} \left[\, \text{high-D edge strength} \times \log(\text{low-D edge strength}) \;+\; (1 - \text{high-D edge strength}) \times \log(1 - \text{low-D edge strength}) \,\right] $$</span>

**In words.** Each pair of points has a number between 0 and 1 saying how strongly they're connected in the high-D nearest-neighbour graph (`ν^h` — "nu high") and the corresponding strength in the 2D layout (`ν^l` — "nu low"). The objective is the standard **cross-entropy** between these two: it's small when the two strengths match across every pair. The first piece pulls strong-in-input pairs together; the second piece (using `1 − ν`) pushes weak-in-input pairs apart. Unlike t-SNE's KL — which only penalises one direction — UMAP balances both forces, which is why it preserves more global structure.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`high-D edge strength (ν^h)`how strongly two points are linked in the input k-NN graph (0 to 1)

</li>
<li markdown="1">

`low-D edge strength (ν^l)`same number for the 2D embedding

</li>
<li markdown="1">

`cross-entropy`standard "distance" between two distributions, sensitive to both directions

</li>
<li markdown="1">

Both attractive (similar pairs) and repulsive (dissimilar pairs) forces are balanced

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The topological framing.** McInnes et al. (2018). Build a "fuzzy simplicial complex" — a weighted graph where edges are k-NN neighbours with weights for how strongly connected they are. Find a 2D layout whose own fuzzy complex matches the high-D one as closely as possible (cross-entropy). The categorical-theory motivation is intimidating; the implementation is approachable.

**Symmetrisation.** The high-D graph is asymmetric (k-NN is). UMAP symmetrises via a t-conorm: *ν<sub>ij</sub> = ν<sub>i|j</sub> + ν<sub>j|i</sub> − ν<sub>i|j</sub> ν<sub>j|i</sub>*. Avoids the "neighbour of, but not neighbour back" inconsistency.

**Optimisation.** Stochastic gradient descent on a subsample of pairs each step. Negative sampling for the repulsive term — sample non-neighbours rather than summing over all. Both make UMAP much faster than naive t-SNE.

**n_neighbors.** Effective local scale. Small (≤ 5) emphasises very local structure — clumps that are close together get pulled apart aggressively. Large (≥ 50) preserves more global structure — the relative positions of clusters become more meaningful.

**min_dist.** Minimum distance between points in the output. Small (0.0–0.1) gives tight clusters with hard boundaries. Large (0.5+) gives spread-out, "blobby" clusters. Affects appearance more than information content.

**Parametric UMAP.** Train a neural net to produce the embedding. Now you can transform new points exactly (rather than approximating). Used heavily in semi-supervised settings where the embedding is half the model.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import umap

# Different parameters for different goals
local  = umap.UMAP(n_neighbors=5,  min_dist=0.0).fit_transform(X)   # tight clusters
global_ = umap.UMAP(n_neighbors=100, min_dist=0.5).fit_transform(X)  # global picture
balanced = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X) # default

# Supervised UMAP — use labels to guide the embedding
sup = umap.UMAP(n_neighbors=15, n_components=2).fit_transform(X, y=labels)

# Parametric UMAP for embedding new points exactly
from umap.parametric_umap import ParametricUMAP
p_umap = ParametricUMAP().fit(X_train)
X_test_2d = p_umap.transform(X_test)
```

</div>

<div class="level-next">
<span>Want supervised UMAP, manifold theory, & comparisons to PaCMAP, TriMap, MDE?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The attractive-repulsive force balance</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ F_{\text{attr}}(d) \propto \frac{-2ab\, d^{2b-1}}{1 + a d^{2b}}, \quad F_{\text{rep}}(d) \propto \frac{2b}{(0.001 + d^2)(1 + a d^{2b})} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`a, b`parameters fit to make the output similarity curve match the chosen *min_dist*

</li>
<li markdown="1">

Same idea as t-SNE's force law but with different "long-range push" dynamics

</li>
<li markdown="1">

More balanced — preserves more global structure

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{attractive force}(d) \;\propto\; \frac{-2ab \cdot d^{2b-1}}{1 + a \cdot d^{2b}}, \qquad \text{repulsive force}(d) \;\propto\; \frac{2b}{(0.001 + d^2)(1 + a \cdot d^{2b})} $$</span>

**In words.** Each pair of points feels two competing forces driven by their 2D distance *d*: an **attractive** one pulling neighbours together, and a **repulsive** one pushing non-neighbours apart. The constants `a` and `b` aren't manually set — they're fit once at the start so that the resulting "low-D similarity curve" matches whatever `min_dist` you chose. The plain version just keeps the same formula because the math *is* the meaning here: an algebraic relationship between distance and force. The crucial property versus t-SNE is that the repulsive term shrinks more gradually with distance, so far-apart clusters still nudge each other — which is why UMAP preserves more global structure.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`d`distance between two points in the 2D embedding

</li>
<li markdown="1">

`a, b`auto-fit constants that calibrate the curve to your `min_dist`

</li>
<li markdown="1">

`attractive force`pulls neighbouring points together — dominant at short range

</li>
<li markdown="1">

`repulsive force`pushes non-neighbours apart — fades more slowly than t-SNE's, hence more global structure

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Supervised UMAP.** Pass `y` to `fit` — UMAP incorporates label information by weighting same-class pairs more strongly. The resulting embedding separates classes more cleanly. Useful for classification visualisation; risk of "the plot looks great but doesn't reflect the data" if you over-rely on it.

**Aligned UMAP.** Embed multiple datasets into the *same* coordinate system. Useful for batch correction (single-cell RNA across labs), or for "before/after" comparisons of model training. `umap.AlignedUMAP`.

**Density-preserving (densMAP).** Most UMAP / t-SNE embeddings squash low-density and high-density regions into similar-sized blobs. densMAP adds a local-density-preserving term so the visual reflects density. `umap.UMAP(densmap=True)`.

**Manifold assumption.** UMAP (and t-SNE, and LLE, and …) assume the data lives on or near a low-dimensional manifold. When this is true (images of faces, MNIST digits) they work beautifully. When the data is genuinely high-D (sparse text features, random embeddings) they impose structure that isn't there.

**UMAP vs PaCMAP vs TriMap vs MDE.** The family of modern manifold-learning algorithms keeps growing. PaCMAP claims better balance of local + global; TriMap uses triplets (anchor, near, far); MDE is a flexible optimiser-based framework. UMAP is still the most widely adopted; the others are worth knowing for specific properties.

**The "embedding hyperparameter sweep" trap.** Trying enough `n_neighbors` × `min_dist` combinations will eventually produce a plot that "looks right". This is overfitting to your intuition rather than to the data. Pick parameters by an external criterion (downstream classification accuracy, expert agreement); report your defaults.

**Reproducibility.** UMAP is more reproducible than t-SNE (lower variance across seeds) but still seed-dependent. Always set `random_state`; consider averaging across seeds.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import umap
from umap.umap_ import nearest_neighbors

# Aligned UMAP — embed two datasets in the same space
aligned = umap.AlignedUMAP(n_neighbors=15)
relations = [{i: i for i in range(min(len(X_a), len(X_b)))}]  # anchor pairs
embeddings = aligned.fit_transform([X_a, X_b], relations=relations)
X_a_2d, X_b_2d = embeddings

# Density-preserving UMAP
dens = umap.UMAP(densmap=True, n_neighbors=15).fit_transform(X)

# Pre-compute k-NN to share across hyperparameter sweeps
nn = nearest_neighbors(X, n_neighbors=30, metric="euclidean",
                       metric_kwds={}, angular=False, random_state=0)
for nn_count in [5, 15, 30]:
    emb = umap.UMAP(n_neighbors=nn_count, precomputed_knn=nn).fit_transform(X)
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

[UMAP Documentation <i class="fas fa-external-link-alt"></i>](https://umap-learn.readthedocs.io/){: target="_blank" }
<span class="annotation">The reference implementation. McInnes &amp; Healy's docs are unusually good — they explain the algorithm thoroughly with worked examples.</span>

</li>
<li data-tier="indepth" markdown="1">

[McInnes, Healy & Melville (2018) — UMAP <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1802.03426){: target="_blank" }
<span class="annotation">The original UMAP paper. The math is intimidating; the practical recipe is the last few sections.</span>

</li>
<li data-tier="intuition" markdown="1">

[Coenen & Pearce — Understanding UMAP <i class="fas fa-external-link-alt"></i>](https://pair-code.github.io/understanding-umap/){: target="_blank" }
<span class="annotation">Google's interactive companion to UMAP, in the spirit of the t-SNE Distill piece. Hands-on tour of the hyperparameters.</span>

</li>
<li data-tier="intuition" markdown="1">

[Distill — Various manifold methods <i class="fas fa-external-link-alt"></i>](https://distill.pub/2019/visual-exploration-gaussian-processes/){: target="_blank" }
<span class="annotation">Distill has multiple essays on related projection methods (t-SNE, UMAP, MDS, Isomap). The visual style is the modern standard.</span>

</li>
</ul>

</div>

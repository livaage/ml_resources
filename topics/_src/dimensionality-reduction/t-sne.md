---
title: t-SNE — ML Resources Hub
eyebrow_text: ← Theory · Dimensionality Reduction
eyebrow_href: {{root}}theory.html
heading: t-SNE
lead: A non-linear projection that preserves local neighbourhoods — the visualisation tool for high-D clusters.
prev_href: pca.html
prev_title: PCA
next_href: umap.html
next_title: UMAP
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Force similar high-D points to be close in 2D; let dissimilar ones drift apart.** Compute pairwise similarities in the original space and in the embedding; minimise their difference. The result is a 2D scatter where local structure — clusters, manifolds — is preserved. *t*-SNE is the go-to for visualising embeddings: digit clusters, gene expression, language embeddings.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Watch random 2D points pull into clusters as t-SNE minimises the KL between high-D and low-D similarities</span>
</div>
<div class="viz-classic-controls">
<button id="viz-tsne-step" type="button">Step</button>
<button id="viz-tsne-play" type="button">Play</button>
<button id="viz-tsne-reset" type="button">Reset</button>
<span class="viz-classic-badge" id="viz-tsne-step-lbl">step 0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-tsne-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-tsne-caption"></div>
</div>

<script src="{{root}}js/viz/tsne.js"></script>

5 "high-D clusters" (we cheat and pretend the colour-coding is the high-D label). The points start scattered randomly in 2D; each step pushes same-cluster points together and different-cluster points apart, following the t-SNE gradient. After ~200 steps the clusters are visible. Real t-SNE does the same on real high-D distances; here we use the colour as the ground-truth similarity to make the dynamics clear.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Pairwise similarities.** In the original space, similarities are Gaussian: *p<sub>j|i</sub> ∝ exp(−‖x<sub>i</sub> − x<sub>j</sub>‖² / 2σ<sub>i</sub>²)*. In the 2D embedding, similarities are Student-t (with heavier tails): *q<sub>ij</sub> ∝ (1 + ‖y<sub>i</sub> − y<sub>j</sub>‖²)<sup>−1</sup>*. Minimise KL(*P* ‖ *Q*) over *y<sub>i</sub>*.

**The Student-t in low dimensions is the trick.** Heavy tails let far-apart points stay far apart without huge gradients; without it, far points get crushed (the "crowding problem"). It's why t-SNE is t-SNE and not SNE.

**Perplexity.** The one knob worth knowing. Loosely the effective number of nearest neighbours each point considers. Default 30; lower → tighter local clusters, higher → more global structure. Try a few values.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Visualising clusters in high-D embeddings (digits, words, single-cell RNA)
- "Are there hidden groups in my data?"
- Inspecting the structure of learned representations
- Communicating high-D results to non-technical stakeholders

</div>

<div class="no" markdown="1">

### Beware

- Distances *between* clusters in the 2D plot are not meaningful
- Cluster sizes in the plot don't reflect real density
- Different runs give different embeddings (initialisation matters)
- Slow: O(N²) by default, O(N log N) with Barnes-Hut
- Not for downstream tasks — UMAP or PCA for that

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Standard t-SNE for embedding visualisation
tsne = TSNE(
    n_components=2,
    perplexity=30,       # the only knob worth touching
    init="pca",          # better starting point than random
    learning_rate="auto",
    random_state=0,
)
X_2d = tsne.fit_transform(X_high_d)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=4)
```

</div>

<div class="level-next">
<span>Want the KL math, Barnes-Hut, and the "don't read distances" rule?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">t-SNE objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \min_y \; \mathrm{KL}(P \;\|\; Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`pij`symmetrised Gaussian similarities in the input space

</li>
<li markdown="1">

`qij`Student-t similarities in the 2D embedding

</li>
<li markdown="1">

Asymmetric KL — heavily penalises "close in P but far in Q"

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \min_y \;\; \sum_{i \neq j} \;\; \text{high-D similarity} \times \log\!\left(\frac{\text{high-D similarity}}{\text{low-D similarity}}\right) $$</span>

**In words.** For every pair of points *i* and *j*, you have two numbers: how close they are in the original high-D space (`p_ij`) and how close they are in your 2D layout (`q_ij`). t-SNE moves the 2D points to make these two numbers agree. The `log(p/q)` term punishes pairs that are close in the input but far in the layout — much more than the opposite mismatch (this is what "asymmetric KL" means). KL divergence is a standard "distance" between two probability distributions; the `Σ` means "sum across every pair of points". The fact that big `p` matters more than big `q` is exactly why t-SNE preserves local neighbourhoods but distorts global distances.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`high-D similarity (p_ij)`how close points *i* and *j* are in input space (Gaussian)

</li>
<li markdown="1">

`low-D similarity (q_ij)`how close they are in the 2D layout (Student-t)

</li>
<li markdown="1">

`Σ across pairs`add up across every pair of points

</li>
<li markdown="1">

`log ratio`punishes "close in input but far in layout" much more than vice versa

</li>
<li markdown="1">

Net effect: local neighbourhoods preserved, global distances distorted

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Perplexity, properly.** For each *i*, the bandwidth σ<sub>i</sub> is chosen so the perplexity of the conditional distribution *p<sub>·|i</sub>* equals a chosen value (default 30). Effectively says "use σ<sub>i</sub> so each point has ~perplexity effective neighbours". Adapts to local density automatically.

**Asymmetric KL.** *p<sub>ij</sub> log(p<sub>ij</sub>/q<sub>ij</sub>)* punishes pairs that are close in *P* but far in *Q* much more than the reverse. Preserves local structure at the cost of global structure.

**Initialisation.** Random init can give wildly different embeddings; PCA init is now the default in scikit-learn and gives more reproducible, often-better results. Always use PCA init.

**Barnes-Hut t-SNE.** Van der Maaten (2014). Approximates pairwise interactions with a tree structure — *O(N log N)* instead of *O(N²)*. Scales to ~100k points without pain. Used by default in scikit-learn for > 5000 examples.

**The "don't read distances" rule.** Distortwitter et al. have a brilliant Distill piece on this. t-SNE deliberately distorts global geometry to preserve local. Two clusters that look close in the plot may be far in input space. Two clusters of similar size in the plot may have wildly different actual sizes.

**Reproducibility.** t-SNE is non-deterministic without a fixed seed. Different seeds give different embeddings, even with the same data. Always set `random_state`; consider running multiple seeds and averaging the visual impression.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.manifold import TSNE

# For larger datasets, openTSNE is faster + has more knobs
# pip install openTSNE
from openTSNE import TSNE as OpenTSNE

tsne = OpenTSNE(
    perplexity=30,
    metric="cosine",        # often better for embeddings
    n_iter=750,
    n_jobs=-1,              # parallelise
    initialization="pca",
)
embedding = tsne.fit(X_high_d)        # returns the fit object
X_2d = embedding.transform(X_high_d)  # not strictly needed; embedding is the result

# Try multiple perplexities and inspect
for p in [5, 30, 100]:
    X_2d = TSNE(perplexity=p, init="pca", random_state=0).fit_transform(X)
    plt.subplot(1, 3, ...); plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=2)
```

</div>

<div class="level-next">
<span>Want the gradient derivation, parametric t-SNE, and modern alternatives?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">t-SNE gradient</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \frac{\partial \mathrm{KL}}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij}) (y_i - y_j) (1 + \|y_i - y_j\|^2)^{-1} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Attractive force when *p<sub>ij</sub> > q<sub>ij</sub>*, repulsive when *p<sub>ij</sub> < q<sub>ij</sub>*

</li>
<li markdown="1">

The *(1 + ‖y<sub>i</sub> − y<sub>j</sub>‖²)<sup>−1</sup>* term gives long-range repulsion → spreads clusters

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{force on point } i \;=\; 4 \sum_j \text{(similarity mismatch)} \times \text{(direction from } i \text{ to } j) \times \frac{1}{1 + \text{distance}^2} $$</span>

**In words.** Each 2D point moves under a sum of pairwise forces from every other point. The "similarity mismatch" `(p_ij − q_ij)` says how wrong the current layout is for this pair: positive means "they should be closer", negative means "they should be further apart". The vector `y_i − y_j` gives the direction between them. The denominator `1 + distance²` shrinks the force as points get farther apart — but only slowly, so points that are *too* close repel each other hard, and points that are far don't pull in too strongly. This balance is what spreads clusters out without crushing them.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`force on point i`how the gradient says to move the 2D position of point *i*

</li>
<li markdown="1">

`similarity mismatch`positive → attractive force, negative → repulsive

</li>
<li markdown="1">

`direction`vector from *i* to *j*

</li>
<li markdown="1">

`1 / (1 + distance²)`shrinks slowly with distance — gives long-range repulsion that spreads clusters

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Optimisation tricks.** Early exaggeration: multiply *p<sub>ij</sub>* by ~12 for the first ~250 iterations. Helps form clusters quickly. Then turn it off; refine. Standard in every modern t-SNE implementation.

**Parametric t-SNE.** Van der Maaten (2009). Train a neural network to produce the embedding. Lets you embed new points (vanilla t-SNE can't). Trades simplicity for the ability to transform unseen data.

**UMAP as a t-SNE alternative.** Faster, preserves more global structure, supports inverse transforms with care. See the next page. The community is gradually shifting from t-SNE to UMAP, but t-SNE is still the more "battle-tested" workhorse for visualisation.

**The crowding problem.** In high dimensions, the volume scales as *r<sup>d</sup>*, so most points are far apart. In 2D there's no room to put them all. Student-t's heavy tails give "room" for distant points without piling them up at the boundary; Gaussian in 2D fails.

**Diagnostics.** Wattenberg et al. (Distill 2016) — multiple worked examples of what t-SNE can and can't tell you. The two main pitfalls: cluster sizes don't reflect density; cluster distances don't reflect input distance.

**Combining with classifiers.** Use t-SNE for visualisation only; don't feed the 2D coordinates into a downstream classifier. The coordinates aren't unique, aren't stable, and aren't necessarily structured the way the classifier needs.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from openTSNE import TSNE
import matplotlib.pyplot as plt

# Parametric / embedding model — embed new points later
tsne = TSNE(perplexity=30, metric="cosine", n_jobs=-1)
embedding = tsne.fit(X_train)

# Now embed new points
X_test_2d = embedding.transform(X_test)

# Compare with PCA init vs random init across multiple seeds
seeds = [0, 1, 2, 3]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, seed in enumerate(seeds):
    for row, init in enumerate(["pca", "random"]):
        emb = TSNE(initialization=init, random_state=seed).fit(X)
        axes[row, col].scatter(emb[:, 0], emb[:, 1], c=y, s=2)
        axes[row, col].set_title(f"{init}, seed {seed}")
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

[Wattenberg et al. (2016) — How to Use t-SNE Effectively <i class="fas fa-external-link-alt"></i>](https://distill.pub/2016/misread-tsne/){: target="_blank" }
<span class="annotation">The Distill piece. Interactive examples of every common t-SNE misreading. Required reading before drawing conclusions from any t-SNE plot.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[van der Maaten & Hinton (2008) — t-SNE <i class="fas fa-external-link-alt"></i>](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf){: target="_blank" }
<span class="annotation">The original t-SNE paper. The gradient and optimisation tricks are all here.</span>

</li>
<li data-tier="indepth" markdown="1">

[openTSNE <i class="fas fa-external-link-alt"></i>](https://opentsne.readthedocs.io/){: target="_blank" }
<span class="annotation">The fastest modern Python t-SNE implementation. Parametric, parallel, with all the modern tricks.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn — TSNE <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html){: target="_blank" }
<span class="annotation">The standard reference. Defaults updated in 0.22 (PCA init) and 1.2 (learning_rate="auto") to match best practice.</span>

</li>
</ul>

</div>

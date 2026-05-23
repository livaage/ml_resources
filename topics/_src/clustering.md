---
title: Clustering Algorithms — ML Resources Hub
eyebrow_text: ← Theory · Model Families
eyebrow_href: ../theory.html
heading: Clustering Algorithms
lead: Group similar items together without labels — k-means, DBSCAN, hierarchical, and friends.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Find natural groups in your data without telling the algorithm what to look for.** No labels, no target variable. The algorithm decides what "similar" means based on a distance metric you provide.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Step through Lloyd's algorithm — assign points, move centroids, repeat — and watch where k-means fails</span>
</div>
<div class="viz-classic-controls">
<button id="viz-cluster-step" type="button">Step</button>
<button id="viz-cluster-auto" type="button">Auto</button>
<button id="viz-cluster-reset" type="button">Reset</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Dataset
                <select id="viz-cluster-data"></select>
</label>
<select id="viz-cluster-k" aria-label="number of clusters"></select>
<span class="viz-classic-badge" id="viz-cluster-phase">iter 0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-cluster-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-cluster-caption"></div>
</div>

<script src="{{root}}js/viz/clustering.js"></script>

Lloyd's algorithm alternates two steps: **assign** each point to its nearest centroid, then **move** each centroid to the mean of its assigned points. Step through and watch the centroids settle. Try the *Two moons* dataset to see why k-means is a poor fit for non-convex clusters — it slices the moons clean in half because it only ever draws straight Voronoi boundaries.
{: .viz-intro }

<article class="tldr-body" markdown="1">

The most famous algorithm is **k-means**: pick *k* in advance, then iteratively assign each point to the nearest cluster center and move each center to the average of its assigned points. Simple, fast, but biased toward equal-size spherical clusters.

Other approaches: **DBSCAN** finds clusters of arbitrary shape based on local density (and labels sparse regions as noise). **Hierarchical clustering** builds a tree of clusters at every scale, letting you pick the granularity afterward.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You suspect there are natural groups but have no labels
- Customer / user segmentation
- Exploratory data analysis on a new dataset
- Vector quantization / image compression

</div>

<div class="no" markdown="1">

### Skip it when

- You have labels — use supervised learning instead
- You don't know what distance metric makes sense for your data
- The data has no real cluster structure (then any algorithm invents some)
- You need to "explain" the clusters — clustering doesn't justify itself

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)

print("Cluster sizes:", [(km.labels_ == k).sum() for k in range(4)])
print("Cluster centers shape:", km.cluster_centers_.shape)

# Predict cluster for new points
new_labels = km.predict(X_new)
```

</div>

<div class="level-next">
<span>Want to see how these algorithms actually work?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">k-means objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \min_{\{\boldsymbol{\mu}_k\}} \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Ck`the set of points assigned to cluster *k*

</li>
<li markdown="1">

`μk`the centroid of cluster *k* (the mean of its points)

</li>
<li markdown="1">

Lloyd's algorithm alternates assignment and centroid update — guaranteed to converge to a local minimum

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{minimise}\;\; \text{total squared distance from each point to its cluster centre} $$</span>

**In words.** Pick *K* cluster centres so that, when every point is assigned to its nearest centre, the sum of squared distances from points to their centre is as small as possible. `μk` (mu sub k) is the location of centre *k* — for the optimal solution it's literally the mean of the points assigned to it. Lloyd's algorithm finds a local minimum by alternating two steps until things stop moving: assign each point to its nearest centre, then move each centre to the mean of its assigned points. The k-means++ initialiser improves which local minimum you land in.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`cluster centre`μ<sub>k</sub> — one of *K* locations the algorithm picks; ends up as the mean of its points

</li>
<li markdown="1">

`squared distance`(point − centre)², summed across all dimensions of *x*

</li>
<li markdown="1">

`total squared distance`summed over every point, then over every cluster — the quantity being minimised

</li>
<li markdown="1">

`K`number of clusters — you have to choose this in advance (try the elbow / silhouette method)

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**k-means.** Assumes spherical, equal-size, equal-variance clusters. Sensitive to initialization — use `k-means++` seeding and `n_init > 1` to mitigate. Picking *K*: elbow on inertia, silhouette score, or the gap statistic.

**DBSCAN** (Density-Based Spatial Clustering). Two parameters: `eps` (neighbourhood radius) and `min_samples`. A point is "core" if it has ≥ min_samples neighbours within eps; clusters grow from core points. Points outside any cluster are labelled noise. Handles arbitrary shapes; no *K* to choose. Struggles with varying densities.

**Hierarchical / agglomerative.** Start with each point its own cluster, then repeatedly merge the closest pair. Choose linkage: single (chain-like clusters), complete (compact), average, or Ward (minimum variance). Produces a dendrogram you can cut at any height.

**Spectral clustering.** Build a similarity graph, embed it via the eigenvectors of the graph Laplacian, then run k-means in that embedding. Handles non-convex shapes that k-means can't.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- **k-means:** roughly spherical groups, large data, fast iteration
- **DBSCAN:** arbitrary shapes, noise present, unknown *K*
- **Hierarchical:** you want to see structure at multiple scales
- **Spectral:** data lies on a manifold, similarity matters more than Euclidean distance

</div>

<div class="no" markdown="1">

### Skip it when

- Very high-dimensional data — distances become uninformative ("curse of dimensionality")
- Mixed-type features — clustering is metric-bound
- You need overlapping clusters (use GMM for soft assignment)
- Cluster validity is unknown — silhouette & gap statistic only help so much

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

# Compare three algorithms with silhouette
for name, algo in [
    ("k-means",     KMeans(n_clusters=4, n_init=10, random_state=0)),
    ("dbscan",      DBSCAN(eps=0.5, min_samples=5)),
    ("agglomerative", AgglomerativeClustering(n_clusters=4, linkage="ward")),
]:
    labels = algo.fit_predict(X)
    # silhouette is undefined if any single cluster
    if len(set(labels)) > 1:
        s = silhouette_score(X, labels)
        print(f"{name:20s} silhouette = {s:.3f}, n_clusters = {len(set(labels))}")
```

</div>

<div class="level-next">
<span>Want the embedding view and modern density-based methods?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Spectral clustering</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ L \;=\; D - W \quad\text{or}\quad L_{\text{sym}} = I - D^{-1/2} W D^{-1/2} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`W`similarity matrix (e.g. Gaussian kernel of pairwise distances)

</li>
<li markdown="1">

`D`diagonal degree matrix, *D<sub>ii</sub> = Σ<sub>j</sub> W<sub>ij</sub>*

</li>
<li markdown="1">

Embed via the smallest *k* eigenvectors of *L*, then cluster in that space

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{Laplacian} \;=\; \text{degree matrix} \;-\; \text{similarity matrix} \quad\text{(or its normalised form)} $$</span>

**In words.** Build a graph where every data point is a node and edges carry a similarity weight (the `W` matrix, often a Gaussian kernel of distances). The `degree matrix` `D` is a diagonal matrix whose entries are each node's total similarity to all others. The `Laplacian` `L = D − W` encodes graph structure; the symmetric normalised form just rescales it. The smallest eigenvectors of `L` give you an embedding where points connected by strong similarity end up close together — and running plain k-means in that embedding recovers non-convex clusters (rings, moons) that k-means in the original space can't find.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`similarity matrix`W — pairwise weights saying how similar two points are

</li>
<li markdown="1">

`degree matrix`D — diagonal matrix of each point's total similarity to all others

</li>
<li markdown="1">

`Laplacian`L — a graph operator; its small eigenvectors give cluster-aware coordinates

</li>
<li markdown="1">

`eigenvectors`special vectors of L that, taken together, form a coordinate system where similar points cluster naturally

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**k-means is a Voronoi quantizer.** It minimizes the variance within Voronoi cells. With high-dim data, every pair of random points is approximately equidistant (concentration of measure), which is why k-means quality degrades sharply above ~50 dimensions. Pre-embed via PCA or autoencoder before clustering.

**HDBSCAN.** A hierarchical variant of DBSCAN that doesn't require a global eps. Builds a tree of mutual-reachability distances and extracts stable clusters at varying densities. The de-facto modern default when you'd previously have used DBSCAN.

**Mean shift.** Non-parametric mode-seeking: each point climbs the density gradient toward a local mode. Number of clusters determined by the number of distinct modes. Bandwidth selection is the main hyperparameter.

**Validity indices.** Internal indices (silhouette, Davies-Bouldin, Calinski-Harabasz) score partitions without ground truth but assume some geometric notion of "good clustering". External indices (ARI, NMI) compare against a labelled ground truth — useful for benchmark datasets, less so in the wild.

**Deep clustering.** Joint learning of representations and cluster assignments (DEC, DeepCluster, contrastive methods). Outperforms classical methods on high-dim structured data (images, text) where the right embedding is task-specific.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- **HDBSCAN:** varying densities, unknown noise level — modern default
- **Deep clustering:** images, text, audio — learn the embedding too
- **Mean shift:** mode-finding in non-Gaussian densities
- Stability is a more important property than partition quality

</div>

<div class="no" markdown="1">

### Skip it when

- Very large data and you need streaming — most methods are batch
- You can't pre-compute or cheaply approximate the full similarity matrix
- You'd benefit from a generative model instead — fit a GMM or Bayesian non-parametric
- The point of the analysis is causal — clustering doesn't infer causes

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import hdbscan
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    cluster_selection_method="eom",   # "excess of mass" — good default
)
labels = clusterer.fit_predict(X_scaled)

# -1 = noise; cluster persistence scores indicate stability
print(f"Found {labels.max() + 1} clusters, {(labels == -1).sum()} noise points")
print(f"Cluster persistence: {clusterer.cluster_persistence_.round(3)}")
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

[scikit-learn — Clustering <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/clustering.html){: target="_blank" }
<span class="annotation">Visual comparison of all clustering algorithms on toy datasets. Always start here to pick a candidate.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[HDBSCAN documentation <i class="fas fa-external-link-alt"></i>](https://hdbscan.readthedocs.io/){: target="_blank" }
<span class="annotation">Exceptionally well-written. The "How HDBSCAN Works" guide is the clearest introduction to modern density-based clustering.</span>

</li>
<li data-tier="indepth" markdown="1">

[von Luxburg (2007) — Spectral Clustering tutorial <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/0711.0189){: target="_blank" }
<span class="annotation">Definitive introduction to spectral methods. Derivation, normalized vs. unnormalized variants, eigengap heuristic.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Wikipedia — Choosing K <i class="fas fa-external-link-alt"></i>](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set){: target="_blank" }
<span class="annotation">Survey of methods (elbow, silhouette, gap statistic, BIC). Useful summary when you need to defend your choice.</span>

</li>
</ul>

</div>

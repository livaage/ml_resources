---
title: Gaussian Mixture Models — ML Resources Hub
eyebrow_text: ← Theory · Probabilistic Models
eyebrow_href: ../theory.html
heading: Gaussian Mixture Models
lead: Density estimation and soft clustering via a weighted sum of Gaussian components.
prev_href: naive-bayes.html
prev_title: Naive Bayes
next_href: hidden-markov-models.html
next_title: Hidden Markov Models
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Each point belongs to multiple clusters at once.** A GMM models the data as coming from several Gaussian "blobs". Each point gets a probability of belonging to each blob — not a hard assignment like k-means.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Step through EM — ellipses rotate and stretch to fit the data, where k-means' spherical regions can't</span>
</div>
<div class="viz-classic-controls">
<button id="viz-gmm-step" type="button">Step</button>
<button id="viz-gmm-auto" type="button">Auto</button>
<button id="viz-gmm-reset" type="button">Reset</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Data
                <select id="viz-gmm-data"></select>
</label>
<select id="viz-gmm-k" aria-label="components"></select>
<span class="viz-classic-badge" id="viz-gmm-counter">iter 0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-gmm-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-gmm-caption"></div>
</div>

<script src="{{root}}js/viz/gmm.js"></script>

Each Step runs one EM iteration — first an *E step* assigns soft responsibilities *γ<sub>ik</sub>* to every (point, component) pair, then an *M step* re-estimates each component's mean, covariance, and weight from those responsibilities. Watch the ellipses rotate and stretch — that's full-covariance Gaussians earning their keep. Switch to the *Tilted ellipses* dataset and notice how a GMM fits the diagonal stretch that k-means would slice straight through.
{: .viz-intro }

<article class="tldr-body" markdown="1">

K-means says: "this point is in cluster 3, period." A GMM says: "this point is 80% in cluster 3, 20% in cluster 1." That uncertainty is useful when clusters overlap or have very different shapes and sizes — k-means struggles with both.

GMMs are also a *density estimator*. Once fit, you can ask "how likely is this point under my model?" — useful for anomaly detection (low-likelihood points are anomalies) or for generating new samples from the learned distribution.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Clusters overlap or have different shapes / sizes
- You want soft assignments, not hard ones
- You need a density estimate (for anomaly detection or sampling)
- You can pick a reasonable number of components

</div>

<div class="no" markdown="1">

### Skip it when

- Clusters aren't well-modelled by Gaussians (e.g. moons, spirals)
- Very high-dimensional data — covariance matrices get huge
- You truly want hard assignments — k-means is faster
- You don't know the number of components and don't want to use model selection

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=0).fit(X)

# Soft assignments (responsibilities)
proba = gmm.predict_proba(X)

# Hard assignments (for cluster labels)
labels = gmm.predict(X)

# Density estimate: log p(x) under the fitted mixture
log_density = gmm.score_samples(X)
```

</div>

<div class="level-next">
<span>Want to see EM and the math?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Key idea</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ p(\mathbf{x}) \;=\; \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`K`number of components

</li>
<li markdown="1">

`πk`mixing weight — prior probability of component *k*, sums to 1

</li>
<li markdown="1">

`μk, Σk`mean and covariance of component *k*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ p(\mathbf{x}) \;=\; \pi_1 \cdot \text{Gauss}_1(\mathbf{x}) \;+\; \pi_2 \cdot \text{Gauss}_2(\mathbf{x}) \;+\; \cdots \;+\; \pi_K \cdot \text{Gauss}_K(\mathbf{x}) $$</span>

**In words.** The probability of seeing a point `x` is a weighted sum of *K* Gaussian "blobs." Each Gaussian `Gaussk` (the script `𝒩` in the math version) is a bell-shaped density with its own centre `μk` (mu — the mean vector) and shape `Σk` (sigma — the covariance matrix that controls how it stretches). The **mixing weights** `πk` (pi) say how much each blob contributes; they're positive and add to 1, so you can read them as "probability this point came from blob *k*." Sample a point by first rolling a weighted die to pick a component, then drawing from that component's Gaussian.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`K`number of Gaussian blobs you're fitting

</li>
<li markdown="1">

`πk`weight of blob *k* — positive numbers that add to 1

</li>
<li markdown="1">

`μk`centre of blob *k* in feature space

</li>
<li markdown="1">

`Σk`shape/spread of blob *k* (a covariance matrix)

</li>
<li markdown="1">

`Gaussk(x)`the bell-curve density of blob *k* evaluated at the point *x*

</li>
</ul>

</div>

Each point is assumed to be generated by first picking a component (with probability *π<sub>k</sub>*), then sampling from that component's Gaussian. The parameters are fit by maximizing likelihood, but there's no closed form — we use EM.

</div>

<article class="tldr-body" markdown="1">

**EM algorithm.** Alternates two steps until convergence:

**E-step:** compute *responsibilities* — the posterior probability that point *i* came from component *k*, holding parameters fixed.

**M-step:** update component parameters (μ<sub>k</sub>, Σ<sub>k</sub>, π<sub>k</sub>) as weighted averages, where the weights are the responsibilities from the E-step.

Each iteration is guaranteed to increase the log-likelihood (or leave it unchanged). It converges to a local maximum — sensitive to initialization, so re-run with several random restarts and keep the best.

**Covariance types.** sklearn offers `full` (each component free Σ), `tied` (shared Σ), `diag` (diagonal Σ), and `spherical` (scalar variance). Less flexible = fewer parameters = faster but more biased.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Soft clustering with calibrated probabilities
- Density estimation in low / moderate dimensions
- Anomaly detection via low likelihood
- Generative sampling of new points like the data

</div>

<div class="no" markdown="1">

### Skip it when

- Non-Gaussian-shaped clusters (try kernel density or DBSCAN)
- K is unknown and you don't want to fit several and compare via BIC
- Very high dimensions — full covariance scales as O(d²) per component
- You need outlier-robust fitting (single outliers move μ a lot)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Pick the number of components via BIC
ks = range(1, 11)
bics = [
    GaussianMixture(n_components=k, n_init=5, random_state=0)
        .fit(X).bic(X)
    for k in ks
]
best_k = ks[int(np.argmin(bics))]
print(f"BIC-optimal K = {best_k}")

gmm = GaussianMixture(
    n_components=best_k, covariance_type="full",
    n_init=5, random_state=0,
).fit(X)

print(f"Mixing weights: {gmm.weights_.round(3)}")
print(f"Converged: {gmm.converged_}, iterations: {gmm.n_iter_}")
```

</div>

<div class="level-next">
<span>Want the variational / Dirichlet-process extensions?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">EM as ELBO maximization</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \log p(\mathbf{X}) \;\geq\; \mathbb{E}_{q(\mathbf{Z})}\!\left[\log p(\mathbf{X}, \mathbf{Z}) - \log q(\mathbf{Z})\right] \;=\; \mathcal{L}(q, \theta) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Z`latent component assignments

</li>
<li markdown="1">

`q(Z)`variational distribution over assignments

</li>
<li markdown="1">

E-step optimizes *q*; M-step optimizes *θ*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \log p(\text{data}) \;\geq\; \text{average over } q(\text{assignments}) \text{ of } \big[\log p(\text{data}, \text{assignments}) - \log q(\text{assignments})\big] $$</span>

**In words.** EM is solving a hidden-variable problem: every point really did come from *some* component, but we never get to see which one — that hidden choice is `Z`. The ELBO is a tractable lower bound on the log-likelihood of the data, parameterised by a guess `q(Z)` for which component each point came from. The **E-step** sets `q` to the posterior over assignments (the soft responsibilities `γik`) — this closes the gap to the true log-likelihood. The **M-step** then maximises the bound over the component parameters `θ = (π, μ, Σ)`, using those responsibilities as soft labels. Alternating these two steps is coordinate ascent on the ELBO — each iteration increases (or holds) the data likelihood.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`data`observed points *X*

</li>
<li markdown="1">

`assignments`hidden component labels *Z* — which blob each point came from

</li>
<li markdown="1">

`q(assignments)`the current guess at the distribution over hidden labels

</li>
<li markdown="1">

`avg over q`expectation taken with *q* as the probability distribution

</li>
<li markdown="1">

E-step: refresh *q*. M-step: refresh component parameters. Repeat.

</li>
</ul>

</div>

EM is coordinate ascent on the evidence lower bound (ELBO). The E-step makes the bound tight (*q* = posterior); the M-step pushes the bound up by re-fitting parameters.

</div>

<article class="tldr-body" markdown="1">

**Identifiability and label switching.** The likelihood is invariant under permutation of components — there are *K!* equivalent maxima. Standard MLE is fine, but Bayesian inference (especially MCMC) has to handle this; it's the canonical example of a non-identifiable model.

**Singularities.** If a component collapses onto a single data point, its covariance → 0 and the likelihood → ∞. Regularize with a small ridge added to each Σ, or use a Bayesian prior on Σ to keep it bounded away from singular.

**Bayesian GMM (Dirichlet Process).** Instead of fixing *K*, place a Dirichlet process prior over mixing weights — the model decides how many components to use. sklearn's `BayesianGaussianMixture` implements this via variational inference; it can effectively prune unused components by sending their weights toward zero.

**k-means as a limit.** Take a GMM with *π<sub>k</sub> = 1/K*, *Σ<sub>k</sub> = σ²I*, and let *σ → 0*. The E-step responsibilities collapse to hard 0/1 assignments and the M-step becomes the k-means centroid update. K-means is the small-noise limit of spherical GMM.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Generative modelling of low-dimensional continuous data
- Bayesian model selection — BIC, DIC, or a DP prior
- Soft clustering with uncertainty quantification
- Embedded as part of a larger probabilistic graphical model

</div>

<div class="no" markdown="1">

### Skip it when

- Data has heavy tails — use Student-t mixtures
- Strong asymmetry — Gaussian shape is wrong
- You need conditional density *p(y | x)* — use mixture-of-experts
- Highly multi-modal latent structure that's not blob-shaped

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.mixture import BayesianGaussianMixture

# Variational Bayesian GMM: set K large; unused components get tiny weight
bgmm = BayesianGaussianMixture(
    n_components=20,
    weight_concentration_prior_type="dirichlet_process",
    weight_concentration_prior=1.0 / 20,    # smaller -> sparser solution
    n_init=5, random_state=0,
).fit(X)

active = (bgmm.weights_ > 0.01).sum()
print(f"Effective number of components: {active}")
print(f"Top weights: {sorted(bgmm.weights_, reverse=True)[:active]}")
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

[scikit-learn — Gaussian mixture models <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/mixture.html){: target="_blank" }
<span class="annotation">Practical reference with comparison of covariance types and BIC-based model selection.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Bishop, PRML chapter 9 <i class="fas fa-external-link-alt"></i>](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/){: target="_blank" }
<span class="annotation">The canonical treatment: GMMs, EM, the ELBO view, and variational extensions. Read alongside chapter 10 for VI.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[ESL, chapter 8.5 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">Concise statistical-learning view, with the EM derivation and connections to soft k-means.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Wikipedia — EM algorithm <i class="fas fa-external-link-alt"></i>](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm){: target="_blank" }
<span class="annotation">Quick reference for the general EM pattern beyond GMMs (works for HMMs, missing-data models, etc.).</span>

</li>
</ul>

</div>

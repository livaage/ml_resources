---
title: PCA as a Generative Model — ML Resources Hub
eyebrow_text: ← Theory · Generative Models
eyebrow_href: {{root}}theory.html
heading: PCA & Probabilistic PCA
lead: The simplest generative model — a linear-Gaussian latent variable. The blueprint every model in this section generalises.
prev_href: ../generative-models.html
prev_title: Generative Models
next_href: vae.html
next_title: Variational Autoencoders
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Generate data by drawing a low-dimensional code, then mapping it up with a straight line plus noise.** Pick a latent point *z* from a standard Gaussian blob, stretch and rotate it through a matrix *W*, shift it, and sprinkle on a little isotropic noise — out comes a sample that looks like your data. That single recipe, *x = Wz + μ + ε*, **is** Probabilistic PCA. Everything else in this section keeps the recipe and replaces the straight line with something more expressive.

</div>

<article class="tldr-body" markdown="1">

You already met PCA as a [dimensionality-reduction](../dimensionality-reduction/pca.html) tool: find the directions of greatest variance, project onto them. That's the *encoder* half of the story. Here we turn it around and ask the generative question: **what process could have *produced* this data?**

**The generative view.** Imagine each data point started life as a handful of latent numbers *z* — a compact "code". A linear map *W* expands that code into the high-dimensional space, we add the mean *μ*, and reality adds a bit of measurement noise. Run that forward and you've *sampled* a new datapoint. Learn *W* and the noise level from data and you've fit a generative model.

**Why start here.** PCA is the "hydrogen atom" of generative modelling — the one case where everything is linear and Gaussian, so every quantity has a closed form. There's no adversary, no sampling loop, no intractable integral. Once you see generation as *latent code → decoder → data*, the [VAE](vae.html) is just "make the decoder a neural net", the [GAN](gan.html) is "drop the likelihood and train a critic", and [diffusion](diffusion.html) is "make the decoder a long denoising chain".

**The catch.** A straight line can only produce ellipsoidal Gaussian blobs. Real data lives on curved, multi-modal manifolds — faces, audio, language. The rest of this section is the story of replacing that line with curves while keeping the same "sample a latent, decode it" skeleton.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### The generative framing helps when

- You want a probabilistic model with a real likelihood (model selection, anomaly scores)
- Data is roughly Gaussian / ellipsoidal — PPCA is the *right* model, not just a baseline
- You need to handle missing values (the latent model imputes them naturally)
- You want to understand VAEs — PPCA is the exact special case to anchor on

</div>

<div class="no" markdown="1">

### It breaks down when

- Data is multi-modal or lives on a curved manifold (a line can't bend)
- You need sharp, realistic samples — Gaussian noise blurs everything
- The interesting structure is non-linear (use a [VAE](vae.html) or [diffusion](diffusion.html))
- You only want the projection, not a generative story — then plain [PCA](../dimensionality-reduction/pca.html) is simpler

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

# Probabilistic PCA as a *generator*: sample latent -> decode -> add noise
# (W, mu, sigma learned from data; here we just sample from a fitted model)
def sample_ppca(W, mu, sigma, n):
    k = W.shape[1]                      # latent dimension
    z = np.random.randn(n, k)           # 1. draw codes from N(0, I)
    x = z @ W.T + mu                    # 2. decode with the linear map
    x += sigma * np.random.randn(*x.shape)   # 3. add isotropic noise
    return x

# Fitting is just PCA: top-k eigenvectors of the covariance give W,
# and the *leftover* variance becomes the noise level sigma^2.
```

</div>

<div class="level-next">
<span>Want the actual likelihood, the EM fit, and the exact VAE connection?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The PPCA model</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ z \sim \mathcal{N}(0, I), \qquad x \mid z \sim \mathcal{N}(Wz + \mu,\; \sigma^2 I) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Latent prior is a unit Gaussian — a featureless blob of codes

</li>
<li markdown="1">

Decoder *Wz + μ* is **linear**; noise is isotropic with a single scale *σ²*

</li>
<li markdown="1">

Marginal likelihood is closed-form: *x ~ N(μ, WWᵀ + σ²I)*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{code} \sim \text{unit Gaussian}, \qquad \text{data} = \text{decoder}(\text{code}) + \text{noise} $$</span>

**In words.** Two steps. First, draw a low-dimensional `code` *z* from a plain unit Gaussian — no structure, just a round cloud of points centred at zero. Second, push that code through a `decoder` that here is just a matrix multiply plus a shift (`Wz + μ`), then add a dab of round `noise` of size *σ*. Because both steps are Gaussian and the map is linear, the data you generate is itself exactly Gaussian, with covariance `WWᵀ + σ²I` — the part `WWᵀ` is the structured spread the model captured, and `σ²I` is the leftover wiggle it couldn't be bothered to explain. That clean formula is the whole reason PPCA is solvable by hand.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`code`the latent vector *z*, drawn from N(0, I) — usually far fewer dims than the data

</li>
<li markdown="1">

`decoder`the linear map *Wz + μ*; *W*'s columns are the directions the code stretches along

</li>
<li markdown="1">

`noise`isotropic Gaussian of scale *σ* — the same in every direction

</li>
<li markdown="1">

`WWᵀ + σ²I`the resulting data covariance — structure plus uniform leftover variance

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**PCA falls out of the limit.** As the noise *σ² → 0*, the maximum-likelihood *W* spans exactly the top-*k* principal subspace — the same eigenvectors classical [PCA](../dimensionality-reduction/pca.html) gives you. So PCA is PPCA with the noise turned off. Keeping *σ²* finite is what buys you a proper probability density.

**You get a real likelihood.** Because *x ~ N(μ, WWᵀ + σ²I)*, you can evaluate *log p(x)* for any point. That powers things classical PCA can't do: principled model selection (how many components?), anomaly detection (low likelihood = outlier), and comparing fits across datasets.

**Posterior over codes.** Given a datapoint, the posterior *p(z | x)* is again Gaussian — a closed-form encoder. This is exactly the object a [VAE](vae.html) has to *approximate* with a neural network because its decoder is non-linear and the integral stops being tractable.

**Fit by EM (or eigen-decomposition).** Tipping & Bishop (1999) gave both: a direct solution via the data covariance's eigenvectors, and an EM algorithm that scales better and handles missing data — E-step infers codes, M-step updates *W* and *σ²*.

**Factor analysis is the cousin.** Allow the noise to differ per feature (a diagonal *Ψ* instead of *σ²I*) and you get factor analysis. PPCA is the special case where every feature shares one noise level.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.decomposition import PCA

# scikit-learn's PCA exposes the PPCA likelihood directly.
pca = PCA(n_components=10).fit(X)        # X: (n_samples, n_features)

# score() returns the average log-likelihood under the Gaussian PPCA model
print("avg log-likelihood:", pca.score(X))

# The estimated isotropic noise variance sigma^2 (mean of discarded eigenvalues)
print("noise variance:", pca.noise_variance_)

# Likelihood-based model selection: which k generalises best?
for k in [2, 5, 10, 20]:
    ll = PCA(n_components=k).fit(X_train).score(X_val)
    print(f"k={k:2d}  val log-likelihood={ll:.2f}")
```

</div>

<div class="level-next">
<span>Want the marginal-likelihood derivation, the rotation ambiguity, and where the line breaks?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Marginal likelihood</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ p(x) = \int p(x \mid z)\, p(z)\, dz = \mathcal{N}\!\big(x \;\big|\; \mu,\; C\big), \qquad C = WW^\top + \sigma^2 I $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Integrating out *z* is tractable because everything is Gaussian — a rare luxury

</li>
<li markdown="1">

The MLE for *W* is *U<sub>k</sub>(Λ<sub>k</sub> − σ²I)<sup>½</sup>R* — top eigenvectors, scaled, times any rotation *R*

</li>
<li markdown="1">

The free rotation *R* is why PPCA axes aren't individually identified

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ p(\text{data}) = \text{average over all codes of } p(\text{data} \mid \text{code}) = \text{one big Gaussian with covariance } WW^\top + \sigma^2 I $$</span>

**In words.** To score a datapoint without knowing its code, you average the decoder's output probability over *every* possible code, weighted by the prior — that's the integral. For the [VAE](vae.html) this integral is hopeless and must be bounded by the ELBO; for PPCA it collapses to a single Gaussian because a Gaussian pushed through a linear map and blurred by Gaussian noise stays Gaussian. The covariance `WWᵀ + σ²I` says it all: structure captured by the decoder plus uniform leftover noise. The price of linearity is the catch in the next paragraph — the solution is only pinned down up to an arbitrary rotation of the latent axes.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`average over all codes`marginalising the latent *z* against its prior — the integral

</li>
<li markdown="1">

`one big Gaussian`the result stays Gaussian only because the decoder is linear

</li>
<li markdown="1">

`rotation R`you can spin the latent axes freely without changing *p(x)* — they aren't unique

</li>
<li markdown="1">

`σ²`recovered as the average of the eigenvalues you *didn't* keep

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The exact bridge to VAEs.** A VAE is PPCA with two upgrades: (1) the decoder *Wz + μ* becomes a neural network *f<sub>θ</sub>(z)*, so the data manifold can curve; (2) the now-intractable posterior *p(z | x)* is approximated by an encoder network *q<sub>φ</sub>(z | x)*. Train by maximising the ELBO — a lower bound on the same *log p(x)* PPCA computes exactly. Set the networks to be linear and the ELBO becomes tight: you recover PPCA. See the [VAE](vae.html) page for the full derivation. This is *the* reason this section opens with PCA.

**Rotation ambiguity.** Because *C = WWᵀ + σ²I* is unchanged if you replace *W* with *WR* for any orthogonal *R*, the latent axes aren't individually identifiable — only the subspace is. This is harmless for density modelling but means PPCA components aren't "interpretable" the way people sometimes hope. (Factor-analysis rotations like varimax exploit exactly this freedom.)

**Why a line is not enough.** PPCA can only ever produce one ellipsoidal blob. Two well-separated clusters? A spiral? A ring (like the viz on the [overview](../generative-models.html) page)? A single Gaussian smears straight across the gap. The fix is non-linearity — and that's the entire motivation for the neural decoders in the rest of the section. A *mixture* of PPCAs patches multi-modality cheaply; a VAE handles curvature directly.

**Missing data, for free.** Because it's a proper latent-variable model, PPCA handles partially-observed inputs by marginalising the missing entries in the E-step — a genuinely useful capability classical PCA lacks.

**Cost.** Closed-form via SVD is *O(nd·min(n,d))*; EM is *O(ndk)* per iteration and shines when *k ≪ d* or data is missing. Both are trivial next to training any neural generative model — the reason PPCA is still the first thing to try when the data might actually be Gaussian.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

# EM for PPCA — the algorithm a VAE generalises when the decoder goes non-linear.
def ppca_em(X, k, iters=100):
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    W = np.random.randn(d, k)
    sigma2 = 1.0
    for _ in range(iters):
        # E-step: posterior over codes  q(z|x) = N(M^-1 W^T xc, sigma2 M^-1)
        M = W.T @ W + sigma2 * np.eye(k)
        Minv = np.linalg.inv(M)
        Ez = Xc @ W @ Minv                         # expected codes
        Ezz = n * sigma2 * Minv + Ez.T @ Ez        # expected outer products
        # M-step: re-fit the linear decoder and the noise level
        W = (Xc.T @ Ez) @ np.linalg.inv(Ezz)
        sigma2 = (np.sum(Xc**2)
                  - 2*np.sum(Ez @ W.T * Xc)
                  + np.trace(Ezz @ (W.T @ W))) / (n * d)
    return W, mu, sigma2
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

[PCA, the deeper view <i class="fas fa-external-link-alt"></i>](../dimensionality-reduction/pca.html)
<span class="annotation">Our own dimensionality-reduction page — the encoder/projection side of the same model, with the interactive variance viz.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Tipping & Bishop (1999) — Probabilistic Principal Component Analysis <i class="fas fa-external-link-alt"></i>](https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis/){: target="_blank" }
<span class="annotation">The original PPCA paper. Derives the maximum-likelihood solution and the EM algorithm — short and very readable.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Bishop — Pattern Recognition and Machine Learning, Ch. 12 <i class="fas fa-external-link-alt"></i>](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/){: target="_blank" }
<span class="annotation">Continuous latent-variable models done properly: PPCA, factor analysis, and the EM machinery that VAEs inherit.</span>

</li>
<li data-tier="indepth" markdown="1">

[Kingma & Welling (2019) — An Introduction to Variational Autoencoders <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1906.02691){: target="_blank" }
<span class="annotation">Read §2 with PPCA in mind — the linear-Gaussian special case is the perfect anchor before the neural generalisation in the next page.</span>

</li>
</ul>

</div>

---
title: Gaussian Processes — ML Resources Hub
eyebrow_text: ← Theory · Kernel Methods
eyebrow_href: ../theory.html
heading: Gaussian Processes
lead: A probability distribution over functions. Predict with calibrated uncertainty, not just a point estimate.
prev_href: support-vector-machines.html
prev_title: Support Vector Machines
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Instead of one guess, give me a range.** A Gaussian process predicts both a value *and* how confident it is about that value. Where you have lots of nearby data, it's confident; where you don't, it widens its uncertainty.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Click to add observations · shift+click to remove · slide ℓ to change how far each point's influence reaches</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Kernel
                <select id="viz-gp-kernel"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                ℓ
                <input id="viz-gp-length" class="viz-classic-slider" type="range"></input>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                σₙ
                <input id="viz-gp-noise" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-gp-resample" type="button">Re-sample</button>
<button id="viz-gp-clear" type="button">Clear</button>
<span class="viz-classic-badge" id="viz-gp-length-lbl">ℓ = 0.25</span>
<span class="viz-classic-badge" id="viz-gp-noise-lbl">σₙ = 0.05</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-gp-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-gp-caption"></div>
</div>

<script src="{{root}}js/viz/gp.js"></script>

The bold line is the posterior mean; the shaded band is the ±2σ uncertainty. The thin lines are four sample functions drawn from the posterior — they show what kinds of functions are still consistent with your data. Clear all observations and you're seeing the *prior* — the GP's belief before any data. Add a point and watch the band collapse to it (within σₙ); the uncertainty inflates the further you go.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Most regression models give you one number per prediction. A Gaussian process gives you a number *and* an error bar — and the error bar is data-aware: it shrinks near your training points and grows when you predict far away from anything you've seen.

That uncertainty is the whole point. GPs are slow and don't scale to big data, but when you have a few hundred points and you care about *knowing what you don't know* — design of experiments, Bayesian optimization, robotics — they're the right tool.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You have small data (≲1000 points)
- Calibrated uncertainty matters
- You want a non-parametric model that adapts to the data
- You're doing Bayesian optimization

</div>

<div class="no" markdown="1">

### Skip it when

- You have a lot of data (GPs are O(N³))
- You only need point predictions, not uncertainty
- The signal is well-behaved and a simpler model would do
- You're working with images or text (use neural nets)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0))
gp.fit(X_train, y_train)

# Predict with uncertainty
y_mean, y_std = gp.predict(X_test, return_std=True)
```

</div>

<div class="level-next">
<span>Want to see what's actually happening under the hood?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Key idea</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ f(\mathbf{x}) \;\sim\; \mathcal{GP}\!\left(m(\mathbf{x}),\; k(\mathbf{x}, \mathbf{x}')\right) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`m(x)`mean function (often zero)

</li>
<li markdown="1">

`k(x,x')`kernel — encodes how similar outputs should be when inputs are similar

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{unknown function } f \;\sim\; \text{GP}(\text{mean function},\; \text{kernel}) $$</span>

**In words.** Read the `∼` as "is drawn from" — a Gaussian process is a probability distribution over *functions*, and `f` is a single sample from it. Two ingredients describe the distribution. The **mean function** `m(x)` is what the function looks like on average at input *x* (people usually set this to zero and let the data do the work). The **kernel** `k(x, x')` is a similarity score between two inputs: it controls how correlated the function's outputs are when the inputs are close. A smooth kernel produces smooth functions; a periodic kernel produces wavy ones. The whole machinery rests on one property — any finite set of function values you ask about is a jointly Gaussian random vector, which is exactly what makes the math close-form-tractable.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`f`the unknown function you're trying to learn — a single draw from the GP

</li>
<li markdown="1">

`mean function`the GP's prior expectation of *f(x)*; usually zero

</li>
<li markdown="1">

`kernel`a similarity function — nearby inputs → highly correlated outputs

</li>
<li markdown="1">

`∼`"is drawn from" — *f* is one sample from a distribution over functions

</li>
</ul>

</div>

A GP is a distribution over functions such that any finite set of function values is jointly Gaussian. The kernel determines what kind of functions are likely — smooth, periodic, rough, etc.

</div>

<article class="tldr-body" markdown="1">

The kernel *k* is the heart of the model. The most common one — the RBF kernel *k(x, x') = exp(−‖x − x'‖² / 2ℓ²)* — assumes smooth functions; nearby inputs produce similar outputs, with "nearby" controlled by the length scale ℓ.

Given training data, the posterior over *f*(*x*<sub>*</sub>) at a test point is also Gaussian, in closed form:

μ<sub>*</sub> = k<sub>*</sub><sup>T</sup>(K + σ²I)<sup>−1</sup>y     σ²<sub>*</sub> = k(x<sub>*</sub>,x<sub>*</sub>) − k<sub>*</sub><sup>T</sup>(K + σ²I)<sup>−1</sup>k<sub>*</sub>

That matrix inverse is the source of the O(N³) cost — it's what kills GPs on large data.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Small to medium data with strong prior beliefs about smoothness
- You want principled hyperparameter selection via marginal likelihood
- Active learning / Bayesian optimization (need uncertainty to pick next point)
- Time-series with seasonality (composable kernels)

</div>

<div class="no" markdown="1">

### Skip it when

- N > a few thousand — use sparse approximations or switch models
- Functions are non-stationary in ways the kernel can't capture
- You have categorical features (need custom kernels)
- Output is high-dimensional (vector-valued GPs are awkward)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# Composable kernel: amplitude * RBF + noise
kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,   # random restarts for marginal-likelihood optimization
    normalize_y=True,
).fit(X_train, y_train)

print("Fitted kernel:", gp.kernel_)
print("Log-marginal likelihood:", gp.log_marginal_likelihood(gp.kernel_.theta))

y_mean, y_std = gp.predict(X_test, return_std=True)
```

</div>

<div class="level-next">
<span>Want sparse approximations and the kernel-engineering tradeoffs?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Marginal likelihood (hyperparameter learning)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \log p(\mathbf{y} \mid X, \boldsymbol{\theta}) \;=\; -\tfrac{1}{2}\mathbf{y}^\top K_\theta^{-1}\mathbf{y} \;-\; \tfrac{1}{2}\log\!|K_\theta| \;-\; \tfrac{n}{2}\log 2\pi $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Kθ`covariance matrix induced by kernel hyperparameters *θ*

</li>
<li markdown="1">

fit term — `yTK-1y` penalizes poor data fit

</li>
<li markdown="1">

complexity term — `log|K|` penalizes overly-flexible kernels

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \log p(\text{targets} \mid \text{kernel params}) \;=\; -\tfrac{1}{2} \underbrace{\mathbf{y}^\top K^{-1} \mathbf{y}}_{\text{data fit}} \;-\; \tfrac{1}{2} \underbrace{\log |K|}_{\text{complexity}} \;-\; \tfrac{n}{2} \log 2\pi $$</span>

**In words.** This is the log-probability of the observed targets `y` under the GP, with the function *f* integrated away — leaving only the kernel hyperparameters `θ` (length scale, amplitude, noise) to tune. Three terms. The **data-fit** term `yTK−1y` punishes you when the targets don't lie comfortably in the kinds of functions the kernel allows — `K−1` is the inverse of the *n × n* kernel matrix at the training inputs. The **complexity** term `log|K|` (the log-determinant) punishes overly flexible kernels — a richer kernel matches anything but spreads probability thin. The final `n log 2π` is a normalisation constant that doesn't depend on *θ*. Maximise this whole expression over *θ* and you get a built-in Occam's razor — fit vs. complexity, balanced automatically.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`targets`training labels *y*

</li>
<li markdown="1">

`kernel params`hyperparameters *θ* like length scale, amplitude, noise

</li>
<li markdown="1">

`K`kernel matrix evaluated at every pair of training inputs

</li>
<li markdown="1">

`K−1`matrix inverse — the O(N³) cost lives here

</li>
<li markdown="1">

`log |K|`log-determinant of *K* — the complexity penalty

</li>
<li markdown="1">

`n`number of training points

</li>
</ul>

</div>

The log marginal likelihood automatically trades off fit vs. complexity — a built-in Occam's razor. Optimize it w.r.t. kernel hyperparameters to set length scales, amplitudes, and noise simultaneously.

</div>

<article class="tldr-body" markdown="1">

**Scaling.** The cubic cost of inverting the N×N kernel matrix is the headline limitation. Sparse approximations represent the GP via *m* ≪ N inducing points and reduce cost to O(Nm²). FITC and DTC are early variants; SVGP (variational, Hensman et al.) is the modern workhorse and scales to millions of points via stochastic optimization.

**Kernel composition.** Sums and products of valid kernels are valid kernels. This lets you encode structure: `RBF + Periodic` for time series with seasonality, `RBF × Linear` for trends, etc. Automatic kernel discovery (Duvenaud et al.) tries to compose these structures from data.

**Connection to neural networks.** An infinitely-wide neural network with random Gaussian weights converges to a GP (Neal, 1996). The NTK (neural tangent kernel) describes the GP that trained networks effectively act as in the infinite-width limit — connecting GPs to deep learning theory.

**Likelihood freedom.** The closed-form posterior requires Gaussian likelihood. For classification or other non-Gaussian likelihoods, approximate inference is needed: Laplace, EP (expectation propagation), or variational methods.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Bayesian optimization / experimental design
- Spatial statistics (kriging is a GP)
- You want interpretable hyperparameters (length scale = "how far is similar?")
- Multi-fidelity modelling — cheap and expensive evaluations

</div>

<div class="no" markdown="1">

### Skip it when

- You can't choose a sensible kernel (no inductive bias)
- You need extrapolation — RBF kernels decay to the prior mean
- Strict inference latency — sparse GPs are still slower than NNs
- Heavy-tailed noise — Gaussian likelihood is too sensitive to outliers

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import gpytorch, torch

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        self.mean   = gpytorch.means.ConstantMean()
        self.cov    = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.cov(x))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model      = ExactGP(X_train, y_train, likelihood)
mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Optimize marginal likelihood via Adam
model.train(); likelihood.train()
opt = torch.optim.Adam(model.parameters(), lr=0.1)
for _ in range(200):
    opt.zero_grad()
    loss = -mll(model(X_train), y_train)
    loss.backward()
    opt.step()
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

[Distill — A Visual Exploration of GPs <i class="fas fa-external-link-alt"></i>](https://distill.pub/2019/visual-exploration-gaussian-processes/){: target="_blank" }
<span class="annotation">Best intuition builder. Interactive widgets for kernels, conditioning, and the posterior. Start here.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Rasmussen & Williams — Gaussian Processes for ML <i class="fas fa-external-link-alt"></i>](https://gaussianprocess.org/gpml/){: target="_blank" }
<span class="annotation">The textbook. Free PDF. Comprehensive treatment from regression to classification to sparse approximations.</span>

</li>
<li data-tier="indepth" markdown="1">

[GPyTorch documentation <i class="fas fa-external-link-alt"></i>](https://docs.gpytorch.ai/en/stable/){: target="_blank" }
<span class="annotation">Modern GPU-accelerated GP library, with tutorials covering sparse and deep GPs. The library most people actually use.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[scikit-learn GP module <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/gaussian_process.html){: target="_blank" }
<span class="annotation">Good for small data and prototyping. No sparse GP support, but the API is the simplest.</span>

</li>
</ul>

</div>

---
title: Bayesian Inference — ML Resources Hub
eyebrow_text: ← Theory · Probabilistic Models
eyebrow_href: ../theory.html
heading: Bayesian Inference
lead: Update beliefs about parameters as you see data — get distributions, not point estimates.
prev_href: hidden-markov-models.html
prev_title: Hidden Markov Models
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Start with a belief, see some data, update.** Classical statistics gives you a single best estimate of a parameter (and maybe a confidence interval). Bayesian inference gives you a full *distribution* over the parameter — capturing everything you know and don't know about it after seeing the data.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Flip coins · adjust the prior · watch the posterior emerge as prior × likelihood (normalised)</span>
</div>
<div class="viz-classic-controls">
<button id="viz-bayes-flip-h" type="button">Flip H</button>
<button id="viz-bayes-flip-t" type="button">Flip T</button>
<button id="viz-bayes-flip-10" type="button">+10 flips</button>
<button id="viz-bayes-reset" type="button">Reset</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                α₀
                <input id="viz-bayes-alpha" class="viz-classic-slider" type="range"></input>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                β₀
                <input id="viz-bayes-beta" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-bayes-alpha-lbl">α₀ = 2.0</span>
<span class="viz-classic-badge" id="viz-bayes-beta-lbl">β₀ = 2.0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-bayes-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-bayes-caption"></div>
</div>

<script src="{{root}}js/viz/bayes.js"></script>

The dashed indigo curve is your prior — Beta(α₀, β₀). Each "Flip H" / "Flip T" click adds an observation. The bold orange curve is the posterior; it's literally *prior × likelihood* normalised. With no data the posterior is the prior; with a lot of data the posterior is dominated by the likelihood (the prior is "swamped"). The dashed grey curve in the middle is the likelihood at the current data, plotted to scale.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Suppose you want to know a coin's bias. After 10 flips you saw 7 heads. A classical estimate is 0.7 — period. A Bayesian says: "Before flipping, I believed the bias was probably around 0.5 (the *prior*). Now I've updated to a distribution centered slightly above 0.5 but with a lot of spread (the *posterior*) — I'm not very sure yet."

That distribution is the answer. Want a point estimate? Take its mean. Want uncertainty? Read off the credible interval. Want to predict? Average predictions over the posterior. The framework gives you principled answers to all of these.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You have small data and strong prior knowledge
- Honest uncertainty matters (clinical trials, A/B tests, forecasts)
- You need to combine evidence from multiple sources
- Decision-theoretic framing (loss × posterior → optimal action)

</div>

<div class="no" markdown="1">

### Skip it when

- You have a lot of data and the prior doesn't matter
- You only need a point estimate and don't care about uncertainty
- The model is complex and you can't afford MCMC / VI
- The audience won't accept "subjective" priors

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Coin-flip example: Beta(α, β) prior + Binomial likelihood -> Beta posterior
from scipy.stats import beta

alpha_prior, beta_prior = 2, 2          # weak prior around p=0.5
heads, tails = 7, 3                     # observed data

alpha_post = alpha_prior + heads
beta_post  = beta_prior  + tails

posterior = beta(alpha_post, beta_post)
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% credible interval: ({posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f})")
```

</div>

<div class="level-next">
<span>Want Bayes' rule and conjugate priors?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Bayes' rule</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} \;=\; \frac{\overbrace{p(\mathcal{D} \mid \theta)}^{\text{likelihood}} \;\overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`θ`parameters or hypotheses

</li>
<li markdown="1">

`D`observed data

</li>
<li markdown="1">

`p(D)`marginal likelihood — usually intractable, but it's just a normalizer over *θ*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{posterior} \;=\; \frac{\text{likelihood} \;\times\; \text{prior}}{\text{evidence}} $$</span>

**In words.** Bayes' rule tells you how to update what you believe about an unknown parameter `θ` after seeing data `D`. The **prior** `p(θ)` is what you believed before; the **likelihood** `p(D | θ)` is how plausible the data would be if a particular θ were true; multiply the two and you get the unnormalised posterior. The **evidence** `p(D)` in the denominator is just a number that makes everything sum (or integrate) to 1 — it doesn't change the *shape* of the answer, which is why people often ignore it and write "`posterior ∝ likelihood × prior`" with a proportionality sign.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`prior`your belief about the parameter before seeing any data

</li>
<li markdown="1">

`likelihood`how well a given parameter value explains the data you observed

</li>
<li markdown="1">

`posterior`your updated belief about the parameter after seeing the data

</li>
<li markdown="1">

`evidence`a normaliser that makes the posterior a valid probability distribution

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Conjugate priors.** If the prior and likelihood are a "conjugate pair", the posterior is in the same family as the prior — closed-form update, no numerical integration needed. Beta-Binomial (coin flips), Gaussian-Gaussian (mean estimation with known variance), Dirichlet-Multinomial (category proportions). Use these when you can.

**When conjugacy fails.** Most real models aren't conjugate. Two options:

**MCMC** (Markov chain Monte Carlo) generates samples from the posterior without computing the normalizer. Modern variants — Hamiltonian Monte Carlo (HMC), the No-U-Turn Sampler (NUTS) — handle high-dim continuous parameters well.

**Variational inference (VI)** turns inference into optimization: pick a tractable approximating family *q(θ)*, then optimize *q* to minimize KL divergence to the posterior. Faster than MCMC but biased — *q* can't capture posterior shapes it doesn't have the flexibility for.

**Predictive distributions.** Instead of plugging in a point estimate, average predictions over the posterior: *p(x<sub>new</sub> | D) = ∫ p(x<sub>new</sub> | θ) p(θ | D) dθ*. This automatically accounts for parameter uncertainty.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Small / heterogeneous data — priors regularize gracefully
- You need a coherent uncertainty propagation through several modelling steps
- Mixed-effects / hierarchical models — Bayesian framework is natural
- Decision making — combine posteriors with utility functions

</div>

<div class="no" markdown="1">

### Skip it when

- Likelihood is too expensive to evaluate repeatedly (no MCMC budget)
- Prior is hard to justify and the audience wants "no priors"
- Posterior is multi-modal and you don't have specialized samplers
- Real-time inference — MCMC is slow

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import pymc as pm

# Bayesian linear regression: model = α + β·x + noise
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta  = pm.Normal("beta",  mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = alpha + beta * x_obs
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    # NUTS sampler — adaptive HMC
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9)

# Posterior summaries
import arviz as az
print(az.summary(trace, var_names=["alpha", "beta", "sigma"]))
```

</div>

<div class="level-next">
<span>Want the MCMC mechanics, VI bounds, and the diagnostics?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Variational lower bound (ELBO)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \log p(\mathcal{D}) \;\geq\; \mathbb{E}_{q(\theta)}\!\left[\log p(\mathcal{D}, \theta) - \log q(\theta)\right] \;=\; \mathcal{L}(q) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Maximize *ℒ* over *q* — equivalent to minimizing *KL(q ∥ posterior)*

</li>
<li markdown="1">

Gap = KL divergence — when zero, *q* = posterior

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \log(\text{evidence}) \;\geq\; \text{average over } q \text{ of } \big[\log(\text{joint}) - \log q\big] $$</span>

**In words.** The true posterior is usually intractable, so we pick a simpler distribution `q(θ)` we *can* compute with and tune it to match. The **ELBO** (evidence lower bound) is a quantity that's always less than or equal to the log evidence `log p(D)` — the `≥` means "at least". The right-hand side averages `log(joint) − log q` across draws from `q`; the joint `p(D, θ)` is just `prior × likelihood`. Pushing the ELBO up is mathematically the same as pulling `q` closer to the true posterior (it shrinks the KL divergence between them). When the gap closes to zero, `q` matches the posterior exactly.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`q`tractable approximating distribution you're tuning

</li>
<li markdown="1">

`joint`prior × likelihood, written *p(D, θ)*

</li>
<li markdown="1">

`evidence`marginal likelihood of the data, *p(D)*

</li>
<li markdown="1">

`avg over q`expectation — draw θ from *q* many times and average

</li>
<li markdown="1">

Maximising the ELBO ≈ minimising the distance from *q* to the true posterior

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**MCMC choices.** Random-walk Metropolis is robust but mixes slowly. Gibbs sampling factorizes posterior into conditionals — efficient when conjugacy is partial. Hamiltonian Monte Carlo uses gradients to make informed proposals — much better mixing on continuous high-dim problems. NUTS auto-tunes HMC step size and trajectory length. Use NUTS by default in modern PPLs (PyMC, Stan, NumPyro).

**Convergence diagnostics.** R-hat compares within-chain to between-chain variance — should be near 1 for convergence. Effective sample size measures autocorrelation-adjusted information per draw. Energy plots reveal HMC pathologies. Trust nothing; always check.

**VI methods.** Mean-field VI factorizes *q(θ) = ∏ q(θ<sub>i</sub>)* — fast but blind to correlations. Black-box VI / SVI uses gradient-based optimization with Monte Carlo gradients (works for any model in a PPL). Normalizing flows give expressive *q* while keeping density evaluation tractable — closing the gap on MCMC for high-dim continuous posteriors.

**Prior sensitivity.** Sensitivity to priors should be checked. Tighter priors = stronger regularization. For hierarchical models, weakly informative priors on top-level scales (HalfNormal, HalfCauchy) prevent funnel pathologies in MCMC. *Reparameterize* centered models to non-centered when geometry is hostile.

**Model comparison.** Bayes factors are tempting but unstable for diffuse priors. Use information criteria — WAIC, LOO-CV via importance sampling (PSIS-LOO) — for predictive comparison.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Hierarchical / multilevel structure to exploit
- You need to propagate uncertainty through a pipeline of models
- Sparse signals where priors do the regularizing work
- Probabilistic programming gives you compositional model construction

</div>

<div class="no" markdown="1">

### Skip it when

- Big data & deep models — VI / MCMC don't scale; use SGD-friendly point estimates with explicit uncertainty (deep ensembles, MC dropout)
- Likelihood-free / simulation-based setting — use ABC or SBI methods
- Strict latency budget at inference time
- Posterior is sharply multi-modal and you can't engineer good chains

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

# Same Bayesian regression in NumPyro — JAX-accelerated NUTS
def model(x, y=None):
    alpha = numpyro.sample("alpha", dist.Normal(0., 10.))
    beta  = numpyro.sample("beta",  dist.Normal(0., 10.))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.))
    mu = alpha + beta * x
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

kernel = NUTS(model, target_accept_prob=0.9)
mcmc   = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(random.PRNGKey(0), x=x_obs, y=y_obs)
mcmc.print_summary()
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

[McElreath — Statistical Rethinking <i class="fas fa-external-link-alt"></i>](https://xcelab.net/rm/statistical-rethinking/){: target="_blank" }
<span class="annotation">The friendliest Bayesian textbook. Builds intuition with concrete examples, every chapter has Stan/PyMC code. Lectures on YouTube too.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyMC learning resources <i class="fas fa-external-link-alt"></i>](https://www.pymc.io/projects/docs/en/stable/learn.html){: target="_blank" }
<span class="annotation">PyMC's curated tutorial collection — A/B testing, GLMs, hierarchical models, time series. Hands-on, well-maintained.</span>

</li>
<li data-tier="indepth" markdown="1">

[Blei et al. (2017) — VI Review <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1601.00670){: target="_blank" }
<span class="annotation">"Variational Inference: A Review for Statisticians" — the standard reference for understanding VI.</span>

</li>
<li data-tier="indepth" markdown="1">

[Brooks et al. — MCMC Handbook <i class="fas fa-external-link-alt"></i>](https://www.mcmchandbook.net/){: target="_blank" }
<span class="annotation">Encyclopedic reference for MCMC algorithms and diagnostics. Free chapters online.</span>

</li>
</ul>

</div>

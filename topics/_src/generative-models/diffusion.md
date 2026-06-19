---
title: Diffusion Models — ML Resources Hub
eyebrow_text: ← Theory · Generative Models
eyebrow_href: {{root}}theory.html
heading: Diffusion Models
lead: The one-shot decoder becomes a long denoising chain — start from noise, refine step by step. The current state of the art, at the cost of slow sampling.
prev_href: gan.html
prev_title: Generative Adversarial Networks
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Stop trying to leap from noise to data in one bound — take a thousand small steps instead.** Every model so far decodes a latent in *one* shot: PPCA with a matrix, the [VAE](vae.html) with one neural pass, the [GAN](gan.html) with one generator pass. Diffusion keeps the same "noise → data" idea but **unrolls the decoder into a long chain**. Take a clean datapoint and slowly drown it in Gaussian noise until nothing is left; then train a network to undo *one* step of that corruption. To generate, start from pure noise and run the denoiser over and over. Each step is a tiny, easy, learnable nudge — and the chain composes into the sharpest samples in the section.

</div>

<article class="tldr-body" markdown="1">

Where the [GAN](gan.html) bet everything on a single generator pass and paid for it with unstable training and dropped modes, diffusion makes a different trade. It keeps the sample *quality* a GAN gets, regains *stability* (no adversary — just a regression) and *mode coverage* (it never stops covering the whole distribution), and pays the bill in one currency: **sampling steps**. One forward pass becomes tens to thousands.

**The forward process — controlled destruction.** Take a real datapoint *x₀* and add a little Gaussian noise. Then a little more. After enough steps the signal is gone and you're left with pure static — a sample from a plain Gaussian. This direction has *no learnable parameters*; it's fixed math you completely control. That's the secret to the whole thing: you only ever learn to invert a process you already understand perfectly.

**The reverse process — what the network learns.** Given a noisy datapoint and its noise level, predict the noise that was mixed in, and subtract a small slice of it to get a slightly cleaner datapoint. Run that loop from pure noise and the chain of small denoising steps walks you from the Gaussian back to the data manifold. The model never has to make the impossible jump in one go — only the easy local correction.

**Why iterative beats one-shot.** A GAN's generator must map a random vector straight onto a photorealistic image; that's a brutal function to learn, which is why GANs are finicky. Diffusion breaks the impossible jump into a thousand questions the model *can* answer — "this is slightly too noisy, clean it up a touch". The objective is plain mean-squared error on the noise: no minimax, no critic, no equilibrium to balance.

**The catch.** That chain is the price. Generating a sample means running the network once *per step* — 10s to 1000s of forward passes versus a GAN's single one. Most of the field's recent work (DDIM, distillation, consistency models, flow matching) is about making the chain shorter without losing quality. For the deep architectural and scheduler detail, see [Diffusion Models, in depth](../neural-networks/diffusion.html).

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for diffusion when

- You want the best sample quality available — it's the current default for images, audio, and video
- Mode coverage matters: you need the model to represent the *whole* distribution, not a few sharp modes
- You want stable, likelihood-style training with no adversary to balance
- Sample diversity and quality matter more than per-sample latency

</div>

<div class="no" markdown="1">

### It breaks down when

- You need strict real-time inference — even distilled diffusion trails a single-pass [GAN](gan.html)
- You need exact likelihoods — diffusion gives a variational bound, not exact log-density
- The data is tiny and you have no pretrained backbone — diffusion needs scale
- Both training *and* inference compute are tightly constrained — both phases are expensive

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# The forward process: corrupt a clean datapoint in ONE shot.
# No learning here — this is the fixed math the model will later invert.
def forward_noise(x0, t, alphas_cumprod):
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)   # signal left at step t
    eps   = torch.randn_like(x0)                   # the noise we add
    x_t   = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * eps
    return x_t, eps                                # the model's job: recover eps
```

</div>

<div class="level-next">
<span>Want the noise-prediction loss, the closed-form forward process, and the sampling loop?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The DDPM objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L} = \mathbb{E}_{x_0,\, t,\, \varepsilon}\,\Big\lVert\, \varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\; t\big)\,\Big\rVert^2, \qquad \varepsilon \sim \mathcal{N}(0, I) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

The whole training signal: predict the noise *ε* that was added at a random step *t*

</li>
<li markdown="1">

The input *√(ᾱ<sub>t</sub>) x₀ + √(1−ᾱ<sub>t</sub>) ε* is the closed-form forward sample *x<sub>t</sub>* — drawable in one shot

</li>
<li markdown="1">

*ᾱ<sub>t</sub>* is the cumulative noise schedule: fraction of signal left at step *t*, falling from ≈1 to ≈0

</li>
<li markdown="1">

Pure regression — no adversary, no partition function, no intractable integral

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} = \text{average squared error between the real noise and the model's guess of that noise} $$</span>

**In words.** Pick a clean datapoint, pick a random `step` *t*, and draw some `noise` *ε*. Blend them with the fixed schedule to make a noisy version `x_t` — that's the forward process in one shot, controlled by *ᾱ<sub>t</sub>* (alpha-bar), which starts near 1 (mostly signal) and ends near 0 (mostly noise). Hand `x_t` and the step *t* to the model and ask it one thing: *which noise did I add?* The loss is just the squared difference between the real `noise` and the model's `guess`. That's it — the cleanest objective in deep learning. Because every training example is "denoise this by a bit", there is no two-player game to balance and nothing to collapse, which is exactly the stability the [GAN](gan.html) lacked.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`step`the random timestep *t*, sampled uniformly — the model trains on every noise level at once

</li>
<li markdown="1">

`noise`the true Gaussian noise *ε* mixed into the clean datapoint

</li>
<li markdown="1">

`x_t`the noisy datapoint *√(ᾱ<sub>t</sub>) x₀ + √(1−ᾱ<sub>t</sub>) ε* — computed in a single line, no chain needed

</li>
<li markdown="1">

`guess`the network's prediction *ε<sub>θ</sub>(x<sub>t</sub>, t)* — recovering the noise *is* recovering the cleaner image

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The forward process has a closed form.** Adding noise step by step is Markov, but because each step is Gaussian the *composition* is too: you can jump straight to step *t* with *q(x<sub>t</sub> | x₀) = 𝒩(√(ᾱ<sub>t</sub>) x₀, (1−ᾱ<sub>t</sub>) I)*. No loop is needed to make a training example — sample *t*, sample *ε*, blend. This is why training is cheap even though sampling is not.

**Predict the noise, not the mean.** The reverse step *p<sub>θ</sub>(x<sub>t−1</sub> | x<sub>t</sub>)* is itself a Gaussian whose mean the network sets. Rather than predict that mean directly, you reparameterise to predict the added noise *ε* and recover the mean algebraically. ε-prediction has uniform variance across timesteps and gives the simple MSE loss above; it's what most implementations use (some use the related "v-parameterisation" for extra stability at the extremes).

**Sampling is the long chain.** Start from *x<sub>T</sub> ~ 𝒩(0, I)* — pure noise — and step down: predict the noise, subtract a slice, optionally add a little fresh noise, repeat to *x₀*. This is the unrolled decoder. The GAN did this in one pass; diffusion does it in 10s–1000s, which is exactly the quality-for-speed trade.

**It's secretly score matching.** Predicting the noise *ε* is equivalent (up to a known scaling) to estimating the *score* — the gradient *∇<sub>x</sub> log p<sub>t</sub>(x)* of the noisy data density. So the noise-prediction network is a score estimator at every noise level, which is the bridge to the continuous-time view in the next tier. See the section [overview](../generative-models.html) for the score-matching objective.

**Conditioning via classifier-free guidance.** To steer generation (text, class), train the same model with the condition sometimes dropped to a "null" token. At sampling time, run it both ways and extrapolate from the unconditional prediction toward the conditional one, scaled by a guidance weight — the `guidance_scale` knob. The [overview](../generative-models.html) covers the mechanics; here it's enough to know one regression model handles both.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# DDPM training loss — the entire objective in five lines.
def diffusion_loss(model, x0, alphas_cumprod):
    B = x0.size(0)
    T = alphas_cumprod.size(0)
    t = torch.randint(0, T, (B,), device=x0.device)   # random step per example
    eps = torch.randn_like(x0)                          # the noise to recover

    # Closed-form forward sample: no chain needed at training time
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t   = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * eps

    # Predict the noise; loss is plain MSE — no adversary, no balancing act
    return F.mse_loss(model(x_t, t), eps)
```

</div>

<div class="level-next">
<span>Want the score connection, latent diffusion, and how the chain gets shortened?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Noise prediction is the score</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \nabla_{x_t} \log p_t(x_t) \;=\; -\,\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}, \qquad dx = \big[\, f(x,t) - g(t)^2\, \nabla_x \log p_t(x)\,\big]\,dt + g(t)\,d\bar W $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

The trained noise-predictor *ε<sub>θ</sub>* **is** the score (up to the schedule factor *√(1−ᾱ<sub>t</sub>)*)

</li>
<li markdown="1">

Sampling = integrating the reverse-time SDE, which needs only the score

</li>
<li markdown="1">

DDPM is one discretisation; the deterministic "probability-flow ODE" shares its marginals

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{score}(x, t) \;=\; -\,\frac{\text{predicted noise}}{\sqrt{1-\bar\alpha_t}}, \qquad \text{sampling} = \text{follow the score downhill from noise to data} $$</span>

**In words.** The `score` is the gradient of log-density — a vector field pointing toward where data is more likely. The headline: the noise your network predicts *is* the score, just rescaled. Predicting noise and estimating the score are the same task in two costumes. Once you have the score at every noise level, sampling is "follow it back to the data": let time flow continuously and integrate the reverse-time stochastic differential equation, which uses a built-in `drift`, the squared `noise-strength` times the `score`, and a controlled random kick (`d̅W`). Start from pure Gaussian noise, integrate backwards, and out comes a sample. DDPM is just one way of chopping that continuous reverse flow into discrete steps.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`score`∇<sub>x</sub> log p<sub>t</sub>(x) — gradient of log-density at noise level *t*, pointing uphill in probability

</li>
<li markdown="1">

`predicted noise`the network output *ε<sub>θ</sub>(x<sub>t</sub>, t)* — the same object you trained with plain MSE

</li>
<li markdown="1">

`drift / noise-strength`*f(x, t)* and *g(t)* — the forward SDE's deterministic pull and noise scale

</li>
<li markdown="1">

`d̅W`reverse-time Wiener increment — formal name for "a small Gaussian random kick" each step

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Latent diffusion closes the arc.** Pixels are wasteful — most of a 512×512 image is perceptual redundancy. So first train a [VAE](vae.html) to compress images into a small latent grid, then run the whole diffusion chain *in that latent space*, decoding once at the end. This is exactly Stable Diffusion (Rombach et al., 2022), and it's why a few-GB checkpoint runs on a consumer GPU. Note what just happened: the VAE we met as a standalone generator returns here as the **compressor**, with diffusion as the decoder over its latent codes — the [VAE](vae.html) page's one-shot decoder and this page's iterative chain working in tandem.

**Shortening the chain — DDIM.** The original DDPM sampler is stochastic and wants ~1000 steps. DDIM (Song et al., 2021) re-derives the *same* trained model as a deterministic process — same noise vector always yields the same sample — and matches 1000-step quality in ~50. ODE solvers (DPM-Solver, UniPC) push it to 10–20.

**Shortening it further — distillation and consistency models.** Distillation trains a student to take bigger jumps than its teacher. Consistency models (Song et al., 2023) go all the way: train a network to map *any* point on a probability-flow ODE trajectory directly to its endpoint, giving 1–4 step generation at quality close to the full chain. This is how modern generators became fast enough to feel interactive.

**The frontier — flow matching / rectified flow.** Instead of defining noise via an SDE and learning the score, flow matching (Lipman et al., 2023) and rectified flow (Liu et al., 2023) learn the *velocity field* of an ODE that transports noise to data along a near-straight path. Training is even simpler than DDPM, sampling needs fewer steps, and Stable Diffusion 3 and Flux already use it — it has largely replaced score matching as the default formulation.

**The honest trade-off, three ways.** On *sample quality*: diffusion ≥ GAN > VAE (the VAE's one-pass decoder blurs; diffusion's chain sharpens). On *sampling speed*: VAE ≈ GAN (one pass) >> diffusion (the chain — unless distilled). On *mode coverage*: diffusion > VAE > GAN (the GAN's adversary drops modes; diffusion's likelihood-style objective covers the whole distribution). On *training stability*: diffusion ≈ VAE > GAN (no minimax to balance). Diffusion's one weakness is the column it created: inference latency.

**The whole arc, one skeleton.** Every model in this section is *sample a latent, decode it*. PPCA decodes with a line; the [VAE](vae.html) with one neural pass; the [GAN](gan.html) with one generator pass; diffusion unrolls the decoder into a long denoising chain. Same idea the [PCA](pca.html) page opened with — *latent code → decoder → data* — stretched from a single matrix multiply to a thousand learnable steps. That's the section.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# DDPM sampling loop — the unrolled decoder: pure noise -> data, step by step.
@torch.no_grad()
def ddpm_sample(model, shape, betas, alphas, alphas_cumprod, device="cuda"):
    x = torch.randn(shape, device=device)                  # start from pure noise
    T = betas.size(0)
    for t in reversed(range(T)):                           # the long chain
        tb    = torch.full((shape[0],), t, device=device, dtype=torch.long)
        a     = alphas[t]
        a_bar = alphas_cumprod[t]

        eps  = model(x, tb)                                # predicted noise (= score)
        mean = (x - betas[t] / (1 - a_bar).sqrt() * eps) / a.sqrt()

        if t > 0:                                          # inject fresh noise except last step
            x = mean + betas[t].sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x                                               # a fresh sample
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

[Diffusion Models, in depth](../neural-networks/diffusion.html)
<span class="annotation">Our own neural-networks page — the architecture, noise schedules, samplers, and ControlNet detail this page links out to, with an interactive forward/reverse viz.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Ho et al. (2020) — Denoising Diffusion Probabilistic Models <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2006.11239){: target="_blank" }

<span class="annotation">The paper that made diffusion competitive. Derives the simple noise-prediction loss and the U-Net recipe everyone copied.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — What are Diffusion Models? <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/){: target="_blank" }

<span class="annotation">The clearest unified treatment of DDPM, the noise schedule, score matching, and the SDE view, with all the algebra worked through.</span>

</li>
<li data-tier="indepth" markdown="1">

[Rombach et al. (2022) — Latent Diffusion (Stable Diffusion) <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2112.10752){: target="_blank" }

<span class="annotation">Diffusion in a VAE's latent space — the paper that closes this section's loop, putting the VAE back to work as the compressor.</span>

</li>
</ul>

</div>

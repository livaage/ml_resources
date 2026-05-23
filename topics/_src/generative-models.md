---
title: Generative Models — ML Resources Hub
eyebrow_text: ← Theory · Frontier
eyebrow_href: ../theory.html
heading: Generative Models
lead: Learn the distribution; sample from it. VAEs, GANs, diffusion, normalising flows, autoregressive — what each one trades for what.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Modelling *P(data)* instead of *P(label | data)*.** Once you've learned the data distribution, you can sample new data, score the likelihood of new data, infill missing parts, or condition on side-information for controlled generation. Different families make different trade-offs between sample quality, likelihood, speed, and ease of training.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Compare the four major families on a simple 2D distribution — same data, very different generative behaviours</span>
</div>
<div class="viz-classic-controls">
<button id="viz-gen-vae" type="button" class="active">VAE</button>
<button id="viz-gen-gan" type="button">GAN</button>
<button id="viz-gen-flow" type="button">Normalising flow</button>
<button id="viz-gen-diff" type="button">Diffusion</button>
<button id="viz-gen-ar" type="button">Autoregressive</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-gen-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-gen-caption"></div>
</div>

<script src="{{root}}js/viz/generative.js"></script>

A simple 2D target distribution (a ring) shown four ways. Each family is good at different things. **VAE** — fast, blurry. **GAN** — sharp samples, mode collapse risk, no likelihood. **Flow** — exact likelihood, restricted architecture. **Diffusion** — high quality, slow sampling. **Autoregressive** — exact likelihood, sequential sampling.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**VAE (Variational Autoencoder).** Encode to a Gaussian latent; decode back. Train with reconstruction + KL-to-prior. Fast sampling (one forward pass); usually blurry; exact likelihood (sort of — an ELBO bound). See [Autoencoders & VAEs](neural-networks/autoencoder.html).

**GAN (Generative Adversarial Network).** Generator and discriminator in a minimax game. Sharp samples; no likelihood; mode collapse common; finicky training. See [GANs](neural-networks/gan.html).

**Normalising flows.** Invertible transformations from a simple base distribution to the data distribution. Exact likelihood and exact sampling; architecture restricted to invertible maps. Useful when you need likelihoods (anomaly detection, density estimation).

**Diffusion.** Iteratively denoise from Gaussian noise. Slow sampling (10s–1000s of steps), highest sample quality, stable training. State of the art for images. See [Diffusion Models](neural-networks/diffusion.html).

**Autoregressive.** Model *P(x) = ∏ P(x<sub>i</sub> | x<sub>&lt;i</sub>)*. Exact likelihood; sequential (slow) sampling; the architecture behind every modern LLM and PixelRNN/CNN.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Pick by what you need

- **Sample quality**: diffusion > GAN ≈ AR > flow > VAE
- **Sample speed**: GAN ≈ VAE ≈ flow > AR > diffusion (parallel: AR loses)
- **Exact likelihood**: flow = AR >> VAE (lower bound only) >> GAN (none)
- **Mode coverage**: AR ≈ diffusion > flow > VAE > GAN
- **Training stability**: AR > diffusion > flow ≈ VAE > GAN

</div>

<div class="no" markdown="1">

### None is a free lunch

- Diffusion: slow at inference unless distilled
- GAN: unstable, mode collapse, no likelihood
- Flow: architecture constraints hurt quality
- VAE: blurry; high-quality VAE needs a lot of capacity
- AR: sequential, slow for long sequences

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Each family in one breath
import torch, torch.nn as nn, torch.nn.functional as F

# VAE — reconstruct + KL
mu, logvar = encoder(x)
z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
rec = decoder(z)
loss = F.mse_loss(rec, x) - 0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum()

# GAN — discriminator tries to tell real from fake
fake = generator(torch.randn(B, z_dim))
d_loss = F.binary_cross_entropy_with_logits(D(real), torch.ones(B)) \
       + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(B))

# Diffusion — predict noise added at random timestep
t = torch.randint(0, T, (B,))
noise = torch.randn_like(x)
xt = sqrt_alpha_bar[t] * x + sqrt_one_minus[t] * noise
loss = F.mse_loss(model(xt, t), noise)

# Autoregressive — next-token cross-entropy
logits = model(x[:, :-1])
loss   = F.cross_entropy(logits.flatten(end_dim=-2), x[:, 1:].flatten())
```

</div>

<div class="level-next">
<span>Want EBMs, score-based models, and conditional generation?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Score matching</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L} = \mathbb{E}_{x \sim p_{\text{data}}}\big\lVert \nabla_x \log p_\theta(x) - \nabla_x \log p_{\text{data}}(x) \big\rVert^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Match the gradient of log-density, not the density itself

</li>
<li markdown="1">

Avoids the partition function

</li>
<li markdown="1">

The foundation of score-based and diffusion models

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; \text{average squared difference between model's score and data's score, over real samples} $$</span>

**In words.** Rather than fitting the data's *density* directly (which would require a hard-to-compute normaliser), fit the *score*: the gradient of the log-density with respect to *x*, written `∇x log p(x)`. The score is a vector field pointing "uphill" in probability — toward where data is more likely. For each real training sample, compute the model's predicted score and compare it to the true data score, then average the squared error. The trick is that normalising constants drop out when you take the log-gradient, so you can train without the partition function. In practice you don't know the true data score either, but Hyvärinen's identity lets you optimise an equivalent objective, and denoising score matching estimates it by adding noise.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`score`∇<sub>x</sub> log p(x) — the gradient of log-density, a vector pointing toward higher-density regions

</li>
<li markdown="1">

`model's score`the score predicted by your neural net (parameters θ)

</li>
<li markdown="1">

`data's score`the true gradient of log p<sub>data</sub>(x) — not known directly, but estimable

</li>
<li markdown="1">

`average over real samples`expectation taken over data *x* drawn from p<sub>data</sub>

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Energy-based models.** Define *p(x) ∝ exp(-E<sub>θ</sub>(x))*. Maximum likelihood requires the partition function (intractable). Trained via contrastive divergence, score matching, or variational methods. Conceptually powerful; tricky in practice.

**Score-based models.** Learn *∇<sub>x</sub> log p(x)* (the "score"). Sample via Langevin dynamics or related SDEs. Closely related to diffusion: a denoising network is essentially a score estimator at multiple noise levels. Unified by Song et al. 2021's SDE formulation.

**Conditional generation.** *p(x | y)* instead of *p(x)*. Class-conditional images, text-to-image, image-to-image. Classifier-free guidance (Ho & Salimans 2021) trains a single model on conditional and unconditional examples; trade off quality and diversity at sampling time by mixing the two.

**Latent diffusion.** Train a VAE to compress images to a latent space; train a diffusion model in that space. Stable Diffusion is exactly this. Faster sampling (smaller latents) without losing much quality.

**Likelihood vs sample quality.** Not the same! A model can have great likelihood and ugly samples (over-smooth average) or beautiful samples and terrible likelihood (mode-collapsed). Pick the metric that matches what you care about.

**Evaluation.** No single metric is right. FID (Fréchet Inception Distance) compares feature statistics; IS (Inception Score) measures diversity + classifiability; PR (Precision/Recall in image space) decouples mode coverage from sample quality; CLIP score for text-image alignment. All have known failure modes.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn

# Classifier-free guidance — train with random dropout of the condition
def sample_cfg(model, y, num_steps, guidance=7.5):
    x = torch.randn(...)
    for t in reversed(range(num_steps)):
        noise_cond   = model(x, t, y)
        noise_uncond = model(x, t, None)
        noise = noise_uncond + guidance * (noise_cond - noise_uncond)
        x = denoise_step(x, noise, t)
    return x
```

</div>

<div class="level-next">
<span>Want SDE formulation, consistency models, & ImageNet scaling?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Score SDE</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ dx = \big[\, f(x, t) - g(t)^2 \nabla_x \log p_t(x)\, \big]\, dt + g(t)\, d\bar W $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Reverse-time SDE for sampling — needs the score *∇log p<sub>t</sub>(x)*

</li>
<li markdown="1">

Unifies diffusion, score-based, and Langevin samplers

</li>
<li markdown="1">

Score networks trained at multiple noise levels approximate ∇log p<sub>t</sub>

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ dx \;=\; \big[\, \text{drift}(x, t) \;-\; \text{diffusion}(t)^2 \times \text{score}(x, t)\, \big]\, dt \;+\; \text{diffusion}(t)\, d\bar W $$</span>

**In words.** This is the reverse-time stochastic differential equation that turns noise back into data. `dx` is the tiny change in *x* at each step of integration. The bracketed part is the deterministic pull: a built-in `drift` term `f(x, t)`, minus the squared `diffusion` coefficient `g(t)` times the `score` — the gradient of log-density at the current noise level, which points uphill toward more-likely *x*. The final term `g(t)·d̅W` injects a controlled amount of random noise at every step (`d̅W` is the reverse Wiener-process increment — formal mathematical jargon for "infinitesimal Gaussian noise"). Start from pure Gaussian noise, integrate this equation backwards in time, and you get a sample from the data distribution. Diffusion, score-based models, and Langevin sampling are all instances of this with different choices of `f` and `g`.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`drift`f(x, t) — the deterministic pull, often a simple linear function of *x*

</li>
<li markdown="1">

`diffusion`g(t) — how much noise to inject at time *t*

</li>
<li markdown="1">

`score`∇<sub>x</sub> log p<sub>t</sub>(x) — gradient of log-density at the current noise level, approximated by a neural net

</li>
<li markdown="1">

`d̅W`reverse-time noise increment — mathematical name for "a small Gaussian random kick"

</li>
<li markdown="1">

`t`time, running from noise (t=T) back to data (t=0) during sampling

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Consistency models.** Song et al. (2023). Train a one-step distillation of a diffusion model — sample in 1 to 4 steps instead of 50–1000. Trade-off: somewhat lower quality than the multi-step teacher; orders-of-magnitude faster.

**Flow matching & rectified flow.** Lipman et al. (2023), Liu et al. (2022). Train a vector field that maps noise to data; sample by solving the ODE. Often faster than diffusion at comparable quality. The frontier of "diffusion done better".

**Discrete diffusion & masked language models.** The diffusion framework generalises beyond Gaussian noise — for discrete data (text, tokens), use absorbing-state diffusion or masked modelling. Mask-and-predict objectives are the conceptual cousin.

**Autoregressive scaling.** Most modern foundation models are autoregressive over discrete tokens, including over images (Parti, MaskGIT, ImageGPT). Tokenise the image with a VQ-VAE or similar, then run an autoregressive transformer over the tokens. Slower at inference than parallel diffusion but conceptually simpler.

**Likelihood-based vs adversarial.** Likelihood-based models (AR, diffusion, flows, VAEs) tend to cover all modes but produce blurry/smoothed samples. Adversarial models (GANs) produce sharp samples but miss modes. Hybrids (VAE-GAN, diffusion-GAN) try to combine.

**Controllable generation.** ControlNet, T2I-Adapter, LoRA, IP-Adapter, image conditioning, depth-conditioning, prompt-engineering — the modern stack for steering diffusion models. Less a "model family" than a meta-pattern of adding more conditioning signals.

**Evaluation difficulties.** FID is the de facto standard but can disagree with human judgement. CLIP-IQA, DINOv2-FID, and human evaluation (preference models) are alternative measures. Always look at samples, not just numbers.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# Consistency-model-style one-step generation
def one_step_sample(consistency_model, noise):
    return consistency_model(noise, t=1.0)   # one forward pass

# Flow matching — train a velocity field
def flow_loss(model, x_data, sigma=1.0):
    t = torch.rand(x_data.size(0))
    x0 = sigma * torch.randn_like(x_data)
    x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x_data
    target_velocity = x_data - x0
    return F.mse_loss(model(x_t, t), target_velocity)
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

[Lilian Weng — Diffusion Models <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/){: target="_blank" }
<span class="annotation">Single best reference for the diffusion / score-based family. Covers DDPM, score-SDE, latent diffusion, and consistency models with worked math.</span>

</li>
<li data-tier="indepth" markdown="1">

[Song et al. (2021) — Score-based Generative Modeling through SDEs <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2011.13456){: target="_blank" }
<span class="annotation">The unifying paper. Shows diffusion = score matching = Langevin sampling under the right lens. Excellent if you want the math.</span>

</li>
<li data-tier="indepth" markdown="1">

[Lipman et al. (2023) — Flow Matching <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2210.02747){: target="_blank" }
<span class="annotation">"Flow matching for generative modelling" — the recent reframing that's becoming the new default.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[HuggingFace diffusers <i class="fas fa-external-link-alt"></i>](https://huggingface.co/docs/diffusers){: target="_blank" }
<span class="annotation">The reference library for diffusion-based generation. Pretrained models, schedulers, samplers, ControlNet — all in one place.</span>

</li>
</ul>

</div>

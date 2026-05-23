---
title: Diffusion Models — ML Resources Hub
eyebrow_text: ← Theory · Neural Networks
eyebrow_href: {{root}}theory.html
heading: Diffusion Models
lead: Iterative denoising — the current default for image, video, and audio generation.
prev_href: gan.html
prev_title: Generative Adversarial Networks
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**Start with pure noise and remove a little bit at a time.** Take a real image, blur it slightly with Gaussian noise, then more, then more, until it's pure static. Now train a network to undo one step of that — given a noisy image, predict the noise that was added. At generation time you start from pure noise and run the denoiser many times. What pops out is a brand-new sample.
</div>

<div class="viz-embed viz-diff" data-fig="diffusion">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            Start with pure noise, repeatedly remove a little bit at every step, and end up with a sample from the data distribution. The model's only job is to predict the noise — the math of inverse diffusion does the rest.
        </span>
    </div>
    <div class="viz-embed-header">
        <span class="viz-embed-title">Drag the slider or hit Forward / Reverse — watch the image walk between clean and pure noise along the cosine schedule</span>
    </div>
    <div class="viz-diff-controls">
        <label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
            Image
            <select id="viz-diff-preset"></select>
        </label>
        <button id="viz-diff-forward" type="button">Forward ▶</button>
        <button id="viz-diff-reverse" type="button">◀ Reverse</button>
        <button id="viz-diff-reset"   type="button">Reset</button>
        <input id="viz-diff-slider" class="viz-diff-slider" type="range" aria-label="diffusion timestep">
        <span class="viz-diff-t" id="viz-diff-t">t = 0</span>
    </div>
    <div class="viz-diff-canvas-wrap">
        <canvas id="viz-diff-canvas"></canvas>
    </div>
    <div class="viz-diff-caption" id="viz-diff-caption"></div>
</div>
<script src="{{root}}js/viz/diffusion.js"></script>

<article class="tldr-body" markdown="1">

### Why iterative denoising instead of one-shot generation

Generating a photorealistic image directly from a random vector is a brutally hard learning problem — that's what GANs try to do, and it's why they're famously unstable to train (mode collapse, vanishing discriminators, fragile balance between two networks). Diffusion sidesteps the whole mess by breaking the impossible jump into a thousand tiny ones. Each step asks the model a question it can actually answer: "you're given an image that's *slightly* noisier than the truth — clean it up just a little". The training objective is plain MSE on noise; no adversarial game, no two-player equilibrium, no mode collapse.

### Why this works at all

You only ever train on the inverse of a process you completely control. The forward process — adding Gaussian noise — has no learnable parameters; you can compute *x<sub>t</sub>* from a clean *x<sub>0</sub>* in one shot using a closed-form formula. The model only has to learn one thing: "given this noisy image and this noise level, what noise was added?" That single learnable mapping is enough to generate brand-new samples from pure noise at inference time, because the chain of small denoising steps composes into a path from the noise distribution back to the data distribution.

### Forward vs reverse process

The **forward process** corrupts data: at each step it mixes a fraction of fresh Gaussian noise into the previous image. After enough steps the original signal is gone and you have pure noise. This direction is fixed math — no neural net involved.

The **reverse process** is what the network learns: at each step it takes a noisy image, predicts the noise component, and subtracts a small amount of it to produce a slightly cleaner image. Run that loop many times starting from pure noise and you end up at a new, plausible sample from the training distribution.

### Classifier-free guidance — how text conditioning works

The model is trained two ways: sometimes given a text prompt, sometimes given a "null" prompt (with some random probability, e.g. 10% of the time). At inference, you run *both* — once conditioned on the prompt, once unconditioned — and extrapolate away from the unconditional prediction toward the conditional one. A guidance scale *s* dials how strongly the prompt pulls the result. This is why Stable Diffusion has a `guidance_scale` knob: low values give diverse, vaguely-related images; high values nail the prompt at the cost of variety.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- High-quality image, video, or audio generation — diffusion is the current default
- Text-conditioned generation with controllable prompt fidelity via guidance scale
- You need spatial control (depth maps, edges, poses) — pair with ControlNet
- Sample diversity matters more than per-sample latency

</div>
<div class="no" markdown="1">

### Skip it when

- Strict real-time inference — even distilled diffusion is slower than a single-step GAN
- You need exact likelihoods — diffusion gives bounds, not exact log-density
- Small dataset with no pretrained backbone — diffusion needs scale to generalise
- Compute-constrained training *and* inference — both phases are expensive

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("A serene mountain lake at sunset", num_inference_steps=30).images[0]
image.save("output.png")
```

</div>

<div class="level-next">
    <span>Want the forward / reverse math and the noise schedule?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Forward and reverse processes</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ q(\mathbf{x}_t \mid \mathbf{x}_0) \;=\; \mathcal{N}\!\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\; (1 - \bar\alpha_t)\,\mathbf{I}\big), \qquad p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \;=\; \mathcal{N}(\boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_\theta) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>q(x<sub>t</sub>|x<sub>0</sub>)</code>forward process — closed-form Gaussian, no learning needed</li>
<li markdown="1"><code>p<sub>θ</sub>(x<sub>t-1</sub>|x<sub>t</sub>)</code>reverse process — Gaussian whose mean is parameterised by the network</li>
<li markdown="1"><code>ᾱ<sub>t</sub></code>cumulative noise schedule — fraction of signal left at step *t*</li>
<li markdown="1"><code>𝒩(μ, Σ)</code>Gaussian with mean *μ* and covariance *Σ*; *I* is the identity (noise is independent across pixels)</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{forward: } x_t \;=\; \sqrt{\bar\alpha_t}\,\cdot\,\text{clean image} \;+\; \sqrt{1 - \bar\alpha_t}\,\cdot\,\text{Gaussian noise} \qquad \text{reverse: model predicts cleaner version from noisier one} $$</span>

**In words.** The **forward process** has no learnable parts: given a clean image *x<sub>0</sub>*, you produce its noisy version *x<sub>t</sub>* at any timestep *t* in one shot, by blending the clean image with fresh Gaussian noise. The blend is controlled by *ᾱ<sub>t</sub>* (alpha-bar) — a fixed schedule that starts near 1 (mostly clean) at *t=0* and ends near 0 (mostly noise) at the final step. The **reverse process** is what the model learns: given the noisy *x<sub>t</sub>*, predict a slightly less noisy *x<sub>t-1</sub>*.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Noise schedule.** *ᾱ<sub>t</sub>* controls how fast signal decays into noise. **Linear** schedules (original DDPM) destroy signal too fast at the end — the last 20% of timesteps add almost no information. **Cosine** schedules (Nichol & Dhariwal, 2021) keep more signal late and give noticeably better samples. Modern diffusers usually default to cosine or a learned schedule.

**The reparameterisation trick.** Instead of having the network predict the mean of *p<sub>θ</sub>(x<sub>t-1</sub>|x<sub>t</sub>)* directly, you reparameterise: predict the noise *ε* that was added, and recover the mean algebraically. The training loss collapses to plain MSE between predicted noise and true noise — the cleanest objective in deep learning.

**Predict noise vs predict x₀.** Mathematically equivalent (one is an algebraic rearrangement of the other), but ε-prediction has uniform variance across timesteps while x₀-prediction's variance explodes near *t = T*. Most implementations predict ε; some predict the "v-parameterisation" of Salimans & Ho, which interpolates between the two and is more stable at the extremes.

**DDPM vs DDIM samplers.** DDPM (the original) is stochastic — adds fresh noise at every reverse step — and needs ~1000 steps. DDIM (Song et al., 2021) re-derives the same model as a deterministic process: the same noise vector always gives the same image, and 50 steps match 1000-step DDPM quality. DPM-Solver, UniPC, and other ODE solvers push this to 10–20 steps.

**Latent diffusion.** Pixels are wasteful: most of a 512×512 image is perceptual redundancy that a small autoencoder can compress away. Train a VAE to encode images into a 4× smaller latent grid, then run diffusion in that latent space. Stable Diffusion is exactly this — and it's the reason a 4 GB checkpoint can run on a consumer GPU.

**U-Net vs DiT backbone.** Early diffusion used a U-Net (skip-connections between matched encoder/decoder resolutions — well-suited to noise prediction at every scale). Modern systems (Stable Diffusion 3, Sora) use **Diffusion Transformers (DiT)**: a ViT over latent patches, conditioned on timestep and prompt via AdaLN. Transformers scale more cleanly with compute and data.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- State-of-the-art quality on a generation task and you control the schedule and sampler
- Conditional generation (text, class, image-to-image, inpainting)
- You want stable training without an adversarial loss
- You can afford 10–50 inference steps with a fast solver

</div>
<div class="no" markdown="1">

### Skip it when

- Strict latency budget — distilled or single-step models exist but cost quality
- Likelihoods are required exactly — diffusion only gives a variational bound
- Small dataset with no pretrained backbone — fine-tune an existing model instead
- You're doing density estimation rather than sampling

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# DDPM training loop, distilled to its essence
def diffusion_loss(model, x_0, alphas_cumprod):
    B = x_0.size(0)
    T = alphas_cumprod.size(0)
    t = torch.randint(0, T, (B,), device=x_0.device)
    noise = torch.randn_like(x_0)

    # Closed-form noisy version: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t   = a_bar.sqrt() * x_0 + (1 - a_bar).sqrt() * noise

    # Model predicts the noise; loss is just MSE
    pred = model(x_t, t)
    return F.mse_loss(pred, noise)
```

</div>

<div class="level-next">
    <span>Want the SDE view, flow matching, and the modern frontier?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Score-based view (SDE)</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \mathrm{d}\mathbf{x} \;=\; \boldsymbol{f}(\mathbf{x}, t)\,\mathrm{d}t + g(t)\,\mathrm{d}\mathbf{w}, \qquad \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \approx \mathbf{s}_\theta(\mathbf{x}, t) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>f, g</code>drift and diffusion coefficients of the forward noising SDE</li>
<li markdown="1"><code>s<sub>θ</sub></code>learned *score function* — gradient of log-density at time *t*</li>
<li markdown="1">Sampling = solve the reverse-time SDE (or its deterministic ODE counterpart), which only needs the score</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{tiny change in } x \;=\; \text{drift}(x, t)\,\text{dt} \;+\; \text{noise-strength}(t)\,\text{dW}, \qquad \text{model } s_\theta \;\approx\; \text{slope of log-density at time }t $$</span>

**In words.** Step back from discrete timesteps and let time flow continuously. The forward noising can be written as a **stochastic differential equation** (SDE): in every infinitesimal slice *dt*, *x* drifts a little (the **drift** term) *and* gets bumped by random noise scaled by some **noise-strength** (the *dW* term — Brownian motion). What the model actually learns is the **score**: the gradient of *log p<sub>t</sub>(x)*, i.e. "in which direction does the data density grow most steeply at this noise level". Knowing the score is equivalent to knowing how to denoise. DDPM is one discretisation of this continuous picture.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Score matching connection (Song et al., 2021).** Denoising score matching shows that learning to predict noise is mathematically equivalent to learning the score *∇<sub>x</sub> log p<sub>t</sub>(x)* of the noisy data distribution. Once you have the score at every noise level, the reverse-time SDE turns it into a sampler. This unifies DDPM, NCSN, and continuous-time diffusion under a single framework — and gives you an exact deterministic ODE ("probability-flow ODE") whose trajectories yield identical marginals to the SDE.

**Flow matching — the modern reformulation.** Rather than defining noise via an SDE and learning the score, **flow matching** (Lipman et al., 2023) and **rectified flow** (Liu et al., 2023) directly learn the velocity field of an ODE that transports a noise sample to a data sample along a straight (or near-straight) path. Training is even simpler than DDPM, sampling needs fewer steps, and the math generalises beyond Gaussian endpoints. Stable Diffusion 3 and Flux use rectified flow; it has effectively replaced score matching as the default formulation.

**Consistency models for fast sampling.** Train a network *f<sub>θ</sub>(x<sub>t</sub>, t)* that maps any point on a probability-flow ODE trajectory to its endpoint *x<sub>0</sub>* in one step. Once trained you get single-step generation with quality close to multi-step diffusion. Consistency models (Song et al., 2023), Latent Consistency Models, and Adversarial Diffusion Distillation (SDXL Turbo) all sit in this family — they're how modern image generators got fast enough to feel interactive.

**ControlNet and spatial conditioning.** A ControlNet (Zhang et al., 2023) clones the diffusion U-Net's encoder, freezes the original weights, and trains the clone to inject a conditioning signal (depth map, edge map, OpenPose skeleton, segmentation mask) into the frozen backbone via zero-initialised connections. Result: pixel-accurate spatial control over composition without retraining the base model. This is how production pipelines combine prompt control with layout control.

**The current frontier.** Three threads are reshaping the field. **Rectified flow** is replacing score matching for the cleanest training and sampling math. **Hybrid diffusion / autoregressive** systems (e.g. MAR, Transfusion) wrap a diffusion head inside an AR backbone, getting AR's flexibility on tokens with diffusion's quality on continuous outputs. And **video diffusion** at scale (Sora, Veo) treats time as just another axis in a 3D DiT, with the same training recipe extended to spatiotemporal latents.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- **DiT in latent space**: scaling text-to-image / video
- **Rectified flow**: training a new generator from scratch with the simplest recipe
- **Consistency model / LCM**: latency matters more than the final 5% of quality
- **ControlNet**: you need pixel-level spatial control with a frozen base model

</div>
<div class="no" markdown="1">

### Skip it when

- Hard real-time inference — single-step GANs and amortised samplers still win
- Exact likelihoods required — use a normalising flow or autoregressive model
- You can't afford to run the U-Net / DiT at inference at all
- Tiny dataset with no related pretrained model — fine-tuning needs a starting point

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch

# DDIM sampling — deterministic, faster than DDPM
@torch.no_grad()
def ddim_sample(model, shape, alphas_cumprod, n_steps=50, device="cuda"):
    T  = alphas_cumprod.size(0)
    timesteps = torch.linspace(T - 1, 0, n_steps + 1).long().to(device)
    x = torch.randn(shape, device=device)

    for i in range(n_steps):
        t      = timesteps[i].expand(shape[0])
        t_next = timesteps[i + 1].expand(shape[0])
        a_t    = alphas_cumprod[t].view(-1, 1, 1, 1)
        a_next = alphas_cumprod[t_next].view(-1, 1, 1, 1)

        # Predict the noise, derive x_0, then advance to next timestep
        noise = model(x, t)
        x0    = (x - (1 - a_t).sqrt() * noise) / a_t.sqrt()
        x     = a_next.sqrt() * x0 + (1 - a_next).sqrt() * noise

    return x
```

</div>

<div class="level-next">
    <span>Want the picture instead?</span>
    <button data-go-to="intuition" type="button">← Back to Intuition</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="diffusion" markdown="1">

### Walking between clean and noise

The **left panel** is the clean image *x<sub>0</sub>*. The **middle panel** is what the model sees at timestep *t* — the same image with cumulative noise mixed in along the cosine schedule. The **right panel** is what a "perfect denoiser" would predict for *x<sub>0</sub>* given the middle panel.

Drag the slider or hit **Forward** / **Reverse** to walk *t* between 0 (clean) and *T* (pure noise). The equation under the canvas — *x<sub>t</sub> = √ᾱ · x<sub>0</sub> + √(1 − ᾱ) · ε* — is the closed-form forward formula. Training a diffusion model is just teaching a network to invert it: predict *ε* from *x<sub>t</sub>* and the timestep. Everything else — DDPM, DDIM, flow matching, ControlNet — falls out of that one prediction.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The clearest unified treatment of DDPM, score matching, and the SDE view. Start here.</span>
</li>
<li data-tier="intuition" markdown="1">
[Yang Song — Generative Modeling by Estimating Gradients](https://yang-song.net/blog/2021/score/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The unified score-matching / SDE picture written by one of the field's main authors. The right pivot from DDPM to modern theory.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Ho et al. (2020) — DDPM](https://arxiv.org/abs/2006.11239) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The paper that made diffusion competitive. Establishes the noise-prediction objective and the U-Net backbone everyone copied.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Tony Duan — Annotated DDPM](https://github.com/tonyduan/diffusion) <i class="fas fa-external-link-alt"></i>
<span class="annotation">A clean, line-by-line PyTorch implementation of DDPM with derivations alongside the code. The best way to internalise the training loop.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Hugging Face Diffusers](https://huggingface.co/docs/diffusers) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The library most modern diffusion work runs on. Model zoo, schedulers, training scripts, and tutorials covering DDPM, DDIM, latent diffusion, ControlNet.</span>
</li>
<li data-tier="indepth" markdown="1">
[Rombach et al. (2022) — Latent Diffusion](https://arxiv.org/abs/2112.10752) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Stable Diffusion's paper. Shows how decoupling perceptual compression and diffusion makes high-resolution generation tractable on a single GPU.</span>
</li>
</ul>

</div>

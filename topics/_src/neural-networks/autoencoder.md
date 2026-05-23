---
title: Autoencoders & VAEs — ML Resources Hub
eyebrow_text: ← Theory · Neural Networks
eyebrow_href: {{root}}theory.html
heading: Autoencoders & VAEs
lead: Encoder-decoder networks for representation learning, dimensionality reduction, and generation.
prev_href: gnn.html
prev_title: Graph Neural Networks
next_href: gan.html
next_title: Generative Adversarial Networks
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**Squeeze data through a narrow channel, then rebuild it.** An *encoder* compresses the input to a small *code*; a *decoder* tries to reconstruct the original from that code alone. The bottleneck has nowhere to hide redundancy — only the features that actually help reconstruction survive.
</div>

<div class="viz-embed viz-ae" data-fig="autoencoder">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            Compress the <strong>12×12 image on the left</strong> through a narrow bottleneck of just <strong>two numbers</strong> (the orange dot's position in the middle), then reconstruct the image on the right. Click a preset to encode it; drag the orange dot to roam the latent space; toggle <strong>VAE mode</strong> for stochastic codes. The reconstruction error tells you what the code couldn't keep.
        </span>
    </div>
    <div class="viz-embed-header">
        <span class="viz-embed-title">Encoder · 2-D latent · decoder — drag the dot to decode any code</span>
    </div>
    <div class="viz-ae-controls">
        <span style="font-size: 0.78rem; color: var(--muted); margin-right: 0.2rem;">Input:</span>
        <span class="viz-ae-presets" id="viz-ae-presets"></span>
        <button id="viz-ae-vae"   type="button">VAE mode: off</button>
        <button id="viz-ae-reset" type="button">Reset</button>
    </div>
    <div class="viz-ae-canvas-wrap">
        <canvas id="viz-ae-canvas"></canvas>
    </div>
    <div class="viz-ae-caption" id="viz-ae-caption"></div>
</div>
<script src="{{root}}js/viz/ae.js"></script>

<article class="tldr-body" markdown="1">

### Why the bottleneck matters

If the code were as wide as the input, the network could just copy values through and learn nothing — the trivial identity function. The bottleneck makes that impossible: only a fixed number of dimensions get to pass, so the encoder has to *choose* what's worth keeping. The decoder, working from those few numbers alone, becomes a reconstruction prior — it fills in what it has learned tends to be there. The two halves trained jointly converge on a compact summary of the data's structure.

### What the latent space encodes

The 2-D latent in the figure is doing a job a lot like PCA, but non-linear. Each preset lands at its own spot in the plane; points *between* presets decode to blends. Whatever varies most across the training set ends up as a direction in latent space. Walk along that direction and the reconstruction smoothly morphs — without you ever telling the network what "morph" means.

### Why this is unsupervised

There are no labels. The target *is* the input — every example supervises its own reconstruction. That's the appeal: any pile of unlabelled data is enough to learn an embedding you can then plug into a classifier, a clustering algorithm, or an anomaly detector.

### Denoising autoencoders

A small twist with outsized benefits: corrupt the input (add noise, mask pixels) and train the decoder to recover the *clean* original. The model can no longer memorise inputs because the inputs change every step — it has to learn what *should* be there. This is the direct ancestor of masked-token pretraining in language models.

### Variational autoencoders — the generative version

A plain AE's latent space has holes — sample a random code and the decoder usually outputs garbage, because nothing trained it on that point. A **VAE** fixes this by making the encoder output a *distribution* over codes (a mean and a variance) and pushing those distributions toward a fixed prior, usually a unit Gaussian. After training, you can sample any code from that prior, run it through the decoder, and get a plausible new sample. That's the "generative" in VAE.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Learning representations from a pile of unlabelled data
- Non-linear dimensionality reduction (a deeper cousin of [PCA]({{root}}topics/dimensionality-reduction/pca.html))
- Anomaly detection — large reconstruction error flags an outlier
- Denoising or inpainting where the corruption pattern is known
- You need a smooth, navigable latent space and can accept softer samples (VAE)

</div>
<div class="no" markdown="1">

### Skip it when

- You have labels — supervised pretraining usually gives sharper embeddings
- Highest-quality image / audio generation is the goal — reach for [diffusion]({{root}}topics/neural-networks/diffusion.html) or a [GAN]({{root}}topics/neural-networks/gan.html)
- Linear methods (PCA) are already good enough
- You need calibrated likelihoods — the ELBO is a lower bound, not the real thing

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, d_in=784, d_latent=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, d_latent))
        self.dec = nn.Sequential(nn.Linear(d_latent, 256), nn.ReLU(), nn.Linear(256, d_in))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

# Loss: just reconstruction error
loss = ((model(x)[0] - x) ** 2).mean()
```

</div>

<div class="level-next">
    <span>Want the loss choices, regularised variants, and the VAE objective?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">VAE objective (ELBO)</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \mathcal{L}(\theta, \phi; \mathbf{x}) \;=\; \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]}_{\text{reconstruction}} \;-\; \underbrace{D_\text{KL}\!\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p(\mathbf{z})\big)}_{\text{regulariser}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>q<sub>φ</sub>(z|x)</code>encoder — outputs a distribution over codes, not a single point</li>
<li markdown="1"><code>p<sub>θ</sub>(x|z)</code>decoder — likelihood of reconstructing *x* from *z*</li>
<li markdown="1"><code>p(z)</code>prior, typically *𝒩(0, I)*</li>
<li markdown="1">Maximise this lower bound on *log p(x)*</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{ELBO} \;=\; \underbrace{\text{avg over codes drawn from encoder}[\log p(\text{reconstruction} \mid \text{code})]}_{\text{reconstruction term}} \;-\; \underbrace{\text{distance(encoder's code distribution, prior)}}_{\text{regulariser term}} $$</span>

**In words.** The objective has two competing parts. The first — **reconstruction** — rewards the model for decoding back to something like the input: sample a `code` from the encoder's distribution given `x`, run it through the decoder, and measure how well it reconstructs `x`. The second — the `D`<sub>`KL`</sub> term — penalises the encoder when its per-sample distribution drifts away from the chosen **prior** (a fixed unit Gaussian). That keeps the latent space tidy and Gaussian-shaped so you can sample from it later. `D`<sub>`KL`</sub> (KL divergence) is just a measure of how far one distribution sits from another. Maximising this whole expression (the **ELBO** — Evidence Lower BOund) is provably a lower bound on the true data log-likelihood; you'll see that derived in the In-depth tier.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Reconstruction loss — pick by data type.** Use **MSE** (squared error) for continuous outputs like image pixels in `[0,1]` or audio amplitudes; use **binary cross-entropy** per pixel when the decoder produces Bernoulli probabilities (the original VAE paper's choice for MNIST). For discrete data — text tokens, VQ codes — use ordinary cross-entropy. The loss implicitly encodes a *noise model* on the output: MSE assumes Gaussian noise around the reconstruction, BCE assumes Bernoulli. Mismatched losses produce blurry or weirdly sharp samples.

**Undercomplete vs. overcomplete.** *Undercomplete* means the latent is smaller than the input — the bottleneck does the regularising for you. *Overcomplete* (latent bigger than input) needs help to avoid copying the input straight through: sparsity penalties, denoising corruption, or contractive penalties on the encoder Jacobian. Without one of those, an overcomplete AE happily learns the identity and tells you nothing.

**Tied weights.** A classical trick: force the decoder weights to be the transpose of the encoder weights (`W_dec = W_enc^T`). Halves the parameter count and gives a clean analogy to PCA — for a linear AE with MSE loss and tied weights, the optimal solution literally spans the top principal components. Modern deep AEs usually don't bother, but it's a clean inductive bias for small models.

**Regularised variants.**

- **Sparse AE** — add an L1 penalty (or a KL penalty on activation means) so most latent units stay near zero per example. Each input then activates a different small subset, encouraging interpretable units.
- **Contractive AE** — penalise the Frobenius norm of the encoder Jacobian, *‖∂z/∂x‖²*. Makes the code locally insensitive to tiny input perturbations — it stays put on the data manifold.
- **Denoising AE** — corrupt the input, reconstruct the clean version. Practically the most useful — no extra term in the loss, just data augmentation on the input side.

**VAE essentials.** The encoder outputs *μ(x)* and *log σ²(x)*. To sample *z*, the **reparameterisation trick** rewrites *z = μ + σ · ε* with *ε ~ 𝒩(0, I)*, keeping gradients flowing through *μ* and *σ*. The KL term against a unit Gaussian has a closed form: `−½ Σ (1 + log σ² − μ² − σ²)`. Train, then sample new outputs by drawing *z ~ 𝒩(0, I)* and decoding.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- You need a continuous, navigable latent space (interpolation, vector arithmetic)
- Anomaly detection via reconstruction error or likelihood
- Representation learning where you can't get labels
- You want a fast single-pass generator and can accept some blur

</div>
<div class="no" markdown="1">

### Skip it when

- Highest possible sample quality matters — diffusion / GANs win
- You only need embeddings — a contrastive method (SimCLR, CLIP) is usually sharper
- Sharp generation (no blur) is non-negotiable
- You need exact likelihoods — reach for a normalising flow

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d_in=784, d_latent=32):
        super().__init__()
        self.enc       = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU())
        self.fc_mu     = nn.Linear(256, d_latent)
        self.fc_logvar = nn.Linear(256, d_latent)
        self.dec       = nn.Sequential(nn.Linear(d_latent, 256), nn.ReLU(),
                                       nn.Linear(256, d_in))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h      = self.enc(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    rec = F.mse_loss(x_hat, x, reduction="sum")
    kl  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    return rec + kl
```

</div>

<div class="level-next">
    <span>Want ELBO derivation, β-VAE, VQ-VAE, and the modern story?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">ELBO derivation</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \log p(\mathbf{x}) \;=\; \mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\!\big[\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}\mid\mathbf{x})\big] \;+\; D_\text{KL}\!\big(q(\mathbf{z}\mid\mathbf{x}) \,\|\, p(\mathbf{z}\mid\mathbf{x})\big) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">Log-marginal = ELBO + KL to the true posterior</li>
<li markdown="1">Since KL ≥ 0, ELBO is a lower bound on log *p(x)*</li>
<li markdown="1">Maximising ELBO simultaneously fits the data and makes *q* close to the posterior</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \log p(\text{data}) \;=\; \underbrace{\text{ELBO}}_{\text{what we maximise}} \;+\; \underbrace{D_\text{KL}(\text{encoder} \,\|\, \text{true posterior})}_{\,\ge\,0\,\text{ — the gap we can't see}} $$</span>

**In words.** This is the algebraic identity that makes the VAE training objective principled. The **log-marginal** `log p(data)` — how well the model explains the data — splits exactly into the **ELBO** plus the KL distance from the encoder's distribution to the *true* posterior (the distribution over codes you'd want, given the data). Because `D`<sub>`KL`</sub> is always non-negative, the ELBO is provably a *lower bound* on `log p(data)` — that's where the name "Evidence Lower BOund" comes from. Pushing the ELBO up does two good things at once: it makes the model fit the data better, and it makes the encoder's approximation closer to the true posterior. The true posterior is intractable, so we never compute that gap directly — the ELBO is what we actually optimise.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**β-VAE — knob for disentanglement.** Re-weight the KL term: *ℒ = E[log p(x|z)] − β · KL(q‖p)*. With *β > 1*, the prior pressure goes up and each latent dimension is pushed to carry information independently — encouraging "disentangled" axes that correspond to single factors of variation (pose, lighting, identity). With *β < 1*, you get sharper samples but a messier latent. Locatello et al. (2019) proved purely unsupervised disentanglement is impossible without inductive biases — β-VAE is one such bias, not a guarantee.

**Posterior collapse.** When the decoder is powerful enough to model the data on its own (think autoregressive PixelCNN decoder), it can ignore *z* entirely — the encoder maps everything to the prior, the KL term goes to zero, and the latent carries no information. Fixes: **KL annealing** (start at zero, ramp up), **free bits** (don't penalise KL below a per-dimension floor), or simply weaken the decoder.

**VQ-VAE — discrete latents.** Replace the Gaussian latent with the nearest entry in a learned **codebook** of vectors. The encoder outputs a continuous vector, you snap it to the closest codebook entry, and the decoder reconstructs from that. The discrete bottleneck eliminates posterior collapse and produces *tokens* — which is why VQ-VAE is the tokeniser behind DALL-E v1, Parti, and most multi-modal LLMs that handle images or audio.

**Latent diffusion — the modern role.** Rombach et al. (2022, "Stable Diffusion") split generation into two stages: first train a VAE to compress images to a small latent grid, then train a diffusion model in *that* space rather than pixel space. The VAE handles the boring perceptual compression; the diffusion model handles the interesting semantic generation. This is by far the biggest production use of autoencoders today — almost every modern image / video model uses a learned VAE encoder under the hood.

**Where this leaves plain AEs.** For high-fidelity *generation*, diffusion has clearly won. But autoencoders are alive and well: as the *compression frontends* for latent diffusion, as **tokenisers** for multi-modal models (VQ-VAE), for **anomaly detection** in industrial monitoring, for **non-linear dimensionality reduction** that beats PCA on complex manifolds, and as a clean teaching example of representation learning.

**Compared to self-supervised pretraining.** Modern self-supervised methods (SimCLR, DINO, MAE, CLIP) often beat plain AEs at producing useful embeddings for downstream tasks — they optimise for *invariance* and *separability* rather than pixel-level reconstruction. The Masked Autoencoder (He et al., 2022) is a notable bridge: it's structurally an AE with heavy masking, but trained at ViT scale it produces excellent transfer features. The line between "autoencoder" and "self-supervised method" is blurry once you start masking.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- **Latent diffusion** — almost every modern image / video generator needs a VAE frontend
- **VQ-VAE** — tokeniser for multi-modal models, discrete codes for downstream LMs
- **β-VAE** — studying disentangled representations in a controlled setting
- **Anomaly detection** at scale — reconstruction error is cheap and unsupervised
- **Masked AE pretraining** — strong transfer features when you have a ViT and a lot of data

</div>
<div class="no" markdown="1">

### Skip it when

- Sample sharpness is the only thing you care about — use diffusion directly in pixel space
- You just need good embeddings — contrastive (CLIP, SimCLR, DINO) usually wins
- You need exact likelihoods — use a normalising flow instead
- Your data is small and tabular — PCA or a [denoising AE](https://d2l.ai/chapter_computer-vision/image-augmentation.html) tiny model is plenty

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
import torch.nn.functional as F

# β-VAE with KL annealing and free bits
class BetaVAE(VAE):
    pass    # same module; the loss is what changes

def beta_vae_loss(x_hat, x, mu, logvar, beta=4.0, free_bits=0.05):
    rec = F.mse_loss(x_hat, x, reduction="sum")
    # KL per latent dim, with a "free bits" floor (don't penalise below it)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = (kl_per_dim - free_bits).clamp(min=0).sum()
    return rec + beta * kl_per_dim

# Train with KL annealing: ramp beta from 0 to target over warmup steps
def beta_schedule(step, warmup=5000, target=4.0):
    return min(1.0, step / warmup) * target
```

</div>

<div class="level-next">
    <span>Want the picture instead?</span>
    <button data-go-to="intuition" type="button">← Back to Intuition</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="autoencoder" markdown="1">

### Inside the figure

The **left grid** is the 12×12 input. The **orange dot** in the centre panel is the latent code — *two numbers* that together summarise the entire image. The **right grid** is what the decoder reconstructs from those two numbers alone.

Click a preset and the encoder lands the dot at a learned position. Drag the dot anywhere and the decoder will try to reconstruct *something* — points near a preset look like that preset, points between presets give blends, and points far from any preset decode to garbage.

Toggle **VAE mode** and the encoder starts outputting a small cloud instead of a single point: each forward pass samples a slightly different code. That stochasticity is exactly what lets the trained VAE generate plausible new samples — the decoder has learned to handle a whole neighbourhood, not just one point.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[Zhang et al. — Dive into Deep Learning (autoencoders)](https://d2l.ai/chapter_generative-adversarial-networks/index.html) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Interactive, code-first textbook. The generative-models chapters walk through plain AEs, denoising AEs, and the family with runnable PyTorch.</span>
</li>
<li data-tier="intuition" markdown="1">
[Lilian Weng — From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">A single tour through AE, denoising AE, sparse AE, contractive AE, VAE, and β-VAE. The clearest single source on the whole family.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Kingma & Welling — An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The original VAE authors' modern tutorial. The right place to go for the ELBO derivation, reparameterisation trick, and amortised inference framing.</span>
</li>
<li data-tier="indepth" markdown="1">
[van den Oord et al. (2017) — VQ-VAE](https://arxiv.org/abs/1711.00937) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Discrete-codebook latents — the backbone of modern multi-modal tokenisers (DALL-E v1, Parti, audio LMs).</span>
</li>
<li data-tier="indepth" markdown="1">
[Rombach et al. (2022) — High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Stable Diffusion's paper. Shows how a VAE frontend lets diffusion scale to high-resolution generation cheaply — the dominant production use of autoencoders today.</span>
</li>
</ul>

</div>
</content>
</invoke>
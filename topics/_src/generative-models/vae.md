---
title: Variational Autoencoders (VAE) — ML Resources Hub
eyebrow_text: ← Theory · Generative Models
eyebrow_href: {{root}}theory.html
heading: Variational Autoencoders
lead: PPCA with the linear decoder swapped for a neural network — and an encoder that learns to approximate the now-intractable posterior.
prev_href: pca.html
prev_title: PCA & Probabilistic PCA
next_href: gan.html
next_title: Generative Adversarial Networks
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Take the PPCA recipe and bend the line.** [PCA & Probabilistic PCA](pca.html) generates data by drawing a code *z* from a unit Gaussian and pushing it through a *straight line* *Wz + μ*, plus noise. A VAE keeps that exact skeleton — *sample a code, decode it* — but replaces the straight line with a **neural network** so the data manifold can curve. The price: the tidy closed-form posterior PPCA enjoyed is now intractable, so a second network — the **encoder** — learns to approximate it. Train both by maximising the ELBO, a lower bound on the same *log p(x)* PPCA computes exactly.

</div>

<article class="tldr-body" markdown="1">

PPCA was the "hydrogen atom": linear, Gaussian, solvable by hand. But a straight line can only ever produce one ellipsoidal blob. Faces, digits, audio live on *curved* manifolds. The VAE is the smallest possible upgrade that lets the manifold bend — and it earns its name from how it copes with the fallout.

**The three pieces.** An **encoder** *q(z\|x)* reads a datapoint and outputs a *distribution* over codes — a mean and a spread — rather than a single point. A **latent code** *z* is sampled from that distribution. A **decoder** *p(x\|z)* — now a neural net, not a matrix — maps the code back up to data space. Encoder and decoder are trained jointly so that codes round-trip back to their inputs.

**Why a distribution, not a point?** This is the whole trick. A plain [autoencoder](../neural-networks/autoencoder.html) squeezes each input to a single point, and the gaps between those points decode to garbage — there's no way to *sample*. By forcing the encoder to emit a fuzzy cloud and pulling every cloud toward one shared unit-Gaussian prior, the VAE fills the latent space in. Afterwards you can draw any *z ~ N(0, I)*, decode it, and get something plausible. That's exactly the PPCA generative story, now with a curved decoder.

**The reparameterisation trick, in words.** You can't backpropagate through "draw a random sample". So instead of sampling *z* directly, you sample a fixed bit of noise *ε* from a standard Gaussian and *build* the code as `z = mean + spread × ε`. The randomness now lives in *ε*, which has no parameters, while gradients flow cleanly through the mean and spread the encoder produced. Same samples, differentiable path.

**Fast but blurry.** Generation is a *single forward pass* through the decoder — far faster than a [GAN](gan.html)'s adversarial dance or [diffusion](diffusion.html)'s long denoising chain. The catch is softness: the Gaussian likelihood rewards a blurry average over plausible outputs rather than committing to one crisp one. VAEs trade sharpness for speed, a stable likelihood objective, and a navigable latent space.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for a VAE when

- You want PPCA's probabilistic latent-variable story but the data manifold is *curved*
- A fast, single-pass generator matters and you can tolerate some blur
- You need a smooth, navigable latent space — interpolation, attribute arithmetic
- You want a learned compressor to feed a downstream model (latent diffusion, autoregressive tokens)

</div>

<div class="no" markdown="1">

### It breaks down when

- Razor-sharp samples are the goal — the Gaussian likelihood blurs (use a [GAN](gan.html) or [diffusion](diffusion.html))
- The data really is Gaussian — then plain [PPCA](pca.html) is exact and far cheaper
- You need *exact* likelihoods — the ELBO is only a lower bound
- A strong decoder ignores the latent entirely (posterior collapse — see In-depth)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn

# A VAE is PPCA with a *neural* decoder and a *learned* encoder.
# Sample a code from N(0, I), decode it -> a new datapoint (one forward pass).
@torch.no_grad()
def sample_vae(decoder, n, d_latent):
    z = torch.randn(n, d_latent)   # 1. draw codes from the unit-Gaussian prior
    return decoder(z)              # 2. decode with a NEURAL NET (not a matrix)

# Compare to PPCA: there the decoder was just `z @ W.T + mu`.
# Make `decoder` linear and a VAE collapses straight back to PPCA.
```

</div>

<div class="level-next">
<span>Want the ELBO, the reparameterisation trick spelled out, and the PPCA bridge?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The training objective (ELBO)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\big\|\, p(z)\big)}_{\text{regulariser}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*q<sub>φ</sub>(z\|x)* — the **encoder**, outputting a Gaussian *N(μ<sub>φ</sub>(x), σ<sub>φ</sub>²(x))* over codes

</li>
<li markdown="1">

*p<sub>θ</sub>(x\|z)* — the **decoder** network; the likelihood of reconstructing *x* from *z*

</li>
<li markdown="1">

*p(z) = N(0, I)* — the same unit-Gaussian prior PPCA used

</li>
<li markdown="1">

Maximise *ℒ*: it is a lower bound on *log p(x)* (derived in the In-depth tier)

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{ELBO} = \underbrace{\text{how well codes reconstruct the input}}_{\text{reconstruction}} \;-\; \underbrace{\text{how far the encoder drifts from the prior}}_{\text{regulariser}} $$</span>

**In words.** The objective pits two pressures against each other. The **reconstruction** term rewards round-tripping: sample a `code` from the encoder's cloud given `x`, decode it, and check how close the result is to the original `x`. Left alone, that pressure would scatter codes anywhere convenient — leaving holes you can't sample from. So the **regulariser**, a KL divergence, penalises every per-input cloud for drifting away from the shared `prior` (a plain unit Gaussian). The balance packs all the codes into one smooth Gaussian-shaped region, so that after training you can draw a fresh code from the prior and the decoder knows what to do with it. Maximising the whole expression — the **ELBO**, or Evidence Lower BOund — provably pushes up the model's true log-likelihood.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`reconstruction`expected log-likelihood of decoding the input back from its own code

</li>
<li markdown="1">

`regulariser`KL distance from the encoder's per-input cloud to the unit-Gaussian prior

</li>
<li markdown="1">

`code`the latent *z*, sampled from the encoder's distribution — fewer dims than the data

</li>
<li markdown="1">

`prior`*N(0, I)* — the round cloud you sample from at generation time

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The reparameterisation trick.** The encoder emits a mean *μ* and a (log-)variance, so the spread *σ*. You need a sample *z ~ N(μ, σ²)*, but sampling isn't differentiable. The fix is to factor the randomness out: draw *ε ~ N(0, I)* and set

$$ z = \mu + \sigma \odot \varepsilon. $$

Now *z* is a deterministic, differentiable function of *μ* and *σ*; the only stochastic input *ε* carries no parameters, so gradients flow back into the encoder. Without this trick the VAE simply couldn't be trained by backprop.

**The KL term is closed-form.** Because both *q<sub>φ</sub>(z\|x)* and the prior are Gaussian, the regulariser has an exact formula — no sampling needed: $D_{\mathrm{KL}} = -\tfrac12 \sum_j \big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$. It is minimised when *μ → 0* and *σ → 1*, i.e. when each code-cloud matches the standard normal. That is precisely the pull that makes the latent space samplable.

**Why this gives PPCA back.** PPCA's posterior *p(z\|x)* was Gaussian and known in closed form because the decoder was linear. The VAE's *q<sub>φ</sub>* is the *amortised* stand-in for that posterior — one network that infers codes for any input. Make the encoder and decoder linear with a Gaussian output and the ELBO becomes tight: the VAE reduces exactly to [PPCA](pca.html). Everything new is the cost of letting the decoder curve.

**Pick the reconstruction loss by data type.** MSE for continuous pixels or audio (a Gaussian output noise model — the direct heir of PPCA's *σ²I*); per-pixel binary cross-entropy for `[0,1]` Bernoulli outputs (the original paper's MNIST choice); ordinary cross-entropy for discrete tokens.

**β-VAE and disentanglement.** Scale the KL term by a factor *β*: *ℒ = E[log p(x\|z)] − β·KL*. With *β > 1* the prior pressure rises and each latent axis is nudged to carry one independent factor of variation (pose, lighting, identity) — "disentanglement". It's an inductive bias, not a guarantee. The [Autoencoders & VAEs](../neural-networks/autoencoder.html) page covers the regularised variants in depth.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d_in=784, d_latent=32):
        super().__init__()
        self.enc       = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU())
        self.fc_mu     = nn.Linear(256, d_latent)   # encoder mean
        self.fc_logvar = nn.Linear(256, d_latent)   # encoder log-variance
        self.dec       = nn.Sequential(nn.Linear(d_latent, 256), nn.ReLU(),
                                       nn.Linear(256, d_in))  # NEURAL decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)      # z = mu + sigma * eps

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

def elbo_loss(x_hat, x, mu, logvar):
    rec = F.mse_loss(x_hat, x, reduction="sum")           # reconstruction term
    kl  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()  # KL to N(0, I)
    return rec + kl    # minimise (rec + kl)  ==  maximise the ELBO
```

</div>

<div class="level-next">
<span>Want the ELBO derived as a bound, posterior collapse, VQ-VAE, and the exact PPCA limit?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">ELBO as a lower bound on log p(x)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \log p(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x \mid z)\big] - D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{ELBO}} \;+\; \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)}_{\ge\,0\ \text{— the gap}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

The same *log p(x)* PPCA evaluates in closed form — here it is intractable

</li>
<li markdown="1">

It splits *exactly* into the ELBO plus the KL from *q<sub>φ</sub>* to the **true** posterior

</li>
<li markdown="1">

That KL is *≥ 0*, so the ELBO is a genuine lower bound; the gap shuts as *q<sub>φ</sub> → p<sub>θ</sub>(z\|x)*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \log p(\text{data}) = \underbrace{\text{ELBO}}_{\text{what we maximise}} \;+\; \underbrace{\text{distance from encoder to the TRUE posterior}}_{\ge\,0\ \text{— invisible gap}} $$</span>

**In words.** This identity is what makes VAE training principled. The quantity you actually care about — `log p(data)`, how well the model explains the data, the very thing [PPCA](pca.html) computes by hand — splits *exactly* into two pieces: the **ELBO** you can optimise, plus the KL distance from the encoder's guess to the *true* posterior (the perfect distribution over codes given the data). Because a KL distance is never negative, the ELBO can only ever sit *below* `log p(data)`, so maximising it is safe. And it does double duty: pushing the ELBO up both fits the data better and squeezes the encoder closer to the true posterior. The true posterior is exactly the object PPCA had in closed form and the curved decoder destroyed — so here we never compute the gap; we just optimise the bound.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`log p(data)`the marginal likelihood — intractable here, exact for PPCA

</li>
<li markdown="1">

`ELBO`the bound we maximise: reconstruction minus KL-to-prior

</li>
<li markdown="1">

`true posterior`*p<sub>θ</sub>(z\|x)* — what the encoder approximates; closed-form only when the decoder is linear

</li>
<li markdown="1">

`invisible gap`the KL between encoder and true posterior — never computed, only bounded

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Linear networks ⇒ PPCA, exactly.** Strip the non-linearities: let the encoder and decoder be affine maps and the decoder's output noise be isotropic Gaussian. Then *q<sub>φ</sub>(z\|x)* can represent the true Gaussian posterior, the invisible gap above closes to zero, the bound becomes tight, and the ELBO is maximised by precisely the [PPCA](pca.html) solution — *W* spanning the top principal subspace. The VAE is, quite literally, PPCA with the linear maps replaced by neural networks and the now-unreachable posterior amortised into an encoder. That is the spine of this whole arc.

**Blurriness is the Gaussian likelihood.** An MSE reconstruction loss assumes the decoder output is the mean of a Gaussian. When several sharp outputs are all plausible for one code, the likelihood-maximising choice is their *average* — and an average of crisp images is a blurry one. This is intrinsic to the maximum-likelihood objective, not a bug; it's exactly the trade the next page, [GANs](gan.html), refuses to make — abandoning likelihood for a critic that punishes blur directly.

**Posterior collapse.** Give the decoder enough power (say an autoregressive PixelCNN) and it can model the data *without* the code. The encoder then takes the free lunch: it sets *q<sub>φ</sub>(z\|x) = p(z)*, driving the KL term to zero and leaving the latent carrying no information. Standard fixes: **KL annealing** (ramp the KL weight up from zero), **free bits** (don't penalise KL below a per-dimension floor), or deliberately weakening the decoder.

**VQ-VAE — discrete latents.** Replace the Gaussian code with the nearest vector in a learned **codebook**: the encoder emits a continuous vector, you snap it to the closest entry, the decoder reconstructs from that. The discrete bottleneck sidesteps posterior collapse and produces *tokens*, which is why VQ-VAE is the image/audio tokeniser feeding autoregressive transformers (DALL·E v1, Parti) and most multi-modal LLMs.

**Latent diffusion — the VAE as compressor.** The dominant production role for VAEs today is *not* generating directly. Stable Diffusion (Rombach et al., 2022) trains a VAE to compress images to a small latent grid, then runs a [diffusion](diffusion.html) model in *that* space rather than over pixels. The VAE handles dull perceptual compression; diffusion handles the hard semantic generation — and the VAE's blurry single-pass decoder becomes a feature, not a flaw. This links the arc forward: the next-but-one page is diffusion, and a VAE is what sits underneath it.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
import torch.nn.functional as F

# Posterior-collapse defences: KL annealing + free bits
def vae_loss_robust(x_hat, x, mu, logvar, beta=1.0, free_bits=0.05):
    rec = F.mse_loss(x_hat, x, reduction="sum")
    # KL per latent dim, floored so we don't punish small but useful codes
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = (kl_per_dim - free_bits).clamp(min=0).sum()
    return rec + beta * kl

def kl_anneal(step, warmup=5000):
    return min(1.0, step / warmup)   # ramp beta 0 -> 1 over warmup steps

# The PPCA limit: a *linear* VAE. With these affine maps and a Gaussian
# output, maximising the ELBO recovers Probabilistic PCA exactly.
class LinearVAE(nn.Module):
    def __init__(self, d_in, d_latent):
        super().__init__()
        self.fc_mu     = nn.Linear(d_in, d_latent)      # linear encoder
        self.fc_logvar = nn.Linear(d_in, d_latent)
        self.dec       = nn.Linear(d_latent, d_in)      # linear decoder == W z + mu
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

[Autoencoders & VAEs, the deep view](../neural-networks/autoencoder.html)
<span class="annotation">Our own neural-networks page — the architectural and training detail this page deliberately skips, with an interactive encode/decode viz and the regularised-AE family.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Kingma & Welling (2013) — Auto-Encoding Variational Bayes <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1312.6114){: target="_blank" }
<span class="annotation">The original VAE paper. Introduces the ELBO objective and the reparameterisation trick — short, foundational, and still the clearest first read.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — From Autoencoder to Beta-VAE <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2018-08-12-vae/){: target="_blank" }
<span class="annotation">One tour through AE, denoising AE, VAE, and β-VAE with worked math. The best single source for cementing the reconstruction-plus-KL picture.</span>

</li>
<li data-tier="indepth" markdown="1">

[Kingma & Welling (2019) — An Introduction to Variational Autoencoders <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1906.02691){: target="_blank" }
<span class="annotation">The authors' modern monograph. Read §2 with PPCA in mind — the linear-Gaussian special case is the perfect anchor for amortised inference and the bound.</span>

</li>
</ul>

</div>

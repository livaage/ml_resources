---
title: Generative Adversarial Networks (GAN) — ML Resources Hub
eyebrow_text: ← Theory · Generative Models
eyebrow_href: {{root}}theory.html
heading: Generative Adversarial Networks
lead: Throw away the likelihood. Keep the latent-to-data decoder, but train it to fool a critic instead of to reconstruct — sharp samples, finicky training.
prev_href: vae.html
prev_title: Variational Autoencoders
next_href: diffusion.html
next_title: Diffusion Models
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Keep the decoder; throw away the likelihood.** PPCA and the [VAE](vae.html) both kept the same skeleton — *sample a latent code, push it through a decoder, get data* — and trained that decoder to maximise (a bound on) the likelihood of the real data. A GAN keeps the very same skeleton: a *generator* still maps a noise vector *z* to a sample *G(z)*. What it throws out is the likelihood objective. Instead of asking "how probable is the real data under my model?", it trains a second network — a *discriminator* — to tell real samples from generated ones, and trains the generator to fool it. The latent→decoder skeleton survives; the **training signal** is what changes.

</div>

<article class="tldr-body" markdown="1">

The [VAE](vae.html) optimises a lower bound on *log p(x)* and pays a tax for it: its reconstruction term is (usually) a Gaussian likelihood, which means "be close to the real pixel on average". When several plausible outputs exist, averaging them is the safe bet — and the average of many sharp faces is a blurry face. Every likelihood-based decoder in this section inherits some version of that blur. A GAN makes a radical move to escape it: **abandon likelihood entirely.**

**The counterfeiter and the detective.** Picture a counterfeiter (the generator) printing fake banknotes from scratch, and a detective (the discriminator) trying to spot them. The detective gets better at noticing tells; the counterfeiter gets better at hiding them. Run that arms race long enough and the fakes become indistinguishable from the real thing. There is no "compare pixel to pixel" anywhere in this loop — the loss function *is* a learned network that notices whatever is currently wrong. That's why GAN samples are sharp where VAE samples are soft: nothing rewards hedging.

**No encoder, no likelihood.** Notice what's gone. There is no encoder mapping data back to codes (the basic GAN doesn't need one), and crucially there is no *p(x)* — you cannot ask a trained GAN "how likely is this image?" or score a held-out point. That's a real loss. Likelihood is what powered PPCA's model selection and anomaly detection; a GAN gives none of it. To know whether a GAN is any good you have to *look at samples* and fall back on proxy metrics like [FID](#) (Fréchet Inception Distance), which compares the statistics of generated and real samples in a feature space.

**The price of sharpness.** Two networks chasing each other is a *minimax game*, not a descent on a fixed loss — and that makes training famously unstable. The signature failure is **mode collapse**: the generator discovers one output (or a few) that reliably fools the current discriminator and pours all its mass there, ignoring the rest of the data distribution. There's no term anywhere that rewards diversity. Sharp but narrow is a GAN's characteristic failure; blurry but complete is a VAE's. For the full architectural and training-stability story see [GANs, in depth](../neural-networks/gan.html).

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### The adversarial framing helps when

- Sample *sharpness* matters more than likelihood — faces, textures, art
- You need single-pass sampling: one forward through *G*, no iterative loop
- You want a smooth, navigable latent space to *edit* in (StyleGAN-style)
- Image-to-image translation, super-resolution, or distilling a slow model

</div>

<div class="no" markdown="1">

### It breaks down when

- You need a likelihood — model selection, anomaly scores, density estimation
- Full mode coverage is critical — GANs drop modes; use [diffusion](diffusion.html)
- You want forgiving, stable training without convergence drama
- Best-in-class fresh image quality — [diffusion](diffusion.html) has largely won that

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

# Same latent->decoder skeleton as PPCA/VAE — the generator IS the decoder.
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, img_dim), nn.Tanh(),   # noise z -> data
        )
    def forward(self, z): return self.net(z)

# The new piece: a learned loss. D scores "how real does this look?"
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)
```

</div>

<div class="level-next">
<span>Want the minimax objective, why it's really a divergence, and where the instability comes from?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The minimax game</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \min_G \max_D \;\; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] \;+\; \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*D(x)* is the discriminator's estimated probability that *x* is real

</li>
<li markdown="1">

*D* **maximises**: push *D(x)*→1 on reals, *D(G(z))*→0 on fakes

</li>
<li markdown="1">

*G* **minimises**: push *D(G(z))*→1 — make fakes look real

</li>
<li markdown="1">

No *p(x)* anywhere — the objective is a *game*, not a likelihood

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \min_G \max_D \;\; \text{avg over real}[\log D(\text{real})] \;+\; \text{avg over fakes}[\log(1 - D(\text{decode}(\text{noise})))] $$</span>

**In words.** Two players share one scoreboard. The **discriminator** `D` reads a sample and outputs a number between 0 and 1 — "how likely is this real?". It wants to score reals near 1 and fakes near 0, and *maximising* the expression does exactly that (both terms grow when D is right). The **generator** `G` is the old decoder: it turns `noise` into a fake and tries to *minimise* the same expression, which means driving `D(decode(noise))` toward 1 — fooling D. `avg` (the `𝔼`) just means "average over many samples". Notice there is no reconstruction term and no `p(data)`: the only learning signal is whether D can be fooled. At the ideal equilibrium G's samples are indistinguishable from real data and D is reduced to a coin flip.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`decode(noise)`the generator *G(z)* — the same latent→data map PPCA and the VAE had

</li>
<li markdown="1">

`D`the learned loss: a network that scores realism instead of a fixed formula

</li>
<li markdown="1">

`max` then `min`a two-player game, not a single descent — the root of the instability

</li>
<li markdown="1">

`coin flip`at the optimum *D(x) = ½* everywhere: it can no longer tell real from fake

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The non-saturating fix.** Train by alternating: one gradient step up on *D*, one step down on *G*, repeat. There's a snag — early on, fakes are obviously fake, so `D(G(z))≈0`, `log(1 − D(G(z)))≈0`, and its gradient *vanishes* exactly when the generator most needs to learn. The standard repair is the **non-saturating loss**: have *G* maximise `log D(G(z))` (equivalently minimise `−log D(G(z))`) instead. Same fixed point, strong gradient when D is winning. Every modern GAN uses this — it's in the code below.

**What the game is secretly optimising.** Hold *G* fixed and solve for the best *D*. The optimum is the density ratio *D\*(x) = p<sub>data</sub>(x) / (p<sub>data</sub>(x) + p<sub>g</sub>(x))*. Substitute it back and the generator's objective becomes the **Jensen–Shannon divergence** between the real and generated distributions (up to a constant). So a GAN *is* doing distribution matching — just not by maximising likelihood. This is the bridge back to the rest of the section: PPCA and the VAE minimise (a bound on) the *forward* KL to the data; the GAN minimises a symmetric JS divergence *estimated by a network* rather than computed in closed form.

**Why JS is fragile.** When the real and generated distributions barely overlap — which is the norm early in training, when fakes are noise — the JS divergence is locally flat, so the generator gets near-zero gradient. That's the formal reason vanilla GANs stall, and the motivation for the Wasserstein objective in the next tier.

**Mode collapse.** Because *G* is graded sample-by-sample (does *this* fake fool D?) and never on the diversity of its *distribution*, it can win by emitting one perfect output. It collapses onto a mode; D learns to flag everything else as fake, which only reinforces the collapse. Nothing in the minimax loss pushes back. The standard countermeasures — minibatch discrimination, unrolled GANs, and especially the Wasserstein loss — all add weak diversity pressure; the [deep GAN page](../neural-networks/gan.html) catalogues the practical training tricks (TTUR, spectral norm, label smoothing, EMA of weights).

**Conditioning and evaluation.** Feed a label or embedding *y* to both networks and you get a **conditional GAN**: `G(z, y)` makes class-conditional samples, `D(x, y)` judges them in context — the basis of pix2pix and class-conditional image models. And because there's no likelihood to report, GANs are scored by proxies: **FID** compares Inception-feature statistics of real vs generated samples (lower is better, the de facto standard), **IS** (Inception Score) rewards recognisability plus diversity, and precision/recall metrics split "are samples realistic?" from "do they cover the data?".

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# One GAN training step with the non-saturating generator loss.
def train_step(real_x, G, D, g_opt, d_opt, z_dim=100):
    B = real_x.size(0)

    # --- Train D: maximise log D(real) + log(1 - D(fake)) ---
    z      = torch.randn(B, z_dim, device=real_x.device)
    fake_x = G(z).detach()                    # detach: no grad to G here
    d_loss = F.binary_cross_entropy(D(real_x), torch.ones(B, 1)) \
           + F.binary_cross_entropy(D(fake_x), torch.zeros(B, 1))
    d_opt.zero_grad(); d_loss.backward(); d_opt.step()

    # --- Train G: minimise -log D(G(z))  (non-saturating) ---
    z      = torch.randn(B, z_dim, device=real_x.device)
    g_loss = F.binary_cross_entropy(D(G(z)), torch.ones(B, 1))   # target=1: "look real"
    g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    return d_loss.item(), g_loss.item()
```

</div>

<div class="level-next">
<span>Want the Wasserstein objective, why it stabilises training, and where GANs stand against diffusion?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Wasserstein GAN objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ W(p_r, p_g) = \sup_{\lVert f \rVert_L \le 1} \;\; \mathbb{E}_{x \sim p_r}[f(x)] \;-\; \mathbb{E}_{z \sim p_z}[f(G(z))] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*W* is the **Wasserstein-1** (earth-mover) distance between real *p<sub>r</sub>* and generated *p<sub>g</sub>*

</li>
<li markdown="1">

*f* is the **critic** — replaces the discriminator; outputs a raw score, no sigmoid

</li>
<li markdown="1">

‖f‖<sub>L</sub> ≤ 1: *f* must be **1-Lipschitz** (its slope is capped at 1)

</li>
<li markdown="1">

Gradient stays meaningful even when *p<sub>r</sub>* and *p<sub>g</sub>* don't overlap

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ W = \max_{f \,\text{(bounded slope)}} \;\; \text{avg over real}[\,f(\text{real})\,] \;-\; \text{avg over fake}[\,f(\text{decode}(\text{noise}))\,] $$</span>

**In words.** Drop the log and the sigmoid. The discriminator becomes a **critic** `f` that outputs a raw realness score and is asked to push scores *up* on real samples and *down* on fakes; the gap between the two averages estimates the **earth-mover distance** — informally, *"how much probability mass would you have to haul, and how far, to turn the generated distribution into the real one"*. The one crucial leash is that `f` must be **1-Lipschitz**: its output can't change faster than its input, so its slope is capped. That cap is exactly what keeps the gradient alive when reals and fakes don't yet overlap — it replaces the flat-and-useless JS gradient of the vanilla GAN with a smooth, always-informative one.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`critic f`like the discriminator but unbounded — it ranks realness instead of classifying

</li>
<li markdown="1">

`earth-mover`cost of the cheapest plan to reshape one distribution into the other

</li>
<li markdown="1">

`bounded slope`the 1-Lipschitz constraint — enforced by gradient penalty or spectral norm

</li>
<li markdown="1">

`always-informative`Wasserstein gradients don't vanish on disjoint supports, unlike JS

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Why WGAN stabilises training.** The vanilla objective minimises Jensen–Shannon divergence, which goes flat — zero gradient — whenever real and fake distributions are disjoint, exactly the situation at the start of training. The Wasserstein distance varies smoothly even then, so the critic always hands the generator a usable direction to move. Empirically this removes much of the convergence drama: loss curves become meaningful (the critic's score *correlates* with sample quality), and mode collapse eases because the critic measures a *distributional* distance rather than per-sample realness.

**Enforcing the Lipschitz constraint.** The whole trick rests on keeping *f* 1-Lipschitz, and there are three generations of how. Original WGAN used crude **weight clipping**. **WGAN-GP** ([Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028)) added a **gradient penalty** — penalise the critic when the norm of its gradient at points interpolated between real and fake samples drifts away from 1 (sketched in the code). **Spectral normalisation** ([Miyato et al., 2018](https://arxiv.org/abs/1802.05957)) caps each weight matrix's largest singular value — one line in PyTorch and the modern default. The [deep GAN page](../neural-networks/gan.html) covers these alongside the rest of the stability toolkit.

**What GANs are uniquely good at: latent-space editing.** StyleGAN-class generators ([Karras et al.](https://arxiv.org/abs/1912.04958)) feed a learned *style* code in at every layer and produce a latent space that is strikingly **disentangled** — single directions correspond to interpretable changes (age, pose, smile) with identity held fixed. Because a GAN is a single fast map *z → x*, you can walk that latent space and watch a face morph smoothly. This kind of clean, real-time semantic editing is something likelihood models and even diffusion still don't match as cleanly.

**The honest comparison with diffusion.** For unconditional and text-to-image quality and for *mode coverage*, [diffusion models](diffusion.html) have largely overtaken GANs: they train stably, cover the full distribution, and hit the highest sample quality. What GANs keep is **speed** — a GAN samples in a *single forward pass*, while diffusion needs tens to hundreds of denoising steps. That one number drives a lot: real-time filters, on-device generation, and the wave of *diffusion-distilled* GANs (ADD, SDXL-Turbo) that train a one-step generator adversarially against a slow diffusion teacher to reclaim the speed.

**Where this points next.** The [diffusion](diffusion.html) page is the resolution of the tension this page set up. It keeps the high sample quality that made GANs exciting, but recovers what GANs gave away — stable training, full mode coverage, and a genuine likelihood interpretation — by replacing the single adversarial leap with a long, gentle chain of denoising steps. The cost is precisely the speed GANs still own. That speed/quality trade-off is the through-line of the whole section; see the [overview](../generative-models.html) for the four families side by side.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch

# WGAN-GP critic loss: maximise the Wasserstein gap, enforce 1-Lipschitz via gradient penalty.
def wgan_gp_critic_loss(real_x, fake_x, critic, lambda_gp=10.0):
    # Wasserstein gap: critic scores reals high, fakes low (no sigmoid, no log).
    d_real = critic(real_x).mean()
    d_fake = critic(fake_x).mean()
    w_loss = d_fake - d_real                         # minimise -> maximise (real - fake)

    # Gradient penalty: push ||grad critic|| toward 1 at random interpolations.
    eps   = torch.rand(real_x.size(0), 1, 1, 1, device=real_x.device)
    x_hat = (eps * real_x + (1 - eps) * fake_x).requires_grad_(True)
    grad  = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)[0]
    gp    = ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()

    return w_loss + lambda_gp * gp                   # generator loss is just -critic(G(z)).mean()
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

[GANs, in depth](../neural-networks/gan.html)
<span class="annotation">Our own deep GAN page — architecture (DCGAN, StyleGAN), the full stability toolkit, and an interactive 2-D GAN you can train live to watch mode collapse happen.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Goodfellow et al. (2014) — Generative Adversarial Nets <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1406.2661){: target="_blank" }
<span class="annotation">The original paper, and surprisingly readable. Sections 2–4 derive the minimax game, the optimal discriminator, and the Jensen–Shannon-divergence interpretation used above.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — From GAN to WGAN <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2017-08-20-gan/){: target="_blank" }
<span class="annotation">The best single blog tour: walks from vanilla GAN through the JS-divergence problem to WGAN and WGAN-GP, with the math behind every step.</span>

</li>
<li data-tier="indepth" markdown="1">

[Arjovsky et al. (2017) — Wasserstein GAN <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1701.07875){: target="_blank" }
<span class="annotation">The paper that diagnosed why JS-divergence gradients vanish and replaced them with the earth-mover distance. The clearest source for the In-depth formula on this page.</span>

</li>
</ul>

</div>

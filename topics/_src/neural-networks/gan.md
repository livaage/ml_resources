---
title: Generative Adversarial Networks (GAN) — ML Resources Hub
eyebrow_text: ← Theory · Neural Networks
eyebrow_href: {{root}}theory.html
heading: Generative Adversarial Networks
lead: A generator and a discriminator locked in a minimax game — the architecture that opened the modern image-generation era.
prev_href: autoencoder.html
prev_title: Autoencoders & VAEs
next_href: diffusion.html
next_title: Diffusion Models
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**Two networks playing against each other.** A *generator* invents fake samples from noise. A *discriminator* tries to spot the fakes. They train in an adversarial loop — each gets better as the other improves, like a counterfeiter and a detective sharpening each other's skills.
</div>

<div class="viz-embed viz-gan" data-fig="gan">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            A <strong>generator</strong> tries to create fake samples; a <strong>discriminator</strong> tries to spot the fakes. They train in an adversarial loop. Hit <strong>Train</strong> and watch the orange (fake) particles slide toward the indigo (real) distribution as the discriminator pushes back. Switch to <strong>Two clusters</strong> to watch <em>mode collapse</em> — the generator pours its mass into a single mode and ignores the other.
        </span>
    </div>
    <div class="viz-embed-header">
        <span class="viz-embed-title">A 2-D GAN you can train live — orange wash = "fake", indigo wash = "real" in the discriminator's eyes</span>
    </div>
    <div class="viz-gan-controls">
        <button id="viz-gan-step"  type="button">Step</button>
        <button id="viz-gan-train" type="button">Train</button>
        <button id="viz-gan-reset" type="button">Reset</button>
        <label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
            Real distribution
            <select id="viz-gan-dist"></select>
        </label>
        <span class="viz-gan-counter" id="viz-gan-counter">Step 0</span>
    </div>
    <div class="viz-gan-canvas-wrap">
        <canvas id="viz-gan-canvas"></canvas>
    </div>
    <div class="viz-gan-caption" id="viz-gan-caption"></div>
</div>
<script src="{{root}}js/viz/gan.js"></script>

<article class="tldr-body" markdown="1">

### Why an adversarial game

No pixel-wise loss can capture what makes an image "look real". L2 distance to the nearest training image rewards blurry averages — that's exactly the failure mode of a vanilla [autoencoder]({{root}}topics/neural-networks/autoencoder.html). Realism is *perceptual*: a face with one eye higher than the other is mathematically close to a real face but obviously wrong. GANs sidestep the problem by **learning the loss function itself** — the discriminator is a neural network whose job is to notice exactly the kinds of mistakes the generator is currently making. As long as something is off, D has something to penalise; once nothing is off, training is done.

### Why GANs are unstable

The minimax setup is a **two-player game with two moving targets**. Standard supervised training optimises one loss against a fixed ground truth; here the generator's "ground truth" is whatever D currently believes, and D's "ground truth" is whatever G is currently producing. Both keep changing. The four classic pathologies:

- **Discriminator dominates** — D gets perfect, the generator's gradient saturates to zero, training stalls.
- **Generator dominates** — G finds a blind spot in D and exploits it forever.
- **Oscillation** — both networks chase each other in circles without converging.
- **Vanishing gradients on disjoint supports** — when real and fake distributions don't overlap, the JS-divergence the original objective is computing is locally flat, so neither side learns.

### Mode collapse — what it is and why it happens

Hit the **Two clusters** preset in the figure. The generator quickly discovers that putting *all* its mass on one cluster fools D almost as well as covering both — and once it commits, D learns to flag the *other* cluster as fake, which only reinforces the collapse. There's no term in the minimax loss that rewards diversity; the generator is graded on each sample individually, not on its distribution as a whole. WGAN, minibatch discrimination, and unrolled GANs all add weak diversity pressure; diffusion sidesteps the problem entirely by training a denoiser.

### Where GANs still win

Diffusion took over for fresh high-quality image generation, but GANs are still the default for:

- **Speed at inference** — one forward pass, no 20–50 denoising steps. Real-time face filters, video, games.
- **Paired image-to-image translation** — pix2pix, CycleGAN. The adversarial loss handles the "make it look like a Monet" half of the problem cleanly.
- **Super-resolution** — ESRGAN-style models still produce sharper textures than diffusion at the same compute.
- **Identity-preserving editing** — StyleGAN's disentangled latent space lets you change "age" or "smile" without touching identity. Diffusion is catching up but isn't there yet.
- **Distillation of diffusion** — many of the fastest "single-step" image models today are GANs trained to mimic a teacher diffusion model.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- You need single-pass inference (real-time, on-device, video)
- Paired or unpaired image-to-image translation (pix2pix, CycleGAN)
- Super-resolution with sharp, plausible textures
- Identity-preserving face / object editing in a learned style space
- Distilling a slow diffusion model into a one-step generator

</div>
<div class="no" markdown="1">

### Skip it when

- Fresh text-to-image at SOTA quality — diffusion has won this fight
- You need mode coverage of a complex multi-modal distribution
- You want forgiving, stable training without convergence drama
- You need likelihoods or controllable inference at sample time

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, img_dim), nn.Tanh(),
        )
    def forward(self, z): return self.net(z)

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
    <span>Want the minimax objective, DCGAN, and the training tricks that actually keep it stable?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">The minimax game</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \min_G \max_D \;\; \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[\log D(\mathbf{x})] \;+\; \mathbb{E}_{\mathbf{z}\sim p_z}[\log(1 - D(G(\mathbf{z})))] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>D(x)</code>discriminator's probability that *x* is real</li>
<li markdown="1"><code>G(z)</code>generator's output given noise *z*</li>
<li markdown="1"><code>D</code>maximises — wants real scored near 1, fakes near 0</li>
<li markdown="1"><code>G</code>minimises — wants <code>D(G(z))</code> near 1, i.e. wants to fool D</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \min_G \max_D \;\; \text{avg over real}[\log D(\text{real})] \;+\; \text{avg over fakes}[\log(1 - D(G(\text{noise})))] $$</span>

**In words.** Two players. The **discriminator** `D` outputs a number between 0 and 1 — "how likely is this real?" Its goal is to score reals near 1 and fakes near 0, which is exactly what *maximising* the expression does (both terms get larger when D is right). The **generator** `G` turns random noise into fake samples and tries to *minimise* the same expression, which means pushing `D(G(noise))` close to 1 — fooling D. `𝔼` just means "average over many samples". At the theoretical equilibrium, G produces samples indistinguishable from the data and D is reduced to a 50/50 guess.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Alternating optimisation.** One gradient step on *D* (maximise), then one on *G* (minimise), repeat. In practice the generator's gradient *saturates* early — fakes are obviously fake, so `log(1 − D(G(z)))` is near zero and so is its derivative. The standard fix is the **non-saturating loss**: G minimises `−log D(G(z))` instead of `log(1 − D(G(z)))`. Same fixed point, much stronger gradient when D is winning. Every modern GAN uses this.

**DCGAN — the convolutional recipe.** Radford et al. (2015) figured out the architecture choices that made image GANs reliably trainable:

- **Strided convolutions** instead of pooling in D; **fractionally-strided (transposed) convs** instead of upsampling in G.
- **BatchNorm** in both networks, but *never* in G's output layer or D's input layer (it destroys the input statistics).
- **ReLU** in G (Tanh on the output), **LeakyReLU** in D.
- No fully-connected hidden layers — just convs all the way down.

These choices remained the default for years and still inform modern GAN designs.

**Conditional GANs.** Concatenate a label (or a learned class embedding) to both the generator input and the discriminator input. Now `G(z, y)` generates class-conditional samples and `D(x, y)` evaluates them given the class. The foundation for pix2pix, BigGAN, and StyleGAN's class-conditioned variants.

**Training-stability tricks that actually help.**

- **Two time-scale update rule (TTUR).** Run D at a higher learning rate than G (e.g. `lr_D = 4e-4`, `lr_G = 1e-4`). Lets D track the generator without dominating.
- **Wasserstein loss + gradient penalty (WGAN-GP).** Replace the BCE objective with a critic that estimates earth-mover distance; gradients flow even when the supports don't overlap. (Full derivation in the In-depth tier.)
- **One-sided label smoothing.** Train D against real labels of `0.9` instead of `1.0` — stops D from becoming overconfident.
- **Spectral normalisation.** Constrain every weight matrix's largest singular value to 1; cheaper and cleaner than gradient penalty for enforcing Lipschitz-ness on D.
- **Exponential moving average of G's weights.** Sample from the EMA copy, not the live G — dramatically smoother results.

**Failure-mode diagnostics.** Plot both losses. If `loss_D → 0` while `loss_G → ∞`, D has won and G is stuck. If they oscillate wildly, lower the learning rate or add spectral norm. If sample diversity collapses, that's mode collapse — try minibatch discrimination or switch to WGAN-GP.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Real-time image generation — single forward pass beats diffusion's sampling loop
- Image-to-image translation (CycleGAN, pix2pix)
- Domain adaptation between unpaired distributions
- StyleGAN-style controllable face / object synthesis with a navigable latent space

</div>
<div class="no" markdown="1">

### Skip it when

- Mode coverage is critical — diffusion captures more of the distribution
- You need text-to-image at high quality — modern diffusion wins decisively
- You can't afford training instability or careful tuning
- You need likelihoods or controllable inference at sample time

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F

# Standard GAN training step with the non-saturating generator loss
def train_step(real_x, G, D, g_opt, d_opt, z_dim=100):
    B = real_x.size(0)

    # --- Train D ---
    z      = torch.randn(B, z_dim, device=real_x.device)
    fake_x = G(z).detach()
    d_real = D(real_x)
    d_fake = D(fake_x)
    # One-sided label smoothing on the real side
    d_loss = F.binary_cross_entropy(d_real, 0.9 * torch.ones_like(d_real)) \
           + F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
    d_opt.zero_grad(); d_loss.backward(); d_opt.step()

    # --- Train G with non-saturating loss ---
    z      = torch.randn(B, z_dim, device=real_x.device)
    fake_x = G(z)
    d_fake = D(fake_x)
    # minimise -log D(G(z))  ==  BCE with target=1
    g_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
    g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    return d_loss.item(), g_loss.item()
```

</div>

<div class="level-next">
    <span>Want WGAN, StyleGAN, BigGAN, and where GANs lost to diffusion?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Wasserstein GAN objective</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \min_G \max_{D \in \mathcal{D}_{1\text{-Lip}}} \;\; \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}[D(\mathbf{x})] \;-\; \mathbb{E}_{\mathbf{z}\sim p_z}[D(G(\mathbf{z}))] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>𝒟<sub>1-Lip</sub></code>1-Lipschitz functions — D's gradient is capped at 1</li>
<li markdown="1">Approximates the Wasserstein-1 distance between real and generated distributions</li>
<li markdown="1">Gradient stays meaningful when supports don't overlap (unlike JS-divergence)</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \min_G \max_{D \,\text{(bounded slope)}} \;\; \text{avg over real}[\,D(\text{real})\,] \;-\; \text{avg over fake}[\,D(\text{fake})\,] $$</span>

**In words.** Drop the log and the sigmoid. The discriminator (now called a *critic*) outputs a raw score and is asked to push scores up on real samples and down on fakes. The difference of the two averages approximates the **Wasserstein-1 distance** between the two distributions — informally, *"how much earth would you have to move to turn one into the other"*. The crucial restriction is that `D` must be **1-Lipschitz**, meaning its slope is bounded: a small change in input causes only a small change in output. That restriction is what keeps the gradient flowing even when real and fake samples don't overlap, which fixes the vanilla GAN's saturating-gradient problem at the root.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**WGAN (Arjovsky et al., 2017).** The vanilla GAN objective implicitly minimises the Jensen–Shannon divergence. When real and fake distributions don't overlap — which they don't at the start of training, when fakes are pure noise — JS is locally constant and the gradient is zero. Wasserstein distance gives meaningful gradients everywhere. The catch is implementing it: D must be 1-Lipschitz.

**Lipschitz enforcement, three generations.** Original WGAN used **weight clipping** to a small range — crude and degrades capacity. WGAN-GP ([Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028)) added a **gradient penalty**: penalise D when its gradient norm at interpolated points between real and fake samples drifts from 1. **Spectral normalisation** ([Miyato et al., 2018](https://arxiv.org/abs/1802.05957)) bounds the largest singular value of each weight matrix — one line of code in PyTorch, cheap at runtime, and the modern default.

**StyleGAN — controllable generation.** [Karras et al. (2018)](https://arxiv.org/abs/1812.04948) replaced the generator's input vector with a learned **style code** injected at every layer via adaptive instance norm. The intermediate latent space `W` is dramatically more disentangled than the raw `Z` — directions in W correspond to interpretable changes (age, pose, smile). StyleGAN2 fixed water-droplet artifacts; **StyleGAN3** ([Karras et al., 2021](https://arxiv.org/abs/2106.12423)) made the generator equivariant to translation and rotation, which is what made smooth face-animation interpolations possible.

**CycleGAN — unpaired translation.** Two generators (A→B and B→A) and two discriminators. The "cycle-consistency" loss requires that translating an image to the other domain and back recovers the original. Lets you turn horses into zebras, photos into Monets, summer into winter — *without* paired training data. Still the textbook approach for unpaired image translation.

**BigGAN — GANs at scale.** [Brock et al. (2018)](https://arxiv.org/abs/1809.11096) scaled class-conditional GANs to ImageNet: batch size 2048, large models, careful orthogonal regularisation, the truncation trick at sampling time. Then-SOTA on ImageNet. Demonstrated that GANs respond well to compute but require fastidious hyperparameter tuning to stay stable at scale.

**How diffusion ate GANs' lunch (mostly).** Mode coverage, sample diversity, training stability, controllability, and text conditioning all favour diffusion. GANs retain a decisive edge on **inference speed** — one forward pass versus 20–50 denoising steps. The rapprochement is happening from both sides: consistency models and distillation methods are shrinking diffusion's step count, while diffusion-distilled GANs (e.g. ADD, SDXL Turbo) hand the speed crown back to one-step generators trained adversarially against a teacher.

**Evaluation.** GANs don't give likelihoods, so we use proxy metrics. **FID** (Fréchet Inception Distance) compares feature statistics of real and generated samples in Inception-v3 feature space — the de facto standard. **IS** (Inception Score) rewards both recognisability and diversity but is brittle outside ImageNet. **KID** is FID's unbiased cousin. **Precision / Recall in feature space** ([Sajjadi et al., 2018](https://arxiv.org/abs/1806.00035)) splits FID's single number into "are samples realistic?" (precision) and "do they cover the data?" (recall) — the clearest diagnostic for mode collapse.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- **StyleGAN**: face / object synthesis with a navigable, disentangled style space
- **CycleGAN**: unpaired image-to-image translation
- **Conditional GANs**: real-time generation for interactive applications
- **Distilling diffusion**: training a one-step student against a multi-step teacher
- Inference speed dominates sample-quality concerns

</div>
<div class="no" markdown="1">

### Skip it when

- Text-to-image — diffusion has won this fight decisively
- You need broad coverage of a complex multi-modal distribution
- You want straightforward training without convergence drama
- You need probabilistic outputs or per-step controllable inference

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Spectral-norm discriminator + hinge loss — the modern stable recipe
class SNDiscriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),  nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),           nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),          nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 1, 4, 1, 0)),
        )
    def forward(self, x):
        return self.net(x).view(x.size(0), -1).mean(dim=1)

# Hinge loss — pairs well with spectral norm, used by SAGAN, BigGAN, StyleGAN2
def hinge_d_loss(d_real, d_fake):
    return torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()

def hinge_g_loss(d_fake):
    return -d_fake.mean()
```

</div>

<div class="level-next">
    <span>Too dense?</span>
    <button data-go-to="fundamentals" type="button">← Back to Fundamentals</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="gan" markdown="1">

### Reading the figure

Each frame is a snapshot of the **2-D distribution game**. Indigo dots are real samples drawn from the chosen distribution; orange dots are fakes produced by the generator. The two coloured washes in the background are the **discriminator's current belief surface** — orange where D thinks "fake", indigo where D thinks "real". Sharp boundary = confident D; blurry boundary = D is unsure.

Each **Train** step does one gradient update on D, then one on G. The generator's fakes climb the gradient of D toward higher-indigo regions, and D redraws its boundary to chase them. Watch the orange wash tighten around the orange points and the indigo wash settle on the real samples — that's the minimax game converging.

Switch the **real distribution** to *Two clusters* and train past 200 steps. The generator will collapse onto one cluster while ignoring the other — classic **mode collapse**. There's nothing in the loss that rewards diversity; G is graded sample-by-sample, not on the distribution it covers.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[Lilian Weng — From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The single best blog tour — walks from vanilla GAN through WGAN, WGAN-GP, and the variants that followed, with the math behind each step.</span>
</li>
<li data-tier="intuition" markdown="1">
[Soumith Chintala — How to train a GAN (NIPS 2016 tutorial)](https://github.com/soumith/ganhacks) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The classic compilation of empirical training tricks. Some are dated, but the principles (label smoothing, lr balancing, label-noise) still apply.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Goodfellow et al. (2014) — Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The original paper. Surprisingly readable. Sections 2–4 cover the theory of the minimax game and the global-optimum proof.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Goodfellow (2016) — NIPS Tutorial on GANs](https://arxiv.org/abs/1701.00160) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Goodfellow's own 80-page survey two years after the original paper. The clearest single document on the framework, common pitfalls, and the open problems as of 2016.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Salimans et al. (2016) — Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Introduced feature matching, minibatch discrimination, one-sided label smoothing, and the Inception Score. The paper behind half the tricks in the ganhacks list.</span>
</li>
<li data-tier="indepth" markdown="1">
[Karras et al. (2021) — StyleGAN3 (Alias-Free GAN)](https://arxiv.org/abs/2106.12423) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Makes the generator equivariant to translation and rotation by treating intermediate features as continuous signals — what enables smooth face-animation interpolations.</span>
</li>
<li data-tier="indepth" markdown="1">
[Brock et al. (2018) — BigGAN](https://arxiv.org/abs/1809.11096) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Class-conditional GANs at ImageNet scale. The orthogonal regularisation and truncation-trick ideas matter beyond GANs themselves.</span>
</li>
</ul>

</div>
</content>
</invoke>
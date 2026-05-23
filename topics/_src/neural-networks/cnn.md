---
title: Convolutional Neural Networks (CNN) — ML Resources Hub
eyebrow_text: ← Theory · Neural Networks
eyebrow_href: {{root}}theory.html
heading: Convolutional Neural Networks
lead: Translation-equivariant networks for grid-structured data — images, video, audio spectrograms.
prev_href: ../neural-networks.html
prev_title: Neural Networks
next_href: rnn.html
next_title: Recurrent Neural Networks
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**Share one small filter across the whole image — and learn what it should detect.** A CNN scans a tiny window (the *kernel*) over the input, looking for a specific pattern: an edge, a texture, a colour blob. The same handful of weights is reused at every position, so the network has dramatically fewer parameters than a dense net and already "knows" that a pattern in the top-left is the same kind of thing in the bottom-right. **The kernel weights aren't designed by hand — they're learned by gradient descent from a loss signal**, the same way every other neural-net weight is. Stack many such layers and the network grows from edges, to textures, to parts, to whole objects.
</div>

<div class="viz-embed viz-cnn" data-fig="cnn">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            The SAME 3×3 filter slides over every position of an image. Try <strong>preset filters</strong> (edges, blur, sharpen) to see hand-designed kernels in action, or switch to <strong>Learn the filter</strong> and watch a random kernel converge to the target via gradient descent. The 9 filter weights aren't magic — they're learned from a loss signal, just like every other weight in a neural net.
        </span>
    </div>
    <div class="viz-cnn-controls">
        <label class="viz-cnn-label-inline">
            Image
            <select id="viz-cnn-image"></select>
        </label>
        <div class="viz-cnn-mode-toggle">
            <button type="button" data-cnn-mode="preset" class="active">Use a preset filter</button>
            <button type="button" data-cnn-mode="learn">Learn the filter</button>
        </div>
        <div class="viz-cnn-mode-pane viz-cnn-pane-preset">
            <label class="viz-cnn-label-inline">
                Filter
                <select id="viz-cnn-kernel-select"></select>
            </label>
            <button id="viz-cnn-play" type="button">Pause</button>
        </div>
        <div class="viz-cnn-mode-pane viz-cnn-pane-learn" hidden>
            <label class="viz-cnn-label-inline">
                Target
                <select id="viz-cnn-target-select"></select>
            </label>
            <button id="viz-cnn-auto" type="button" class="viz-cnn-primary">▶ Auto-train</button>
            <button id="viz-cnn-step" type="button">+1 step</button>
            <button id="viz-cnn-reset" type="button">Reset</button>
        </div>
    </div>
    <div class="viz-cnn-layout">
        <div class="viz-cnn-panel">
            <div class="viz-cnn-label">Input image · 24×24</div>
            <canvas id="viz-cnn-input"></canvas>
            <div class="viz-cnn-sub">the same filter slides over every position</div>
        </div>
        <div class="viz-cnn-panel">
            <div class="viz-cnn-label">Filter · 3×3 = 9 weights</div>
            <canvas id="viz-cnn-kernel"></canvas>
            <div class="viz-cnn-sub" id="viz-cnn-kernel-sub">applied at every position</div>
        </div>
        <div class="viz-cnn-panel">
            <div class="viz-cnn-label">Feature map · 22×22</div>
            <canvas id="viz-cnn-output"></canvas>
            <div class="viz-cnn-sub">indigo = positive · orange = negative</div>
        </div>
    </div>
    <div class="viz-cnn-learn-row" id="viz-cnn-learn-row" hidden>
        <div class="viz-cnn-panel">
            <div class="viz-cnn-label">Target feature map · what we want</div>
            <canvas id="viz-cnn-target"></canvas>
            <div class="viz-cnn-sub">target filter applied to the image</div>
        </div>
        <div class="viz-cnn-loss-panel">
            <div class="viz-cnn-label">Training loss (MSE)</div>
            <canvas id="viz-cnn-loss"></canvas>
            <div class="viz-cnn-readout" id="viz-cnn-readout">step 0 · loss —</div>
            <div class="viz-cnn-sub">each "Step" runs one gradient-descent update on the 9 filter weights</div>
        </div>
    </div>
    <div class="viz-cnn-pool">
        <div class="viz-cnn-pool-header">
            <div class="viz-cnn-pool-title">
                <strong>Pooling.</strong> Downsample the feature map by taking the max (or mean) of every 2×2 block — fewer numbers, larger receptive field. Hover either side to see the block-to-cell correspondence.
            </div>
            <div class="viz-cnn-mode-toggle">
                <button type="button" data-pool-mode="max" class="active">Max pool 2×2</button>
                <button type="button" data-pool-mode="avg">Average pool 2×2</button>
            </div>
        </div>
        <div class="viz-cnn-pool-layout">
            <div class="viz-cnn-panel">
                <div class="viz-cnn-label">Feature map · 22×22</div>
                <canvas id="viz-cnn-pool-input"></canvas>
                <div class="viz-cnn-sub">each 2×2 block becomes one output cell</div>
            </div>
            <div class="viz-cnn-pool-arrow" aria-hidden="true">→</div>
            <div class="viz-cnn-panel">
                <div class="viz-cnn-label">Pooled · 11×11</div>
                <canvas id="viz-cnn-pool-output"></canvas>
                <div class="viz-cnn-sub">half the spatial size, same channels</div>
            </div>
        </div>
    </div>
</div>
<script src="{{root}}js/viz/cnn.js"></script>

<article class="tldr-body" markdown="1">

### Why share weights across the image

A dense network treats every pixel as an independent feature: a 224×224 RGB image becomes a 150,528-dim vector, and a single hidden unit with 1,024 neurons already burns 150 million parameters before you've learned anything. Worse, the network has to relearn "an edge looks like an edge" separately at every position — it has no built-in notion that the top-left and bottom-right of an image play by the same rules.

A CNN replaces that with a tiny 3×3 (or 5×5) kernel that is reused at every spatial location. Two consequences fall out for free:

- **Parameter count collapses.** A 3×3 conv with 64 output channels has 9·C<sub>in</sub>·64 weights regardless of image size. You can now train a deep network on modest data.
- **Translation equivariance.** Shift the input by one pixel and the feature map shifts by one pixel. The model doesn't need separate evidence for the same pattern at different positions.

### Why pool / downsample

A single conv layer only sees a 3×3 patch. To recognise a face you need to integrate information from hundreds of pixels. The classic trick is **pooling** — replace each 2×2 block of the feature map with its max (or mean), halving the spatial dimensions. After a few rounds, what was a 224×224 image is a 7×7 grid of high-level features, each summarising a chunk of the original. Modern designs often skip pooling and use **strided convolutions** (jump 2 pixels at a time) for the same effect, with the benefit that the downsampling step is itself learned.

### Why stack convolutions

A single 3×3 kernel can only see three pixels in any direction. But two stacked 3×3 convs see a 5×5 patch, three see a 7×7, and so on — the *receptive field* grows with depth. Combine that with downsampling and a deep CNN's top layers see the entire image, but built compositionally: edges from layer 1 → corners and curves from layer 2 → textures and parts from layer 3 → object-shaped responses near the top. This hierarchy is the reason CNNs work, and the reason their feature maps are interpretable in a way few other architectures are.

→ For the receptive-field formula and dilated convs, switch to the In-depth tier.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Image classification, detection, segmentation — pretrained ResNet / EfficientNet / ConvNeXt backbones are a phone call away
- Audio spectrograms, medical scans, satellite imagery — anything on a 2D grid with local structure
- You have limited data and need the locality / translation-equivariance prior to do real work
- On-device or low-latency inference — small CNNs are still hard to beat for FLOPs-per-accuracy

</div>
<div class="no" markdown="1">

### Skip it when

- Text or sequences — use a transformer or RNN
- Graph-structured data — use a GNN
- The important relationships are long-range and don't compose locally
- You have tens of millions of labelled images and the compute for a ViT — it will likely edge ahead

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, n_classes),
)
```

</div>

<div class="level-next">
    <span>Want the convolution math, padding/stride, and ResNets?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">2D convolution</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ (\mathbf{x} \ast \mathbf{k})[i, j] \;=\; \sum_{m, n} \mathbf{x}[i+m,\, j+n] \cdot \mathbf{k}[m, n] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>x</code>input feature map (or image channel)</li>
<li markdown="1"><code>k</code>kernel — small (e.g. 3×3 or 5×5), *learned*</li>
<li markdown="1"><code>(i, j)</code>output spatial position; the kernel slides over every such position</li>
<li markdown="1">Output is "how strongly the filter's pattern appears at each position"</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{output}[i, j] \;=\; \text{sum over the kernel window of (image pixel} \times \text{kernel weight)} $$</span>

**In words.** At every output position `(i, j)`, line up the small **kernel** (a 3×3 or 5×5 grid of learned numbers) over a matching patch of the input. Multiply each kernel weight by the pixel it sits on, add all those products together — that's one output number. Slide the kernel one pixel over and repeat. The `∗` symbol means "convolution"; the `Σ` just means "add up everything inside the kernel window". The same kernel is reused at every position — that's why CNNs have so few parameters.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Anatomy of a conv layer.** Each layer learns *F* filters, each of shape `(C_in, k, k)`, applied at every spatial position. Input shape `(B, C_in, H, W)` becomes `(B, F, H', W')`. Three knobs determine `H'` and `W'`:

- **Padding** — pixels of zeros added around the input border. With `padding=1` and a 3×3 kernel, the spatial dimensions are preserved; without padding, every conv shrinks the map by 2.
- **Stride** — how many input pixels the kernel jumps between output positions. Stride 2 halves the spatial dimensions and is the modern replacement for pooling.
- **Dilation** — inserts gaps between kernel taps, so a 3×3 dilated-by-2 kernel covers a 5×5 patch with the same 9 weights. Used in segmentation networks to grow the receptive field without losing resolution.

**Pooling vs strided convolution.** Max-pooling (or average-pooling) is parameter-free and provides a hard form of translation invariance over its window. Strided convolutions are *learned* downsampling and tend to preserve more information; most modern architectures (ResNet, ConvNeXt, EfficientNet) use them in preference to pooling, keeping a single global average pool at the very end before the classifier head.

**ResNet (He et al., 2015).** Naively stacking 50+ conv layers used to *hurt* — training diverged, gradients vanished. ResNet added a **skip connection** around each block: `y = x + F(x)`. Now the layer only has to learn the *residual* on top of the identity, and gradients have a direct highway back to early layers. This single change unlocked networks 10× deeper than what had been possible and is now standard in essentially every modern architecture, vision or otherwise.

**Normalization.** Batch normalization is the classical default — normalize activations across the batch, learn a per-channel scale and shift. It speeds up training and acts as mild regularization, but it breaks with small batches (detection, segmentation, video). **Group norm** and **layer norm** are batch-independent and have largely taken over in modern recipes. The norm-vs-activation order (pre-norm vs post-norm) matters for training stability — pre-norm is the safer default.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Image classification — start from a pretrained ResNet / EfficientNet / ConvNeXt and fine-tune the head
- Object detection — Faster R-CNN, YOLO, RetinaNet all use CNN backbones
- Semantic / instance segmentation — U-Net for medical, Mask R-CNN for natural images
- Audio, video, medical imaging where labelled data is scarce but a pretrained backbone exists

</div>
<div class="no" markdown="1">

### Skip it when

- ImageNet-scale datasets and you have the compute budget — Vision Transformers may pull ahead
- The task needs reasoning across the *whole* image at every layer (use attention)
- Inputs aren't really grid-structured (sets, graphs, irregularly-sampled point clouds)
- You need symbolic / discrete-token reasoning

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn
import torchvision.models as tvm

# Transfer learning: load a pretrained ResNet, replace the final layer
model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, n_classes)

# Freeze backbone for the first few epochs, then unfreeze and fine-tune
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
```

</div>

<div class="level-next">
    <span>Want receptive fields, depthwise-separable, and ViT vs CNN?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Effective receptive field</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \text{RF}^{(\ell)} \;=\; \text{RF}^{(\ell-1)} \;+\; \big(k^{(\ell)} - 1\big)\,\prod_{j<\ell} s^{(j)} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>k<sup>(ℓ)</sup></code>kernel size at layer *ℓ*</li>
<li markdown="1"><code>s<sup>(j)</sup></code>stride at layer *j*</li>
<li markdown="1"><code>∏</code>product over all earlier layers' strides</li>
<li markdown="1">The patch of input that influences one output activation grows with depth</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{receptive field at layer } \ell \;=\; \text{receptive field at } \ell{-}1 \;+\; (\text{kernel size} - 1) \times \text{product of all earlier strides} $$</span>

**In words.** The **receptive field** is the size of the input patch that influences one pixel of a deeper feature map. Each new layer's kernel sees a window of its input, but the input has already been downsampled by previous strides — so its window covers a larger region of the *original* image. The `∏` (capital pi) means "multiply together" — here, multiply all the strides from earlier layers. A stride-2 layer means each subsequent pixel covers a 2× wider piece of the original; a 3×3 kernel after that sees a 6-pixel-wide patch of the input.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Effective receptive field.** The theoretical formula above is an upper bound. In practice the *effective* receptive field is roughly Gaussian-shaped and much smaller — most of the gradient mass concentrates near the centre. **Dilated (atrous) convolutions** enlarge the receptive field without growing parameters or losing resolution, which is why they show up in semantic segmentation (DeepLab) and dense prediction generally.

**Depthwise-separable convolutions.** A standard 3×3 conv with `C_in → C_out` channels costs `9 · C_in · C_out` parameters. Decompose it into a 3×3 **depthwise** conv (one filter per input channel, no channel mixing) followed by a 1×1 **pointwise** conv (mixes channels at every spatial position), and the cost drops to `9 · C_in + C_in · C_out` — often a 5-10× reduction. This is the trick behind MobileNet, EfficientNet, and ConvNeXt; the small accuracy loss is more than paid back by the FLOPs saved.

**Vision Transformers (ViT) vs CNNs.** A ViT slices the image into 16×16 patches, embeds each patch, and runs them through a stack of self-attention layers. It throws away the CNN's locality and translation-equivariance priors — and so on small data it *underperforms* CNNs badly. On hundreds of millions of labelled images (JFT-300M, LAION) it overtakes them: with enough data, learning the priors is better than hard-coding them. The modern picture is more nuanced — **ConvNeXt** (Liu et al., 2022) showed that a CNN modernised with depthwise convs, LayerNorm, GELU, and a ViT-style training recipe matches ViT performance on ImageNet, suggesting the gap was mostly about training tricks. Hybrids like **CoAtNet** and **MaxViT** combine convs in early stages with attention in later stages and frequently top the leaderboards.

**Attention-augmented convolutions.** Even before full ViTs, people grafted self-attention onto CNN backbones to inject global context — squeeze-and-excitation (channel-wise gating), non-local blocks, axial attention. They're still useful when you want CNN efficiency with the ability to do some long-range reasoning.

**Inductive bias as a feature.** CNNs hard-code three priors: **locality** (kernels are small), **translation equivariance** (the same kernel everywhere), and **hierarchical composition** (depth + downsampling). These priors are *correct* for natural images, which is why CNNs sample-efficiently learn from small datasets. They are *constraints* you'd want to relax once data is no longer the bottleneck. The choice between CNN and ViT is mostly a question of how much data you can throw at the problem.

**Equivariance beyond translation.** For rotation, reflection, or scale equivariance, see **group-equivariant CNNs** (Cohen & Welling, 2016) and **steerable CNNs** — important in molecular property prediction, astronomy, and medical imaging where the orientation of an object carries no information.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- On-device / low-latency inference — depthwise-separable backbones still rule the FLOPs-per-accuracy frontier
- Limited data — CNN priors do real work, ViTs need orders of magnitude more
- Dense prediction (segmentation, depth, optical flow) — receptive-field engineering matters and is well-understood
- Domains with clear locality and translation invariance — natural images, medical scans, spectrograms

</div>
<div class="no" markdown="1">

### Skip it when

- You need genuinely long-range reasoning the receptive field can't reach in a few layers
- You want a single architecture across modalities — transformers generalise more naturally
- You have ImageNet-scale data and a TPU pod budget — a well-trained ViT or hybrid usually wins
- The task isn't equivariant in any obvious sense (text, tabular, symbolic)

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn

# ConvNeXt-style block: depthwise + LayerNorm + inverted bottleneck + residual
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm   = nn.LayerNorm(dim, eps=1e-6)
        self.pw1    = nn.Linear(dim, expand * dim)
        self.act    = nn.GELU()
        self.pw2    = nn.Linear(expand * dim, dim)
    def forward(self, x):
        residual = x
        x = self.dwconv(x)                          # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)                   # → (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)                   # back to (B, C, H, W)
        return residual + x
```

</div>

<div class="level-next">
    <span>Too dense?</span>
    <button data-go-to="fundamentals" type="button">← Back to Fundamentals</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="cnn" markdown="1">

### What the figure shows

**Top — a single 3×3 kernel** (9 numbers) slides over the whole image; at every position it computes a weighted sum that becomes one cell of the feature map. *Use a preset filter* runs hand-designed kernels (Sobel edges, blur, sharpen); click any cell to cycle its value. *Learn the filter* starts from 9 random numbers and runs gradient descent toward a chosen target output — the weights aren't designed, they're learned from a loss signal, exactly like every other weight in a network.

**Bottom — pooling** downsamples the feature map: each 2×2 block becomes one output cell (max or mean). Same channels, half the spatial size. Stack conv + pool a few times and what was a 200×200 image becomes a 7×7 grid of high-level features. Hover either side of the pool to see which input block becomes which output cell.

The same kernel applies at every position — that's **weight sharing**, the reason CNNs need so few parameters compared to a dense net.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[CNN Explainer — poloclub](https://poloclub.github.io/cnn-explainer/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">A real CNN running in your browser. Click any feature-map cell to trace exactly which input pixels produced it. The single best interactive resource for seeing what each layer does.</span>
</li>
<li data-tier="intuition" markdown="1">
[Harley — 3D Visualization of an MNIST CNN](https://adamharley.com/nn_vis/cnn/3d.html) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Rotate, zoom, and inspect the volume of activations at every layer of a small CNN classifying your own hand-drawn digits. Older but still uniquely good for the spatial structure.</span>
</li>
<li data-tier="intuition" markdown="1">
[Olah et al. — Feature Visualization (Distill)](https://distill.pub/2017/feature-visualization/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The seminal article on what individual CNN neurons "see" — synthesised images that maximally activate each filter. Beautiful interactive diagrams; pairs well with the follow-up *Building Blocks of Interpretability*.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Zhang et al. — Dive into Deep Learning, CNN chapters](https://d2l.ai/chapter_convolutional-neural-networks/index.html) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Code-first walkthrough of convolutions, padding/stride, pooling, LeNet, AlexNet, VGG, GoogLeNet, ResNet, DenseNet — every classical architecture rebuilt from scratch with runnable notebooks.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[CS231n — Stanford](https://cs231n.github.io/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The gold-standard course on CNNs for vision. The lecture notes alone are worth working through even without the videos.</span>
</li>
<li data-tier="indepth" markdown="1">
[He et al. (2015) — Deep Residual Learning](https://arxiv.org/abs/1512.03385) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The ResNet paper. Skip connections — one of the most consequential ideas in modern deep learning. Short, readable, and the experiments speak for themselves.</span>
</li>
<li data-tier="indepth" markdown="1">
[Araujo et al. — Computing Receptive Fields (Distill)](https://distill.pub/2019/computing-receptive-fields/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Interactive walkthrough of how receptive fields actually grow with depth, dilation, and stride — including the often-surprising *effective* receptive field. Essential for designing dense-prediction networks.</span>
</li>
<li data-tier="indepth" markdown="1">
[Liu et al. (2022) — A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545) <i class="fas fa-external-link-alt"></i>
<span class="annotation">A CNN rebuilt with modern conventions (LayerNorm, depthwise convs, GELU, larger kernels). Matches ViT performance — useful proof that the CNN-vs-ViT gap was mostly about training recipe, not architecture.</span>
</li>
<li data-tier="indepth" markdown="1">
[Dosovitskiy et al. (2020) — An Image Is Worth 16×16 Words (ViT)](https://arxiv.org/abs/2010.11929) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The Vision Transformer paper — read this to understand what CNNs are being compared against, and why scale changes the answer.</span>
</li>
</ul>

</div>
</content>
</invoke>
---
title: Neural Networks — ML Resources Hub
eyebrow_text: ← Theory · Model Families
eyebrow_href: {{root}}theory.html
heading: Neural Networks
lead: Stacked layers of weighted sums and nonlinearities — the foundation that everything below builds on.
next_href: neural-networks/cnn.html
next_title: Convolutional Neural Networks
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**Stack simple operations until something complex emerges.** Like neurons in the brain, each layer does something simple — a weighted sum of its inputs, then a nonlinear bend. Stack enough of these and you can approximate just about any function: image classification, language translation, game playing.
</div>

<div class="viz-embed viz-dnn">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            Predict <strong>add</strong> or <strong>skip</strong> for 4 songs from <code>tempo</code> and <code>mood</code> (centred around 0). Ŷ is the prediction, <em>target</em> is the truth, ✓ marks a match. The targets are XOR-like — toggle activation to <strong>none</strong> and watch every prediction collapse.
        </span>
    </div>
    <div class="viz-embed-header">
        <span class="viz-embed-title">A small 2-layer network — drag any weight, click a column to peek inside a neuron</span>
        <div class="viz-embed-controls">
            <span class="viz-dnn-label">Activation:</span>
            <label><input type="radio" name="viz-dnn-act" value="none"> none</label>
            <label><input type="radio" name="viz-dnn-act" value="relu" checked> ReLU</label>
            <label><input type="radio" name="viz-dnn-act" value="sigmoid"> sigmoid</label>
            <label style="margin-left: 1rem;"><input type="checkbox" id="viz-dnn-numbers"> show numbers</label>
        </div>
    </div>
    <div class="viz-dnn-canvas-wrap">
        <canvas id="viz-dnn-canvas"></canvas>
    </div>
    <div class="viz-dnn-arch">
        <span class="viz-dnn-label">Hidden neurons:</span>
        <button id="viz-dnn-n-minus" type="button">−</button>
        <span id="viz-dnn-ncount" style="min-width: 1.5em; text-align: center;">3</span>
        <button id="viz-dnn-n-plus" type="button">+</button>
        <button id="viz-dnn-randomize" type="button" style="margin-left: 1rem;">randomize weights</button>
    </div>
</div>
<script src="{{root}}js/viz/dnn.js"></script>

<article class="tldr-body" markdown="1">

### Why we need activation functions

Try toggling activation to **none** in the figure. Every prediction collapses — and that's the point. Without a nonlinearity between layers, two matrix multiplications compose into a single matrix multiplication: `W₂ · (W₁ · X) = (W₂ · W₁) · X`. Stack 100 layers of pure matmuls and you still have exactly the expressive power of one. The activation (ReLU, sigmoid, GELU, …) breaks that collapse — each layer can bend the representation in a new way, and only then does *deep* mean anything.

### How to choose an architecture

Two main knobs: **width** (neurons per layer) and **depth** (how many layers).

- **Wider** layers carry more parallel features at once — useful when many independent things matter.
- **Deeper** stacks build features compositionally — later layers reuse what earlier ones discovered.

For tabular problems, a few hidden layers of 64–256 units is usually plenty. For images, sequences, or graphs, the architecture itself encodes the data's structure — reach for a [CNN]({{root}}topics/neural-networks/cnn.html), [RNN]({{root}}topics/neural-networks/rnn.html), [transformer]({{root}}topics/neural-networks/transformer.html), or [GNN]({{root}}topics/neural-networks/gnn.html). In practice the hardest knobs are rarely depth or width — it's getting the input scale, regularisation, and learning rate right.

### Why going deep works

Each layer can learn an abstraction built from the previous layer's features. In an image classifier: layer 1 picks up edges, layer 2 corners, layer 3 textures, layer 4 object parts, layer 5 whole objects. In language: characters → words → phrases → meaning. The **universal approximation theorem** (switch to the In-depth tier for the formal version) says one sufficiently wide hidden layer can in principle approximate any continuous function — but in practice **deep narrow networks generalise far better** than shallow wide ones with the same parameter count, because hierarchical composition lets the network *reuse* intermediate features instead of memorising every input combination separately.

### Training: how the weights got there

The weights in the figure didn't appear by magic — they were learned. You define a **loss** (how wrong the predictions are), compute the gradient of the loss with respect to every weight using the **chain rule** (this procedure is called *backpropagation*), and nudge each weight a tiny step in the direction that lowers the loss. Repeat for millions of steps. The figure freezes one moment in that process; training is what carved the colours you see.

→ Full mechanics in [Gradient Descent]({{root}}topics/gradient-descent.html) and [Optimisation Algorithms]({{root}}topics/optimization-algorithms.html).

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Tabular data and you want more flexibility than a linear model
- You have plenty of data and compute
- You'll use this as a building block in a bigger model
- You want a starting point before reaching for CNNs / transformers

</div>
<div class="no" markdown="1">

### Skip it when

- Small tabular data — gradient boosting usually wins
- Data has obvious structure (images, sequences, graphs) — use a specialized architecture
- You need interpretability of individual predictions
- You're short on training data

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(n_features, 128), nn.ReLU(),
    nn.Linear(128, 64),          nn.ReLU(),
    nn.Linear(64, n_classes),
)
```

</div>

<div class="level-next">
    <span>Want the forward / backward math?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Forward pass</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \mathbf{h}^{(\ell)} \;=\; \sigma\!\left(W^{(\ell)} \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>h<sup>(ℓ)</sup></code>activations (outputs) of layer *ℓ*, with <code>h<sup>(0)</sup> = x</code></li>
<li markdown="1"><code>W<sup>(ℓ)</sup>, b<sup>(ℓ)</sup></code>weight matrix and bias vector of layer *ℓ*</li>
<li markdown="1"><code>σ</code>elementwise nonlinearity — ReLU, GELU, tanh, …</li>
<li markdown="1">Apply this rule from *ℓ = 1* up to the final layer to get the network's output</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{layer output} \;=\; \sigma\!\left(\text{weights} \times \text{previous layer output} + \text{bias}\right) $$</span>

**In words.** Each layer does three things in order: multiply the previous layer's output (a list of numbers) by a matrix of **weights**, add a **bias** vector to shift the result, then run each number through a **nonlinearity** `σ` (sigma — usually ReLU, which is just `max(0, x)`). You repeat this for every layer, plugging each layer's output into the next. The superscript `(ℓ)` in the math version is just a layer label — "weights for layer 1", "weights for layer 2", and so on; they're different matrices, learned independently.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Training.** Define a loss *ℒ(θ)*, take its gradient with respect to all parameters via backpropagation (chain rule applied layer-by-layer), and step in the negative-gradient direction. SGD, Adam, AdamW are the optimizers people actually use.

**Activations.** ReLU is the default — cheap and avoids the saturating-gradient problem of tanh / sigmoid. GELU is smoother and dominates in transformers. Sigmoid / tanh appear only inside gates (LSTM, attention) where bounded outputs matter.

**Initialization.** Random Gaussian weights need to be scaled carefully — too big and activations blow up, too small and they vanish. Xavier/Glorot for tanh, He/Kaiming for ReLU. Most frameworks default to a sensible scheme.

**Regularization.** A network with enough parameters can memorize the training set perfectly — and then fail on new data. *Regularization* is anything that pushes the model away from rote memorization and toward solutions that generalize. The standard toolkit:

- **Dropout** — randomly zero a fraction of activations during each training step so no single neuron becomes indispensable; equivalent to averaging an ensemble of thinned subnetworks.
- **Weight decay (L2)** — add a penalty proportional to `‖W‖²` to the loss; shrinks weights toward zero, which limits how sharply the function can bend.
- **Early stopping** — hold out a validation set and stop training the moment validation loss turns back up.
- **Data augmentation** — perturb each input (flip, crop, noise, paraphrase) so the model has to learn invariant features rather than memorize examples.

Modern nets often need less *explicit* regularization than older ones — the implicit regularization of SGD itself (its preference for flat, broad minima) does a surprising amount of the work.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- The data has no obvious structural prior — let a deep MLP discover features
- You can apply the modern training recipe — ReLU/GELU, He init, Adam, dropout, weight decay
- You're stacking many layers — residual connections + LayerNorm let you go deep without vanishing gradients
- You'll embed it as a head or block inside a larger architecture (CNN, transformer, …)

</div>
<div class="no" markdown="1">

### Skip it when

- The data has structure you can exploit — use it (convolutions for images, attention for sequences)
- You need calibrated probabilities — neural nets are over-confident without post-hoc calibration
- Embedded / microcontroller-class hardware with no SIMD or GPU and tight latency
- You need step-by-step decision traces — neural nets are opaque inside

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
from torch.optim import AdamW

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

model = MLP(d_in=20, d_out=10)
opt   = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
```

</div>

<div class="level-next">
    <span>Want depth-vs-width, universal approximation, and the failure modes?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Universal approximation</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \forall \,\varepsilon{>}0,\; \exists\, n,\, \{w_i, b_i, c_i\}: \quad \sup_{\mathbf{x}\in K} \left|\, f(\mathbf{x}) - \sum_{i=1}^{n} c_i\, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)\,\right| < \varepsilon $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>∀ ε &gt; 0</code>"for any error tolerance you pick, however tiny"</li>
<li markdown="1"><code>∃ n, {w<sub>i</sub>, b<sub>i</sub>, c<sub>i</sub>}</code>"there exists some number of neurons and a set of weights/biases/output-weights"</li>
<li markdown="1"><code>sup<sub>x ∈ K</sub></code>worst-case error over a bounded input region *K*</li>
<li markdown="1"><code>Σ</code>summation — one term per hidden neuron</li>
<li markdown="1">A single hidden layer with enough units can approximate any continuous *f* on a compact set. Depth makes "enough" tractable.</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{for any error tolerance } \varepsilon \;\Rightarrow\; \text{a 1-hidden-layer network with enough neurons gets within } \varepsilon \text{ of } f \text{ everywhere} $$</span>

**In words.** Pick any continuous function *f* you want to approximate, and any error budget *ε* (epsilon — Greek letter for a small positive number). The theorem guarantees you can find a single-hidden-layer network — with enough neurons — whose output never differs from *f* by more than *ε*, anywhere in a bounded input region. The catch: "enough neurons" can mean an astronomical number. In practice, going *deep* (many layers, fewer neurons each) is far more efficient than going *wide* (one huge layer) — that's why we stack.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Depth vs. width.** The universal approximation theorem says one hidden layer suffices. In practice, deep narrow networks generalize much better than shallow wide ones for the same parameter count — depth lets the model compose features hierarchically. Modern intuition: width gives capacity, depth gives *composition*.

**Vanishing / exploding gradients.** In deep networks, repeated multiplication of small or large Jacobian factors makes gradients shrink or explode through the chain rule. Modern fixes: ReLU (constant gradient on positive inputs), batch / layer normalization (keep activations on a sensible scale), and [residual connections](https://d2l.ai/chapter_convolutional-modern/resnet.html) — additive "gradient highways" that let the loss reach early layers undamaged.

**The optimization landscape** is non-convex with many local minima and saddle points. SGD finds *flat* minima that generalize well; sharp minima generalize poorly. This is part of why batch size and learning rate matter for generalization, not just for training speed.

**Dead ReLUs.** If a unit's pre-activation goes strongly negative and stays there, its gradient is zero and it never updates again. Mitigations: smaller learning rate, Leaky ReLU / GELU, careful initialization, batch norm.

**Double descent.** Past the interpolation threshold (where the network can perfectly memorize the training set), test error often *decreases* as you grow the model further. Connect to [bias-variance]({{root}}topics/bias-variance.html) — classical theory predicts the opposite. Amazon's [MLU-Explain — Double Descent](https://mlu-explain.github.io/double-descent/) has a beautiful interactive walkthrough showing the curve form as model size grows.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- You can afford to scale (more data, more parameters)
- End-to-end differentiability matters — embed in any pipeline
- You have a budget for hyperparameter search (LR, depth, width, regularization)
- Pre-trained embeddings exist for your domain

</div>
<div class="no" markdown="1">

### Skip it when

- Very small data — gradient boosting and probabilistic models win
- You can't engineer reasonable hyperparameters and don't want to AutoML them
- Adversarial robustness is a hard requirement
- Causal inference / counterfactual reasoning is the goal

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn

# Residual MLP block — mitigates vanishing gradients, enables deeper models
class ResBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc   = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
    def forward(self, x):
        return x + self.fc(self.norm(x))

# A "modern" deep MLP: pre-norm + residual + GELU expansion (the MLP-Mixer pattern)
class DeepMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=384, n_blocks=6):
        super().__init__()
        self.embed  = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.head   = nn.Linear(hidden, d_out)
    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks: x = blk(x)
        return self.head(x)
```

</div>

<div class="level-next">
    <span>Want the picture instead?</span>
    <button data-go-to="intuition" type="button">← Back to Intuition</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="neuron" markdown="1">

### Inside one neuron

The smallest unit of a neural network. The two **circles on the left** are inputs — for one specific song, they hold `tempo` and `mood`. The **edges** in between carry the weights: one number per edge, coloured by sign (indigo +, orange −) and weighted by magnitude (thicker = larger).

Each input is multiplied by its weight, the results are summed with a bias, and the activation function (ReLU here) bends the sum into the output you see in the **right circle**.

The figure traces one example song through one of the network's hidden neurons. Click any column of W₁ in the figure below to switch to a different neuron.

</div>

<div class="fig-explainer" data-fig="main" markdown="1">

### The whole network at once

Each cube is **one matrix multiplication**. The left face is the input rows, the top face holds the weight matrix, and the **front face is the result**. Reading left → right: songs go in, get matmul'd by W₁, pass through ReLU (the slab — negatives are absorbed in orange), get matmul'd by W₂, pass through sigmoid, and come out as a final add/skip prediction.

Drag any cell of W₁ or W₂ to change a weight — watch the downstream cells re-colour live. Click a column of W₁ to pick which neuron the figure above is tracing.

</div>

<div class="fig-explainer" data-fig="big" markdown="1">

### Same network, textbook style

The familiar **neurons-and-wires** diagram you'll see in most ML textbooks. Each column is one layer; each circle is a neuron; each line is one edge weight. The colour of the edge matches the corresponding **cell of the W matrix** above — every edge here is one cell in one of those mini-cubes at the bottom.

Read the mini-cubes as "W₁ is what the edges between the first two columns are", "W₂ for the next gap", and so on. Same numbers, two ways of drawing them — cube-style for the matrix machinery, circle-style for tracing signal flow.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[TensorFlow Playground](https://playground.tensorflow.org/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Interactive in-browser net you can train live. Toggle activations, change architecture, and watch the decision boundary form in real time — the canonical hands-on intuition tool.</span>
</li>
<li data-tier="intuition" markdown="1">
[Nielsen — Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Best free intro for understanding backprop from scratch with worked examples and interactive demos.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Zhang et al. — Dive into Deep Learning](https://d2l.ai/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Interactive, code-first textbook with runnable PyTorch / JAX / MXNet notebooks alongside the maths. Best balance of theory and hands-on practice.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Goodfellow, Bengio, Courville — Deep Learning](https://www.deeplearningbook.org/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The textbook. Free HTML version. Chapters 6–8 cover MLPs, optimization, and regularization in depth.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[PyTorch tutorials](https://pytorch.org/tutorials/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The clearest introduction to autograd, optimizers, and the training loop. Hands-on with code.</span>
</li>
<li data-tier="indepth" markdown="1">
[Karpathy — A Recipe for Training NNs](https://karpathy.github.io/2019/04/25/recipe/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Hard-won practical advice on what actually goes wrong when training networks. Read this before debugging anything.</span>
</li>
</ul>

</div>

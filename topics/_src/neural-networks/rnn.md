---
title: Recurrent Neural Networks (RNN) — ML Resources Hub
eyebrow_text: ← Theory · Neural Networks
eyebrow_href: {{root}}theory.html
heading: Recurrent Neural Networks
lead: Networks with an internal state that walks through a sequence one step at a time — LSTMs, GRUs, and the long-standing default for language before transformers.
prev_href: cnn.html
prev_title: Convolutional Neural Networks
next_href: transformer.html
next_title: Transformers
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**One small network, applied to every step of a sequence, passing a memory forward.** Read a token, update memory, move on. Read the next token, update memory again. After the whole sequence the memory summarises everything you've seen — and any prediction you make can read from it.
</div>

<div class="viz-embed viz-rnn" data-fig="rnn">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            Watch text turn into numbers (tokenise → vocab id → embedding) then flow through an RNN one step at a time. The middle panel shows the <strong>four shared weight matrices</strong> (E, W_x, W_h, b) — the entire network. The same matrices are applied at every timestep below: indigo arrows go through W_x, orange arrows through W_h. Try a long sequence on a <em>Vanilla RNN</em> and watch the early signal fade; switch to <em>LSTM-like</em> and memory survives much further.
        </span>
    </div>
    <div class="viz-embed-header">
        <span class="viz-embed-title">The same RNN cell at every timestep — click any step to inspect it</span>
    </div>
    <div class="viz-rnn-controls">
        <label style="display: inline-flex; align-items: center; gap: 0.4rem; color: var(--muted); font-size: 0.85rem;">
            Sequence
            <select id="viz-rnn-seq"></select>
        </label>
        <label style="display: inline-flex; align-items: center; gap: 0.4rem; color: var(--muted); font-size: 0.85rem;">
            Cell
            <select id="viz-rnn-cell"></select>
        </label>
        <button id="viz-rnn-play" type="button">Pause</button>
    </div>
    <div class="viz-rnn-canvas-wrap">
        <canvas id="viz-rnn-canvas"></canvas>
    </div>
    <div class="viz-rnn-formula" id="viz-rnn-formula"></div>
</div>
<script src="{{root}}js/viz/rnn.js"></script>

<article class="tldr-body" markdown="1">

### Why a hidden state

A plain feedforward net sees one fixed-size input and emits one output — it has no notion of "before" or "after". For sequences (sentences, audio frames, sensor readings) the meaning of token *t* depends on everything that came before it. An RNN solves this by carrying a **hidden state** — a vector that gets updated at every step. The same weights are reused at every timestep, so the network is effectively a tiny program running in a loop, with the hidden state acting as its working memory.

### Why RNNs are mostly replaced now

Two problems killed them in NLP. First, **vanishing gradients**: when you backpropagate through hundreds of timesteps, gradients are multiplied by the same matrix over and over and decay to zero — the network can't learn long-range dependencies. Second, **no parallelism across timesteps**: each step has to wait for the previous one's hidden state, so you can't fill a GPU. Transformers sidestep both by replacing recurrence with attention, which sees every position at once.

### Where they still make sense

- **Streaming and real-time inference.** You process one step as it arrives — perfect for a microphone, a network packet stream, or a sensor.
- **Tiny models on tiny hardware.** A 100k-parameter LSTM fits on a microcontroller; a transformer of comparable quality does not.
- **Bounded memory.** The hidden state has fixed size regardless of sequence length. Attention is O(N²) in sequence length; an RNN is O(N) with constant memory per step.
- **Online learning.** You can keep training on a never-ending stream without re-batching.

### LSTM and GRU, intuitively

A vanilla RNN overwrites its memory every step — which is why it forgets. **LSTM** ([Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)) adds three small "gates" — *forget*, *input*, *output* — each a little sigmoid network that outputs values between 0 and 1, acting like dimmer switches on the memory. The cell decides *how much* of the old memory to keep, *how much* new content to write in, and *how much* of the result to expose. Crucially, the memory has its own additive update path, so gradients survive long sequences. **GRU** is the same idea trimmed down to two gates and one state — usually as good, faster to train.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Streaming or real-time data — one step in, one step out
- Edge / on-device inference where a transformer is too heavy
- Short-to-medium sequences with strong temporal structure
- Small data — LSTMs overfit less than transformers in this regime
- You're prototyping against a Mamba / state-space model and want a baseline

</div>
<div class="no" markdown="1">

### Skip it when

- Long sequences with non-local dependencies — attention wins
- You have GPU compute and want to train fast — RNNs don't parallelise across time
- Pretrained transformer weights exist for your domain (almost always do in NLP / vision)
- You need to attend to specific positions — implicit memory is worse than explicit attention

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

# LSTM with 2 layers, hidden size 128
model = nn.LSTM(
    input_size=10,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
)

# Inputs: (B, T, input_size); outputs: (B, T, hidden_size) + final (h, c)
outputs, (h_final, c_final) = model(inputs)
```

</div>

<div class="level-next">
    <span>Want the recursion math and the LSTM gate equations?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Vanilla RNN recursion</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \mathbf{h}_t \;=\; \tanh\!\left(W_{xh}\, \mathbf{x}_t + W_{hh}\, \mathbf{h}_{t-1} + \mathbf{b}_h\right) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>h<sub>t</sub></code>hidden state at time *t* — the network's memory so far</li>
<li markdown="1"><code>x<sub>t</sub></code>input at time *t* (e.g. a token embedding)</li>
<li markdown="1"><code>W<sub>xh</sub>, W<sub>hh</sub></code>input-to-hidden and hidden-to-hidden weights, *shared across timesteps*</li>
<li markdown="1"><code>tanh</code>S-shaped nonlinearity, bounded to [−1, 1], that keeps the state from blowing up</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{memory}_t \;=\; \tanh\!\left(\text{input-weights} \times \text{input}_t \;+\; \text{memory-weights} \times \text{memory}_{t-1} \;+\; \text{bias}\right) $$</span>

**In words.** At every timestep the RNN mixes two things: the **current input** (this token, this sensor reading, this frame) and the **previous memory** (the hidden state from one step ago). Each is run through its own weight matrix, the results are summed with a bias, and the whole thing is squashed through `tanh` so the memory stays bounded. The same weight matrices are reused at every timestep — that weight-sharing is what makes "RNN" different from a deep feed-forward network with *T* layers, and it's how the network can handle sequences of any length with a fixed parameter count.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Unrolling and BPTT.** To train an RNN you "unroll" it: write out the recursion for *T* steps as a deep feedforward graph of depth *T* with shared weights, then apply standard backprop. The result is called **Backpropagation Through Time**. Memory cost grows linearly with *T*; for long sequences you use *truncated BPTT* — backprop over a fixed window (say 200 steps), detach the hidden state, and keep going.

**Vanishing and exploding gradients.** Because BPTT multiplies by the recurrent Jacobian once per step, the gradient through *T* steps is roughly the same matrix raised to the *T*th power. Eigenvalues > 1 explode; eigenvalues < 1 vanish. Vanilla RNNs almost always vanish, which means signals more than ~20 steps back can't influence the loss. The standard fixes:

- **Gradient clipping** — rescale the gradient if its norm exceeds a threshold. Stops explosions cold.
- **Gated cells (LSTM, GRU)** — replace the multiplicative update with an additive one (see below).
- **Orthogonal / identity initialisation** of `W_hh` — keeps the Jacobian's spectral radius near 1 at start of training.

**LSTM ([Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)).** Adds three gates — forget, input, output — and a separate "cell state" `c_t` that travels along its own additive path: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`. The additive update is the key trick — gradients flow back through it without the multiplicative collapse.

**GRU ([Cho et al., 2014](https://arxiv.org/abs/1406.1078)).** Two gates, one state. Comparable accuracy to LSTM on most tasks, ~25% fewer parameters, faster to train.

**Bidirectional RNNs.** Run one RNN left-to-right and another right-to-left; concatenate their hidden states. Captures both past and future context — only valid when you can see the whole sequence (translation, NER), not in streaming settings.

**Sequence-to-sequence.** One RNN as encoder (read the input, produce a final hidden state), another as decoder (start from that hidden state, generate the output token by token). The bottleneck of compressing everything into one final hidden state is what motivated **attention** — which eventually swallowed the RNN entirely.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Time-series forecasting with limited compute and modest sequence length
- Online / streaming classification — one step at a time, constant memory
- Small-data sequence tasks where a transformer overfits
- Edge inference — a quantised LSTM is a fraction the size of any transformer

</div>
<div class="no" markdown="1">

### Skip it when

- You can afford to parallelise across time — a transformer trains an order of magnitude faster
- Sequences are long with non-local structure (>1k tokens, scattered references)
- Pretrained transformer encoders exist for your domain
- Your loss curve plateaus quickly — likely a vanishing-gradient symptom on a vanilla RNN

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn

class SeqClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                             dropout=0.3, bidirectional=True, batch_first=True)
        self.head  = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        # Concat the last hidden states from both directions of the top layer
        last = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(last)
```

</div>

<div class="level-next">
    <span>Want the full LSTM cell, BPTT cost, and the state-space comeback?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">LSTM cell</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \begin{aligned}
\mathbf{f}_t &= \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \\
\mathbf{i}_t &= \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \\
\mathbf{o}_t &= \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \\
\tilde{\mathbf{c}}_t &= \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>f, i, o</code>forget, input, output gates — each a sigmoid output in [0, 1]</li>
<li markdown="1"><code>c<sub>t</sub></code>cell state — long-term memory travelling along the additive path</li>
<li markdown="1"><code>h<sub>t</sub></code>hidden state — the short-term, gated output of the cell</li>
<li markdown="1"><code>⊙</code>elementwise (Hadamard) product</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \begin{aligned}
\text{forget gate} &= \sigma(\ldots) \quad \text{(how much old memory to keep, 0–1)} \\
\text{input gate}  &= \sigma(\ldots) \quad \text{(how much new info to write, 0–1)} \\
\text{output gate} &= \sigma(\ldots) \quad \text{(how much memory to expose, 0–1)} \\
\text{candidate}   &= \tanh(\ldots) \quad \text{(proposed new content)} \\
\text{cell-memory} &= \text{forget gate} \times \text{old cell-memory} \;+\; \text{input gate} \times \text{candidate} \\
\text{hidden}      &= \text{output gate} \times \tanh(\text{cell-memory})
\end{aligned} $$</span>

**In words.** Three little gates — **forget**, **input**, **output** — each a small neural net followed by `σ` (sigmoid), which squashes their outputs into the range 0 to 1. They behave like dimmer switches. The **cell-memory** `c` is a separate "long-term memory" stream: the forget gate decides what fraction of it to keep, the input gate decides what fraction of a new **candidate** (a tanh-bent update) to write in, and the two are added together. The **hidden state** `h` is the network's per-step output, gated from the current cell-memory. The crucial trick is the cell-memory's *additive* update path — gradients flow back through it without the multiplicative collapse that kills vanilla RNNs.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Why transformers won.** Attention ([Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473)) started life as a fix for the seq2seq bottleneck — let the decoder peek at all encoder states with learned weights instead of compressing everything into one final hidden state. By 2017 ([Vaswani et al.](https://arxiv.org/abs/1706.03762)) the recurrence was gone entirely: attention is fully parallel across positions, has direct paths between any two tokens (no vanishing gradient), and scales beautifully on GPUs. On natural language at scale, this combination is decisive.

**The comeback: state-space models.** RNN-style sequential processing is enjoying a renaissance via **state-space models** — [S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752), and friends. They use carefully chosen *linear* recurrences (a structured matrix raised to the *t*th power) that admit a parallel scan algorithm — so you get RNN-style O(N) inference and a bounded state, *plus* parallel training. On long sequences (DNA, audio, video) they're competitive with attention at a fraction of the per-token cost. Mamba adds input-dependent dynamics ("selective" state updates) and matches small transformers on language modelling.

**Attention vs. recurrence, the trade-off.** Attention gives every position direct access to every other position — O(N²) compute, O(N) memory per layer. Recurrence compresses the past into a fixed-size state — O(N) compute, O(1) memory per step, but a bottleneck that limits how much past information can be retained. The right choice depends on whether your bottleneck is compute (favours recurrence), memory (favours recurrence), or long-range precision (favours attention).

**Where RNNs still genuinely win in 2026.** Keyword-spotting on a microcontroller. Continuous audio denoising on earbuds. Online anomaly detection on streaming sensors. Reinforcement-learning policies where each environment step costs a forward pass. Tiny on-device language models. Whenever your latency budget is "one token at a time" and your memory budget is kilobytes, a quantised LSTM still beats anything transformer-shaped.

**BPTT cost in detail.** Full BPTT over *T* steps stores all activations — O(T) memory. *Truncated BPTT* with window *k* uses O(k) memory but biases the gradient (you ignore dependencies > *k* steps back). *Gradient checkpointing* recomputes activations during the backward pass for O(√T) memory — useful when *T* is fixed but large.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Strict memory budget on very long sequences — state size is bounded, attention isn't
- Causal real-time inference where parallel decoding doesn't help
- You're prototyping with modern SSMs (Mamba, S5) and need a recurrent baseline
- Theoretical / interpretability work on dynamical systems

</div>
<div class="no" markdown="1">

### Skip it when

- You can parallelise — transformers train an order of magnitude faster
- A pretrained transformer encoder exists for your domain
- You need to attend to specific positions — explicit attention beats implicit memory
- Long-range global dependencies dominate the task

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn
import torch.nn.utils as utils

# Training loop with gradient clipping and truncated BPTT
model     = nn.LSTM(input_size=128, hidden_size=512, num_layers=2, batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
clip_norm = 1.0
truncate  = 200    # max BPTT length

for sequence in long_sequences:
    # Truncate into chunks of `truncate` steps, detach state between chunks
    state = None
    for chunk in sequence.split(truncate, dim=1):
        out, state = model(chunk, state)
        # Detach to cut the BPTT graph here:
        state = tuple(s.detach() for s in state)
        loss  = loss_fn(out, targets[chunk])
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
```

</div>

<div class="level-next">
    <span>Want the picture instead?</span>
    <button data-go-to="intuition" type="button">← Back to Intuition</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="rnn" markdown="1">

### Reading the figure

**Top — text → numbers.** Each word in the input string becomes a token, then a vocabulary id (an integer), then a vector by looking up row `id` of the **embedding matrix E**. That vector is the actual input the network sees at each step — strings never enter the linear algebra.

**Middle — shared weights.** Four matrices make up the entire network: `E` (embeddings), `W_x` (input weights), `W_h` (recurrent weights), `b` (bias). These same four matrices are applied at *every* timestep below. That weight-sharing is what makes an RNN an RNN — and what lets it process sequences of any length without growing the parameter count.

**Bottom — unrolled in time.** Each column is the same cell applied at a different step. Indigo arrows go through `W_x` (input projection); orange arrows go through `W_h` (recurrent projection). The cell formula is `h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)`. Click any column to inspect its concrete numbers, or try the *long sequence* with a *Vanilla RNN* and watch the early signal decay — that's the vanishing-gradient problem in pictures.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[Karpathy — The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Hands-on char-RNNs generating Shakespeare, LaTeX, C source. The classic that hooked a generation on sequence models. Read it for the demos, stay for the intuition.</span>
</li>
<li data-tier="intuition" markdown="1">
[Olah — Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The clearest visual explanation of LSTM gates ever written. If you remember nothing else about RNNs, remember these diagrams.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Zhang et al. — Dive into Deep Learning · RNN chapter](https://d2l.ai/chapter_recurrent-neural-networks/index.html) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Code-first walkthrough of vanilla RNNs, BPTT, gradient clipping, LSTM and GRU with runnable notebooks. Best balance of theory and practice.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Hochreiter & Schmidhuber (1997) — Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The original LSTM paper. Dense but the foundational source — read after Olah for the historical context and the original motivation.</span>
</li>
<li data-tier="indepth" markdown="1">
[Gu & Dao (2023) — Mamba](https://arxiv.org/abs/2312.00752) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The selective state-space model that revived RNN-style processing at scale. Includes the derivation of parallel training despite sequential semantics — the trick that makes SSMs competitive with attention.</span>
</li>
<li data-tier="indepth" markdown="1">
[Pascanu et al. (2013) — On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The definitive analysis of vanishing and exploding gradients in RNNs. Introduces gradient clipping and gives the spectral-radius intuition behind why long-range training is hard.</span>
</li>
</ul>

</div>
</content>
</invoke>
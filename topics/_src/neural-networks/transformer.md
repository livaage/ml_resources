---
title: Transformers — ML Resources Hub
eyebrow_text: ← Theory · Neural Networks
eyebrow_href: {{root}}theory.html
heading: Transformers
lead: Attention-based architectures — the backbone of modern LLMs, vision models, and almost everything else.
prev_href: rnn.html
prev_title: Recurrent Neural Networks
next_href: gnn.html
next_title: Graph Neural Networks
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**Every token looks at every other token, weighted by relevance.** No left-to-right bottleneck, no fixed window — each token computes a soft lookup over the whole sequence and pulls in whatever is most useful for its own update. That soft lookup is *attention*, and it turns out to be enough.
</div>

<div class="viz-embed viz-tx" data-fig="transformer">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            See how attention lets every token look at every other token. The heatmap shows the attention weights — for each <strong>query</strong> token (row), how much it attends to each <strong>key</strong> token (column). Pick a head to switch between learned patterns; click a token or any cell to make it the query.
        </span>
    </div>
    <div class="viz-embed-header">
        <span class="viz-embed-title">Pick a head · click any token to make it the query</span>
    </div>
    <div class="viz-tx-heads" id="viz-tx-heads"></div>
    <div class="viz-tx-canvas-wrap">
        <canvas id="viz-tx-canvas"></canvas>
    </div>
    <div class="viz-tx-desc" id="viz-tx-desc"></div>
</div>
<script src="{{root}}js/viz/transformer.js"></script>

<article class="tldr-body" markdown="1">

The four heads above show the kinds of patterns real transformers actually learn: simple positional shifts, plus richer linguistic relationships like adjective→noun and verb→subject. A real model has dozens of these per layer, all running in parallel, and the next layer can compose their outputs. That composition is most of what makes transformers work.

### Why attention beats recurrence

An RNN reads tokens one at a time, squeezing the past into a fixed hidden state. Two problems: the early tokens get crushed by everything that comes after (long-range information vanishes), and you can't parallelise the forward pass — token *t* needs token *t-1* to finish first. Attention sidesteps both. Every token sees every other token in **one matrix multiplication**, so the path length between any two positions is constant and the whole sequence runs on the GPU at once. You pay an O(N²) bill for the privilege, but for sequences up to tens of thousands of tokens it's worth it.

### What attention is doing intuitively

Think of a soft database lookup. Each token emits a **query** ("what am I looking for?"), every token also exposes a **key** ("here's what I am") and a **value** ("here's what I'd contribute"). The dot product `query · key` scores how well a query matches each key; softmax turns those scores into weights that sum to 1; the output is a weighted average of the values. Hard lookup would pick the single best match — attention picks all of them, gently, in proportion to how well they match. The whole thing is differentiable, so the model learns *what* to look for.

### Why multi-head

One attention pattern per layer would be a brutal bottleneck. Instead, split the embedding into *h* slices and run *h* independent attention computations in parallel — one head can track syntax, another coreference, another raw position, another semantic similarity. Concatenate the outputs and project back. Same FLOPs as a single big head, vastly more expressive.

### Why position encoding

Attention is permutation-invariant — without help, "the cat sat on the mat" and "the mat sat on the cat" produce identical outputs. So we inject position information into the token embeddings themselves: either added as a sinusoidal vector (original paper) or, in modern models, baked into the Q/K projections via a rotation (RoPE). Without it, the architecture has no idea what order anything is in.

</article>

<div class="viz-grid" markdown="1">
<a href="https://poloclub.github.io/transformer-explainer/" target="_blank" class="viz-card viz-card-featured">
<span class="viz-source">poloclub.github.io</span>
### Transformer Explainer
A real GPT-2 running in your browser. Type a prompt and watch every step — token embedding, positional encoding, multi-head attention (with every head visible separately), layer norms, the final softmax. The clearest visualisation of a working transformer ever made.
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>

<a href="https://bbycroft.net/llm" target="_blank" class="viz-card">
<span class="viz-source">bbycroft.net</span>
### LLM Visualization — Brendan Bycroft
Stunning 3D walkthrough of a tiny GPT, frame by frame. Walks you through every matrix multiplication, attention computation, and softmax in the entire forward pass. Best for understanding the dataflow at the tensor level.
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>

<a href="https://jalammar.github.io/illustrated-transformer/" target="_blank" class="viz-card">
<span class="viz-source">jalammar.github.io</span>
### The Illustrated Transformer
Jay Alammar's classic article. Static but unbeatable diagrams of every transformer component: Q/K/V projections, scaled dot-product attention, multi-head, encoder-decoder attention. Read this *before* attempting the original paper.
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>

<a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" class="viz-card">
<span class="viz-source">transformer-circuits.pub</span>
### Mathematical Framework — Anthropic
Not interactive but visually rich. For understanding what attention heads *actually* compute mechanistically — induction heads, copy circuits, position-tracking. Read after you understand the basics; rewards multiple re-reads.
<span class="viz-cta">Open viz <i class="fas fa-arrow-right"></i></span>
</a>
</div>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Language modelling, translation, summarisation — and a pretrained model exists (it does)
- Long-range dependencies in any sequence (text, code, protein, audio)
- Multimodal fusion — attention composes cleanly across modalities
- You want one architecture template you can reuse for many tasks

</div>
<div class="no" markdown="1">

### Skip it when

- Very small data and no pretrained checkpoint — gradient boosting or a small RNN wins
- Strict memory budget on long sequences — attention is quadratic; consider SSMs (Mamba) or linear attention
- True streaming with bounded state — recurrent variants are cheaper
- Highly local structure (small images, short audio frames) — a CNN may suffice

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
mdl = AutoModel.from_pretrained("bert-base-uncased")

inputs  = tok("Transformers see every token at once.", return_tensors="pt")
outputs = mdl(**inputs)
# outputs.last_hidden_state: (1, seq_len, 768) — contextualised embeddings
```

</div>

<div class="level-next">
    <span>Want the attention formula and architecture details?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Scaled dot-product attention</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \text{Attention}(Q, K, V) \;=\; \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>Q, K, V</code>queries, keys, values — each token projected to three vectors</li>
<li markdown="1"><code>QKᵀ</code>matrix of how well each query matches each key (dot products)</li>
<li markdown="1"><code>√d<sub>k</sub></code>scaling — prevents the softmax from saturating when the key dimension is large</li>
<li markdown="1"><code>softmax</code>turns the match scores into weights that sum to 1 across keys</li>
<li markdown="1">Row *i* of the output = weighted sum of values, weights from query *i*'s match to each key</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{output}_i \;=\; \sum_{j} w_{ij} \cdot \text{value}_j \quad\text{where}\quad w_{ij} \;=\; \text{softmax}_j\!\left(\frac{\text{query}_i \cdot \text{key}_j}{\sqrt{d}}\right) $$</span>

**In words.** For each token *i*: see how well its **query** vector matches every **key** vector — the dot product `q · k` is a single number measuring alignment (bigger when the two vectors point in similar directions). Divide by `√d` to keep the numbers from blowing up when the vectors are long. Run those scores through `softmax`, which squashes any list of numbers into positive weights that sum to 1 (so they behave like probabilities). The `Σ` (sigma) symbol just means "add up across every *j*" — so the final output is a weighted average of all the **value** vectors.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Q, K, V — where they come from.** Each token starts as an embedding `x` (a *d*-dimensional vector). Three learned projection matrices `W_Q, W_K, W_V` produce `q = W_Q x`, `k = W_K x`, `v = W_V x`. Same input, three different views: what you're looking for, what you advertise, what you'd contribute. In self-attention all three come from the same sequence; in cross-attention (encoder-decoder) Q comes from one sequence and K, V from another.

**The √d scaling.** Without it, dot products grow with the dimension *d_k* — a 128-dim dot product is roughly 8× larger than a 16-dim one. Large logits push softmax into a hard one-hot, and the gradient through that softmax vanishes. Dividing by `√d_k` keeps the variance of the logits ≈ 1 regardless of head size, so training stays stable. It's one line of code that prevents an entire failure mode.

**Causal masking.** For autoregressive models (GPT-style), token *i* must not see tokens *j > i* — otherwise the model cheats by reading the answer. Add `-∞` to the upper triangle of the `QKᵀ` matrix *before* the softmax; those entries become 0 weight. Padding mask is the same trick for variable-length batches.

**Multi-head attention.** Split the *d*-dim embedding into *h* slices of *d/h*. Run *h* independent attention computations in parallel — each with its own `W_Q^h, W_K^h, W_V^h` — then concatenate the *h* outputs and project back with `W_O`. Total FLOPs are the same as one big head, but different heads can specialise (one tracks syntax, one tracks position, one tracks coreference). This is the single most important architectural choice in the paper.

**Layer norm + residuals.** The full block is `x + Attention(LN(x))` then `x + MLP(LN(x))`. The residuals give gradients a direct path to early layers (otherwise stacking 96 layers wouldn't train). LayerNorm rescales each token vector to mean 0 / variance 1 so activations stay on a sane scale. Modern models use *pre-norm* (LN before the sublayer) — easier to train deep, slightly worse at the bottom of the loss curve.

**Encoder / decoder / encoder-decoder.** Encoder-only (BERT, ViT) — every token attends to every other; used for understanding. Decoder-only (GPT, Llama, Claude) — causal mask; used for generation. Encoder-decoder (T5, original Transformer) — encoder builds a contextual representation, decoder cross-attends into it; used for seq2seq tasks like translation. The decoder-only stack has eaten the world because pretraining on plain next-token prediction scales beautifully.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- You can start from a pretrained checkpoint (BERT, RoBERTa, Llama, Qwen, ViT — almost always the right move)
- You're building a custom block and need attention as a primitive — `nn.MultiheadAttention` or `F.scaled_dot_product_attention` (FlashAttention under the hood)
- The bottleneck is long-range structure across the input
- You need to attend to specific positions — retrieval, alignment, set-to-set mappings

</div>
<div class="no" markdown="1">

### Skip it when

- Memory budget is tight and sequences are long — try a sparse / linear attention variant or an SSM
- You need a tiny on-device model and there's no good distilled checkpoint
- Very short sequences with strong local structure — a CNN may suffice
- The task doesn't need contextualisation at all (independent per-row classification on tabular features)

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Pre-norm variant: norm before each sublayer, residual around it
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

</div>

<div class="level-next">
    <span>Want scaling laws, RoPE, KV caches, MoE?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Computational cost</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \text{FLOPs}_{\text{attention}} \;\sim\; \mathcal{O}(N^2 \cdot d), \qquad \text{FLOPs}_{\text{MLP}} \;\sim\; \mathcal{O}(N \cdot d^2) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">Attention scales **quadratically** in sequence length *N*</li>
<li markdown="1">MLPs scale linearly in *N* but quadratically in model width *d*</li>
<li markdown="1">For long context, attention dominates; for short context with wide models, the MLP does</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{attention cost} \;\propto\; (\text{seq length})^2 \times \text{model width}, \qquad \text{MLP cost} \;\propto\; \text{seq length} \times (\text{model width})^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>seq length</code>number of tokens in the input (*N*)</li>
<li markdown="1"><code>model width</code>size of each token's embedding (*d*)</li>
<li markdown="1">Attention grows with the *square* of sequence length — doubling tokens quadruples the work</li>
<li markdown="1">MLP grows with the *square* of model width — doubling embedding size quadruples the work</li>
<li markdown="1">Long context → attention dominates. Wide model on short context → MLP dominates.</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The O(N²) problem and FlashAttention.** Naïve attention materialises the full *N*×*N* logits matrix in HBM — at 8K tokens that's 64M entries per head per layer, and memory bandwidth (not FLOPs) becomes the bottleneck. **FlashAttention** computes the same softmax(QKᵀ)V in tiles, never instantiating the full matrix in slow memory. Same big-O, 2–4× wall-clock speedup, drastically lower memory. It's the default kernel behind `F.scaled_dot_product_attention` in modern PyTorch.

**RoPE (Rotary Position Embeddings).** Modern positional encoding. Instead of *adding* a position vector to the embedding, RoPE *rotates* the Q and K vectors in 2D subspaces by an angle proportional to position. The dot product `q · k` then depends only on the *relative* position `i − j` — exactly what you want for language. Generalises to longer contexts than training, and is the default in Llama, Mistral, Qwen, GPT-NeoX, and essentially every recent open model.

**KV caching.** At inference, each new token needs to attend to all previous tokens — but the keys and values for those previous tokens never change. Cache them. Without a KV cache, generating *N* tokens costs O(N³); with it, O(N²). The cache itself is the dominant memory cost at long context, which is why grouped-query attention (GQA) and multi-query attention (MQA) — sharing K/V across heads — exist.

**Mixture of Experts (MoE).** Replace the dense MLP with *E* parallel "expert" MLPs and a router that picks the top-*k* (usually *k*=2) for each token. Total parameters explode, active parameters per token stay roughly constant. Mixtral 8×7B has 47B total params but only ~13B active; DeepSeek-V3 is 671B / 37B active. Sparse activation is how frontier models scale parameters without proportionally scaling compute.

**Scaling laws.** Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") found loss falls as a power law in compute, parameters, and tokens. The Chinchilla rule: optimal training token count ≈ 20× parameter count. Most pre-2022 models were severely under-trained; modern open models train on far more tokens per parameter (Llama-3 used ~15T tokens for 70B params — orders of magnitude past Chinchilla optimal, because inference matters too).

**Inference tricks that matter in production.** Quantisation (int8 / int4 / FP8 weights — minimal quality loss with GPTQ or AWQ). Speculative decoding (a small draft model proposes *k* tokens, the big model verifies in one forward pass). Continuous batching (vLLM, TGI — pack new requests into the same batch as in-flight ones). Paged KV cache (vLLM's PagedAttention — manage the cache like virtual memory).

**What attention *really* does.** It's a differentiable lookup over a learned key-value store. This framing unifies a surprising amount of ML — retrieval, memory networks, set transformers, even some classical algorithms. Anthropic's "[Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)" is the canonical modern source for understanding attention mechanistically — induction heads, copy circuits, the residual stream as a communication channel.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Anywhere with sequential or set-structured data and enough compute
- Multimodal fusion — attention generalises across text, image, audio, video tokens
- You need dense embeddings for search (encoder transformer + mean pool / CLS)
- You want one architecture template you can scale up by adding blocks

</div>
<div class="no" markdown="1">

### Skip it when

- Sequences in the hundreds of thousands of tokens — Mamba / SSMs or hybrid architectures may win
- Online / true streaming inference with bounded state — recurrent variants are cheaper
- You need strict adversarial robustness guarantees — easier in smaller, simpler models
- Per-prediction interpretability is a hard requirement — attention weights are *not* explanations

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Causal self-attention with KV cache — the inference-time pattern in LLMs.
# Uses F.scaled_dot_product_attention, which dispatches to FlashAttention
# when available.
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))

        if cache is not None:                          # extend the KV cache
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        # is_causal only for the prefill (no cache); decoding steps attend to all cached tokens
        out = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out), new_cache
```

</div>

<div class="level-next">
    <span>Want the picture instead?</span>
    <button data-go-to="intuition" type="button">← Back to Intuition</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="transformer" markdown="1">

### What the heatmap shows

Each cell is an **attention weight**: how much the query token on that row pays attention to the key token on that column. Rows always sum to 1 — the softmax forces a budget. Bright cells mean a strong link; dim cells mean the query is ignoring that key.

Switch heads to see distinct patterns: the first two are pure positional (look back / look ahead), the second two encode linguistic structure (adjective → noun, verb → subject). A real transformer has dozens of heads per layer, each one a small N×N matrix like this, all running in parallel and composing across layers.

Click any token or any cell to make a different token the query — the highlighted row tells you exactly which other tokens that token is reading.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The canonical visual walkthrough of attention, Q/K/V, multi-head, and the encoder-decoder stack. Start here before reading the paper.</span>
</li>
<li data-tier="intuition" markdown="1">
[Karpathy — Let's build GPT, from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Two hours of pure value. Builds a decoder-only transformer in PyTorch from first principles — embeddings, attention, training loop, sampling. Watch with the code open.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The original transformer paper. Surprisingly readable. Read after the Alammar walkthrough — the architecture diagrams will finally click.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Zhang et al. — Dive into Deep Learning, ch. 11](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Code-first chapter on attention and transformers with runnable PyTorch / JAX notebooks. The cleanest worked end-to-end implementation outside Karpathy's video.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Tensor2Tensor — Transformer notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The original team's Colab walkthrough. Useful for cross-referencing the paper's diagrams against working code, especially the multi-head and positional encoding internals.</span>
</li>
<li data-tier="indepth" markdown="1">
[Anthropic — A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) <i class="fas fa-external-link-alt"></i>
<span class="annotation">For going deep on what transformers *actually do* mechanistically — induction heads, the residual stream, QK and OV circuits. Dense but illuminating once you have the basics.</span>
</li>
<li data-tier="indepth" markdown="1">
[Dao et al. — FlashAttention](https://arxiv.org/abs/2205.14135) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The IO-aware attention kernel that powers modern training and inference. The paper itself is the clearest explanation of why memory bandwidth — not FLOPs — is the real bottleneck.</span>
</li>
</ul>

</div>

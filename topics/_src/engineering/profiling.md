---
title: Profiling — ML Resources Hub
eyebrow_text: ← Engineering · Debugging &amp; Profiling
eyebrow_href: {{root}}engineering.html
heading: Profiling
lead: Where is the time going — compute, memory, data loading, communication? Profile before you optimise.
active_nav: engineering
prev_href: loss-curve-forensics.html
prev_title: Loss Curve Forensics
next_href: distributed-pitfalls.html
next_title: Distributed Training Pitfalls
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**"Why is training slow?" has a small set of answers.** The four candidates: the GPU is waiting for data (data-loading bottleneck), the GPU is waiting for the network (communication bottleneck in distributed), the GPU is doing the wrong thing (small ops, kernel launch overhead), or the model genuinely takes that long (compute-bound). Profile first; optimise the actual bottleneck.

</div>

<article class="tldr-body" markdown="1">

The first question: is your GPU utilisation high (~90%+) or low? `nvidia-smi -l 1` tells you. Low util almost always means a CPU bottleneck — data loading is the usual culprit. High util means you're compute-bound; further speedup needs kernel-level or precision changes.

**The standard tools.** `nvidia-smi` for utilisation. PyTorch profiler for op-level timing. cProfile for Python-side. `py-spy` for sampling without restart. Chrome trace viewer for everything.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### The 4 bottlenecks

- **Data loader**: GPU idle, CPU pegged. Fix: more `num_workers`, prefetching, faster format (Parquet, WebDataset)
- **Communication**: small bursts of GPU activity, gaps between. Fix: gradient accumulation, gradient bucketing, faster interconnect
- **Small ops**: kernel launch overhead dominates. Fix: `torch.compile`, kernel fusion, larger batches
- **Genuine compute**: GPU pinned, no idle. Fix: mixed precision, more efficient architecture, more / better hardware

</div>

<div class="no" markdown="1">

### Common mistakes

- Optimising before profiling
- Trusting wall-clock without isolating the cause
- Profiling on the warm-up step (cudaMalloc dominates)
- Forgetting to `torch.cuda.synchronize()` before timing

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from torch.profiler import profile, ProfilerActivity

# PyTorch profiler — both CPU and CUDA, with shape info
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(5):
        train_one_step()

# Top operators by GPU time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Or export Chrome trace, view in chrome://tracing
prof.export_chrome_trace("trace.json")
```

</div>

<div class="level-next">
<span>Want kernel fusion, memory profiling, & the data-loader recipes?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Throughput formula</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{throughput} = \frac{\text{batch size} \times \text{steps/sec}}{\text{world size}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Compare across configurations — the same compute, more throughput is the goal

</li>
<li markdown="1">

Plot vs batch size: throughput plateaus when you hit the compute bound

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{samples per second per GPU} = \frac{\text{batch size} \times \text{steps per second}}{\text{number of GPUs}} $$</span>

**In words.** Throughput is just "how many training examples does each GPU chew through per second". You compute it by multiplying the batch size (samples per step) by the step rate (steps per second), then dividing by the number of GPUs (the *world size*) to get a per-GPU number. Comparing this across configurations is the cleanest way to tell if your optimisations are doing anything; if it plateaus as you increase batch size, you've hit the compute ceiling and further gains need precision or kernel changes.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`batch size`samples processed per training step

</li>
<li markdown="1">

`steps/sec`how fast training steps happen

</li>
<li markdown="1">

`world size`number of GPUs across which work is split

</li>
<li markdown="1">

Plot vs batch size: throughput plateaus when you hit the compute bound

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The PyTorch profiler.** Captures CUDA kernels, CPU ops, memory allocations, and (optionally) stack traces. Export to Chrome trace and view in `chrome://tracing` or Perfetto. Shows the gap pattern between kernels — the gaps tell you where the GPU is waiting.

**Memory profiling.** `torch.cuda.memory_summary()` for a snapshot. `torch.cuda.memory._record_memory_history()` for a full timeline (PyTorch 2.0+). Plotted via `memory_viz` — shows every allocation over time.

**Data-loader recipes.** `num_workers` ≈ number of CPU cores ÷ 2. `pin_memory=True` if going GPU. `persistent_workers=True` to avoid restarting workers each epoch. `prefetch_factor=4` to pipeline loading and compute. Store data in fast formats (Parquet for tabular, WebDataset for images, tarred for streaming).

**torch.compile.** PyTorch 2.0+ JIT. Often 30–50% speed-up on transformers; fuses ops and inlines kernels. Worth trying on any model that's not data-loader bound. Some ops fall back to eager; the trace will show you which.

**Mixed precision.** bf16 (Ampere+) is more robust than fp16; same speed-up. `torch.amp.autocast("cuda", dtype=torch.bfloat16)`. Doubles throughput on most transformers; loses a tiny bit of accuracy that's well worth it.

**Gradient accumulation.** Want a batch of 1024 but only 128 fits? Run 8 micro-batches, accumulate gradients, step once. Effectively the same as batch 1024 on most architectures (but watch BatchNorm).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from torch.utils.data import DataLoader

# Sensible data-loader defaults
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)

# torch.compile — often 30%+ speed-up
model = torch.compile(model, mode="default")     # or "max-autotune" for max speedup

# Gradient accumulation — bigger effective batch without OOM
opt.zero_grad()
for micro_batch in chunked(batch, 4):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_fn(model(micro_batch)) / 4    # divide by accumulation steps
    loss.backward()
opt.step()
```

</div>

<div class="level-next">
<span>Want kernel-level optimisation, FlashAttention, & NVIDIA tools?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Roofline model</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{perf} = \min\!\big(\text{peak FLOP/s}, \;\text{arithmetic intensity} \times \text{peak bandwidth}\big) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Memory-bound ops (low arithmetic intensity) can't reach peak FLOPs

</li>
<li markdown="1">

Fusing memory-bound ops shifts you toward compute-bound

</li>
<li markdown="1">

Hardware vendors' roofline plots show the theoretical limits

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{achievable speed} = \min\!\big(\text{compute ceiling},\;\text{memory ceiling}\big) $$</span>

**In words.** Your kernel's speed is capped by whichever resource runs out first: raw compute (FLOP/s = floating-point ops per second) or memory bandwidth (how fast you can move bytes). `min(...)` just picks the smaller of the two ceilings. *Arithmetic intensity* is "how many FLOPs you do per byte fetched" — if it's low, you're stuck on memory (the kernel spends time moving data); if it's high, you're stuck on compute. Fusing small ops into bigger ones raises arithmetic intensity and shifts you toward the compute ceiling, where peak hardware speed lives.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`compute ceiling`peak floating-point operations per second on this hardware

</li>
<li markdown="1">

`memory ceiling`arithmetic intensity × peak memory bandwidth

</li>
<li markdown="1">

`arithmetic intensity`FLOPs done per byte of memory accessed

</li>
<li markdown="1">

Fusing memory-bound ops shifts you toward compute-bound

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**FlashAttention.** Dao et al. (2022, 2023). Memory-aware exact attention — tiles the QK<sup>T</sup>V computation to stay in SRAM. 2–4× speed-up on long sequences and a memory reduction that lets you fit much larger contexts. The standard attention implementation in modern frameworks.

**NVIDIA tools.** `nvprof` / `nsys` for system-wide profiles. `ncu` (Nsight Compute) for kernel-level analysis. `nvidia-smi dmon` for live monitoring. Heavier than PyTorch's profiler; necessary when you're chasing the last ten percent.

**Custom kernels.** Triton (OpenAI) or CUDA. Pays off when you have a hot path that PyTorch's eager / compile path doesn't fuse well. `torch.utils.cpp_extension` for CUDA / C++ kernels; `triton.language` for the easier path.

**Activation checkpointing.** Trade compute for memory — recompute activations on the backward pass instead of storing them. `torch.utils.checkpoint.checkpoint`. Often the only way to fit large models on limited GPUs.

**Distributed bottlenecks.** All-reduce time grows with parameter count. Gradient bucketing lumps small tensors. Overlap compute and communication. ZeRO (DeepSpeed, FSDP) shards optimizer state across ranks. Pipeline parallelism for very long models.

**Memory fragmentation.** Lots of allocations and frees → cudaMalloc returns "no memory" even though some is "free". `torch.cuda.empty_cache()` can help; `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF` reduces fragmentation in PyTorch 2.0+.

**The "compile then optimise" hierarchy.** torch.compile → mixed precision → gradient accumulation → data-loader tuning → activation checkpointing → custom kernels. Apply in this order; most workflows stop after the first 3.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from torch.utils.checkpoint import checkpoint

# Activation checkpointing — recompute on backward, save memory
class CheckpointedBlock(torch.nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x):
        return checkpoint(self.inner, x, use_reentrant=False)

# Use FlashAttention via SDPA backend
from torch.nn.functional import scaled_dot_product_attention
with torch.nn.attention.sdpa_kernel(
    backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]
):
    out = scaled_dot_product_attention(q, k, v, is_causal=True)

# Avoid fragmentation
# Env var (set before launch): PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

[PyTorch Profiler Recipe <i class="fas fa-external-link-alt"></i>](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html){: target="_blank" }
<span class="annotation">Official walkthrough on the PyTorch profiler with Chrome-trace integration.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Horace He — Making Deep Learning Go Brrrr <i class="fas fa-external-link-alt"></i>](https://horace.io/brrr_intro.html){: target="_blank" }
<span class="annotation">Best single read on what actually makes ML workloads fast. Memory bandwidth, kernel fusion, the modern intuition.</span>

</li>
<li data-tier="indepth" markdown="1">

[Dao et al. (2022) — FlashAttention <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2205.14135){: target="_blank" }
<span class="annotation">The paper that made long-context attention practical. Tile the computation to stay in SRAM.</span>

</li>
<li data-tier="indepth" markdown="1">

[Triton <i class="fas fa-external-link-alt"></i>](https://github.com/openai/triton){: target="_blank" }
<span class="annotation">OpenAI's Python-flavoured CUDA. The pragmatic way to write custom kernels without going full CUDA C++.</span>

</li>
</ul>

</div>

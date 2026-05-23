---
title: Distributed Training Pitfalls — ML Resources Hub
eyebrow_text: ← Engineering · Debugging &amp; Profiling
eyebrow_href: {{root}}engineering.html
heading: Distributed Training Pitfalls
lead: Data parallel, model parallel, FSDP — the engineering nuances that turn "scales linearly" into "works at all".
active_nav: engineering
prev_href: profiling.html
prev_title: Profiling
next_href: serving.html
next_title: Serving
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The cluster gives you many GPUs; the framework lets them cooperate.** Data parallel: same model on each GPU, different data, all-reduce gradients. Model parallel: split the model across GPUs. FSDP / ZeRO: shard both. The right choice depends on what fits — and "what fits" is the main constraint.

</div>

<article class="tldr-body" markdown="1">

**Data parallel (DDP).** The default. Each GPU has a full copy of the model. They process different batches, then all-reduce gradients. Easy to set up, scales well for models that fit on one GPU.

**FSDP (Fully Sharded Data Parallel).** Shards parameters, gradients, and optimizer state across GPUs. Lets you train models that wouldn't fit on a single GPU. Built into PyTorch; DeepSpeed's ZeRO has comparable functionality.

**Tensor / model parallel.** Split individual layers across GPUs. Used for very large models where even FSDP is insufficient. Megatron, FairScale. Most teams don't need this.

**Pipeline parallel.** Run different layers on different GPUs, pipelined. GPipe, PipeDream. Useful when one model doesn't fit but a layer does.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Pick by model size

- **< 1 GPU fits**: don't bother distributed — single GPU is simplest
- **Model fits, want more throughput**: DDP
- **Model nearly fits**: FSDP / ZeRO-2
- **Model too big for any GPU**: FSDP + activation checkpointing + sometimes tensor parallel
- **Hundreds of GPUs**: 2D / 3D parallelism (DDP × FSDP × pipeline)

</div>

<div class="no" markdown="1">

### Common pitfalls

- Different seeds per rank → ranks diverge silently
- Logging from all ranks → step counter inflated by world_size
- Batch norm with global stats vs per-rank → wrong stats in distributed
- Saving from rank ≠ 0 → race conditions, corrupted checkpoints

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    dist.init_process_group("nccl")     # NCCL for NVIDIA, gloo for CPU
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def train(rank):
    model = MyNet().cuda(rank)
    model = DDP(model, device_ids=[rank])
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                  rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)           # different shuffle per epoch
        for x, y in loader:
            loss = loss_fn(model(x.cuda(rank)), y.cuda(rank))
            loss.backward()
            opt.step(); opt.zero_grad()

        # Save only on rank 0
        if rank == 0:
            torch.save(model.module.state_dict(), f"epoch_{epoch}.pt")

# Launch: torchrun --nproc-per-node=4 train.py
```

</div>

<div class="level-next">
<span>Want FSDP, checkpoint sharding, & gradient communication optimisations?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">FSDP memory savings</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{memory per GPU} \approx \frac{P + G + O}{W} + A $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*P* parameters, *G* gradients, *O* optimizer state, *W* world size, *A* activations

</li>
<li markdown="1">

Vanilla DDP: *P + G + O + A* per GPU

</li>
<li markdown="1">

FSDP: divides the first three by *W*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{memory per GPU} \;\approx\; \frac{\text{params} + \text{gradients} + \text{optimizer state}}{\text{number of GPUs}} \;+\; \text{activations} $$</span>

**In words.** A back-of-the-envelope for how much GPU memory FSDP saves. There are four big consumers: the model parameters `P`, their gradients `G`, the optimizer state `O` (for Adam, this is roughly 2× the parameter count), and the activations `A` kept for backprop. Vanilla DDP keeps full copies of *all four* on every GPU. FSDP shards the first three across `W` GPUs (the "world size"), so each GPU only holds a fraction. Activations are not sharded by FSDP — they're per-sample, so each GPU still has its own based on its mini-batch.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`params`P — the model weights

</li>
<li markdown="1">

`gradients`G — same shape as params, one per backward pass

</li>
<li markdown="1">

`optimizer state`O — Adam's momentum + variance (~2× params)

</li>
<li markdown="1">

`number of GPUs`W — total GPUs across the cluster ("world size")

</li>
<li markdown="1">

`activations`A — intermediate tensors kept for backward; not sharded

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**FSDP details.** Wraps your model recursively at a chosen granularity (per layer or per block). Each forward pass: all-gathers parameters for the active block, then frees them. Backward gathers again. Costs extra communication for the memory savings.

**Gradient bucketing.** DDP batches small gradient tensors into "buckets" before all-reducing. Reduces per-tensor overhead. Default bucket size is fine; tune for very-small or very-large models.

**Overlap compute and communication.** Modern DDP / FSDP launch the next layer's compute while the previous layer's gradient is being all-reduced. The default is good; verify with the profiler that the gap between kernels is small.

**Checkpoint sharding.** A 100B-parameter model can't be saved as a single 400 GB file. PyTorch's `FullStateDictType.SHARDED_STATE_DICT` writes one file per rank. Reload with the same sharding.

**Mixed precision in distributed.** Use bf16 for compute, fp32 for the master parameters. GradScaler with FSDP. Most frameworks handle this automatically when you turn on AMP + the right FSDP precision policy.

**The launcher.** `torchrun` for PyTorch, `accelerate launch` from HuggingFace, `deepspeed` for DeepSpeed. Each handles process spawning, environment variables, and (sometimes) restart-from-failure.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def fsdp_wrap(model):
    return FSDP(
        model,
        auto_wrap_policy=transformer_auto_wrap_policy({TransformerBlock}),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

# Save sharded checkpoint
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sd = model.state_dict()
    torch.save(sd, f"ckpt-rank{dist.get_rank()}.pt")
```

</div>

<div class="level-next">
<span>Want tensor / pipeline parallelism, NCCL tuning, & multi-node debugging?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">3D parallelism</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{world} = \text{tensor} \times \text{pipeline} \times \text{data} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Three orthogonal axes of parallelism

</li>
<li markdown="1">

Total GPUs = product of the three

</li>
<li markdown="1">

The standard recipe at LLM-training scale

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{total GPUs} \;=\; (\text{tensor-parallel size}) \;\times\; (\text{pipeline-parallel size}) \;\times\; (\text{data-parallel size}) $$</span>

**In words.** At LLM-training scale, you split the work along three orthogonal axes simultaneously. **Tensor parallel** shards individual layers across a few GPUs in a node (NVLink). **Pipeline parallel** puts different layers on different nodes, streaming micro-batches through. **Data parallel** replicates the whole stack on multiple groups, with different data per replica. The cluster's total GPU count factorises into the product of the three group sizes — and the right factorisation depends on the model's shape, the interconnect topology, and what fits where.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`tensor-parallel size`GPUs cooperating on a single layer's weights

</li>
<li markdown="1">

`pipeline-parallel size`stages of the pipeline (each holds some layers)

</li>
<li markdown="1">

`data-parallel size`independent replicas processing different batches

</li>
<li markdown="1">

`total GPUs`product of all three — your full cluster

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Tensor parallel.** Split a single layer (e.g., a linear's weight matrix) across GPUs. Each GPU computes part of the output. Requires fast intra-node interconnect (NVLink). Used for the largest models.

**Pipeline parallel.** Different layers on different GPUs. Micro-batches flow through the pipeline; "bubble" of idle time at start and end. GPipe (Huang et al. 2018), PipeDream, 1F1B (one-forward-one-backward) scheduling.

**3D parallelism.** Combine DP, TP, PP. Standard at the largest scales (1000+ GPUs). Megatron-LM is the reference implementation. NVIDIA's NeMo, DeepSpeed, and PyTorch's `torch.distributed` all support some version.

**NCCL tuning.** `NCCL_DEBUG=INFO` for diagnostics. Topology-aware: NCCL detects PCIe / NVLink. For very large jobs, tuning `NCCL_TREE_THRESHOLD`, `NCCL_ALGO`, `NCCL_PROTO`. Mostly trial-and-error; profile before tuning.

**Multi-node debugging.** When jobs run for hours on hundreds of nodes, a single bad NIC can corrupt training. Heartbeat checks, gradient norm checks, periodic checkpoints. Sentry / Datadog / Grafana with cluster-level alerting.

**Elastic training.** Nodes can join or drop mid-training. `torchrun --rdzv-backend=etcd` for elastic rendezvous. Checkpointing must be fast (incremental, streaming) to make resumes cheap.

**The cluster you wish you had.** Saturated all-reduce bandwidth is rare on real clusters; profile to confirm. Often the bottleneck is something more mundane: a slow storage backend, a bad scheduler queue, or a single bad node.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# Megatron-style tensor parallelism for a Linear layer
import torch, torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Splits the weight matrix's output dimension across ranks."""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        assert out_features % world_size == 0
        local_out = out_features // world_size
        self.linear = nn.Linear(in_features, local_out, bias=False)

    def forward(self, x):
        local_out = self.linear(x)
        # All-gather to reconstruct the full output (if needed downstream)
        gathered = [torch.zeros_like(local_out) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_out)
        return torch.cat(gathered, dim=-1)
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

[PyTorch Distributed Documentation <i class="fas fa-external-link-alt"></i>](https://pytorch.org/docs/stable/distributed.html){: target="_blank" }
<span class="annotation">The reference for DDP, FSDP, RPC, and the lower-level c10d primitives. Long; bookmark the parts you actually use.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[HuggingFace — Efficient Training on Many GPUs <i class="fas fa-external-link-alt"></i>](https://huggingface.co/docs/transformers/perf_train_gpu_many){: target="_blank" }
<span class="annotation">Practical recipes for distributed training, FSDP / DeepSpeed integration, and "which approach do I pick" decision trees.</span>

</li>
<li data-tier="indepth" markdown="1">

[Shoeybi et al. (2019) — Megatron-LM <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1909.08053){: target="_blank" }
<span class="annotation">Tensor parallelism for transformers. Foundational for understanding modern LLM training.</span>

</li>
<li data-tier="indepth" markdown="1">

[DeepSpeed <i class="fas fa-external-link-alt"></i>](https://www.deepspeed.ai/){: target="_blank" }
<span class="annotation">Microsoft's distributed-training toolkit. ZeRO, pipeline, 3D parallelism, and inference optimisations.</span>

</li>
</ul>

</div>

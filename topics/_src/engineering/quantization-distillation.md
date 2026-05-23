---
title: Quantization &amp; Distillation — ML Resources Hub
eyebrow_text: ← Engineering · Production
eyebrow_href: {{root}}engineering.html
heading: Quantization &amp; Distillation
lead: Make models smaller and faster — int8 / int4, knowledge distillation, pruning. What each one costs in accuracy.
active_nav: engineering
prev_href: monitoring.html
prev_title: Monitoring
next_href: ab-testing.html
next_title: A/B Testing
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Inference cost is the binding constraint for most deployed ML.** Three knobs reduce it: quantize (smaller representations), distill (smaller architectures), prune (sparser weights). Combined they routinely give 4-10× speed-ups with marginal accuracy loss. The exact recipe depends on the model and the deployment target.

</div>

<article class="tldr-body" markdown="1">

**Quantization.** Convert fp32 weights and activations to int8 (or int4 / fp8). 4× memory, 2-4× faster matmuls. Three flavours: post-training quantization (PTQ — apply to a trained model), quantization-aware training (QAT — train with simulated quantization), and weight-only quantization (the easiest for LLMs).

**Distillation.** Train a small "student" model to imitate a large "teacher" model. The student learns soft probabilities (not hard labels), often along with hidden representations. Works because the teacher's full output distribution encodes information that the labels alone don't.

**Pruning.** Remove weights / neurons / heads that contribute little. Unstructured (individual weights → sparse matrix) or structured (whole channels / heads → smaller dense matrix). Structured pruning gives real speed-ups on standard hardware.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### When to use each

- **Quantize**: every deployment. Lowest risk, biggest typical win
- **Distill**: when target is much smaller than teacher; needs training pipeline
- **Prune**: structured pruning before quantization for compounded gains
- **Speculative decoding**: for LLMs — use a small model as drafter

</div>

<div class="no" markdown="1">

### Trade-offs

- int8 PTQ: ~1pp accuracy drop, sometimes none
- int4 weight-only: usable for LLMs; vision models often suffer more
- Distillation: needs more training compute, possibly more data
- Unstructured pruning: rarely faster on commodity GPUs

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from torch.ao.quantization import quantize_dynamic

# Dynamic quantization — int8 weights, fp32 activations.
# Easiest possible win; ~2× speedup on CPU inference for transformers / LSTMs.
qmodel = quantize_dynamic(
    model.eval(),
    {torch.nn.Linear},
    dtype=torch.qint8,
)
torch.save(qmodel.state_dict(), "quantized.pt")
```

</div>

<div class="level-next">
<span>Want PTQ vs QAT, distillation losses, & the LLM quantization stack?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Distillation loss</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L} = \alpha \cdot \mathrm{CE}(y_{\text{hard}}, p_S) + (1-\alpha) \cdot T^2 \cdot \mathrm{KL}\!\big(p_T / T \;\|\; p_S / T\big) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*T* = temperature (typically 2–10) — softens both distributions

</li>
<li markdown="1">

*α* mixes hard-label loss with soft-distribution-matching loss

</li>
<li markdown="1">

Hinton et al. (2015) — the original knowledge distillation recipe

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{student loss} = \alpha \cdot (\text{normal label loss}) + (1-\alpha) \cdot T^2 \cdot (\text{difference from teacher's softened probabilities}) $$</span>

**In words.** The student model is trained on two losses at once. The first piece is standard cross-entropy against the hard label `y`. The second piece (the "distillation" part) measures how far the student's softened probability distribution is from the teacher's — `KL` stands for Kullback-Leibler divergence, just a distance between two probability distributions. `T` (temperature) is a Greek letter that softens both distributions when bigger than 1 — small differences in logits become more visible, so the student learns from the teacher's relative uncertainty rather than just its top pick. `α` (alpha) is a knob between 0 and 1 that mixes how much you care about each loss.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`normal label loss`cross-entropy between student's prediction and the true label

</li>
<li markdown="1">

`difference from teacher`KL divergence between teacher and student probability distributions

</li>
<li markdown="1">

`T`temperature — bigger softens probabilities so the student sees the teacher's subtle preferences

</li>
<li markdown="1">

`α`mix factor between hard-label loss and teacher-matching loss

</li>
<li markdown="1">

Hinton et al. (2015) — the original knowledge distillation recipe

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Post-training quantization (PTQ).** Quantize a trained model with calibration data — find scale/zero-point per layer that minimises quantization error. Easy, fast, ~1pp accuracy drop typical. `torch.ao`, ONNX Runtime, TensorRT all support it.

**Quantization-aware training (QAT).** Train with simulated quantization in the forward pass. The model adapts; accuracy is usually equal to fp32. Costs an extra training pass; useful when PTQ loses too much.

**Weight-only quantization (LLMs).** Weights in int4, activations stay in fp16/bf16. Specialised kernels (FlashInfer, ExLlama, AWQ) for the dequantize-and-multiply. Standard for LLM inference. GPTQ, AWQ, bitsandbytes are the three reference algorithms.

**Knowledge distillation.** Loss matches the teacher's softmax outputs (soft labels), often along with intermediate hidden states. Works because the teacher's "wrong" output probabilities encode useful structure. DistilBERT (66% smaller, 97% of BERT's accuracy) is the canonical demo.

**Structured pruning.** Remove entire channels, heads, or layers. `torch.nn.utils.prune` for unstructured; `SparseML`, `nn_pruning` for structured. Modern recipe: prune + fine-tune to recover accuracy.

**Mixed quantization.** Different layers at different precisions. `SmoothQuant` for LLMs; `llm.int8()` for outlier-aware schemes. The largest gains often come from being smart about *where* the precision lives.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn.functional as F

# Hinton-style distillation loss
def distill_loss(student_logits, teacher_logits, y_hard, alpha=0.7, T=4.0):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, y_hard)
    return alpha * soft_loss + (1 - alpha) * hard_loss

# LLM weight-only quantization — bitsandbytes (huggingface)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct",
                                              quantization_config=qconf)
```

</div>

<div class="level-next">
<span>Want SmoothQuant, GPTQ, AWQ, & modern LLM compression?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Per-channel quantization scale</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ q_i = \mathrm{round}\!\left(\frac{w_i}{s_c}\right) \cdot s_c, \quad s_c = \frac{\max_i |w_{i,c}|}{127} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Each output channel *c* gets its own scale

</li>
<li markdown="1">

Mitigates the loss from outlier channels

</li>
<li markdown="1">

Per-tensor is the alternative — cheaper, more loss

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{quantized weight} = \mathrm{round}\!\left(\frac{\text{weight}}{\text{channel scale}}\right) \cdot \text{channel scale}, \quad \text{channel scale} = \frac{\text{biggest abs weight in that channel}}{127} $$</span>

**In words.** To squash a float weight into int8 (which represents −127 to +127), divide by a scale, round to the nearest integer, then multiply back. The trick is picking the scale: a per-channel scale uses the biggest absolute value *within* that output channel, so a noisy outlier in one channel doesn't blow up the scale for every other channel. The 127 in the denominator is the max magnitude of int8. `round(...)` just snaps the result to the nearest representable integer.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`weight`an original floating-point parameter value

</li>
<li markdown="1">

`channel scale`the divisor chosen per output channel

</li>
<li markdown="1">

`biggest abs weight in that channel`the largest magnitude in that channel

</li>
<li markdown="1">

Per-tensor is the alternative — one scale for everything, cheaper but loses more

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**SmoothQuant.** Xiao et al. (2023). LLM activations have huge outliers in a few channels. Shift the difficulty from activations into weights by an absorbed scaling factor; quantize both successfully in int8. Production-quality int8 LLM inference.

**GPTQ.** Frantar et al. (2022). Layer-wise weight quantization with second-order error minimisation. Strong int4 quantization for LLMs at relatively low cost.

**AWQ.** Lin et al. (2023). Identify which weight channels are most "salient" and protect them during quantization. Typically beats GPTQ for int4 LLM serving.

**LoRA + quantization (QLoRA).** Dettmers et al. (2023). Fine-tune a quantized base model with LoRA adapters. Lets you fine-tune 65B-parameter models on a single 48GB GPU.

**Speculative decoding.** Use a small drafter model to propose *k* tokens; the big target model verifies them in one pass. Standard in vLLM, TGI, llama.cpp. Often gives 2× speed-up for LLMs at no accuracy cost.

**Mixture of Experts (MoE) inference.** Only a fraction of parameters active per token. Effective parameter count is much higher than active. Different deployment story than dense models — KV cache management, routing optimisation, expert co-location.

**Hardware-aware design.** Different hardware likes different things: GPUs reward batch parallelism; CPUs reward int8 + vectorisation; mobile rewards int4 + sparsity. The "best" compression is hardware-dependent.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# QLoRA — fine-tune a 4-bit base model with LoRA adapters
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype="bfloat16"),
    device_map="auto",
)
base = prepare_model_for_kbit_training(base)
peft_cfg = LoraConfig(r=16, lora_alpha=32,
                     target_modules=["q_proj", "v_proj"],
                     bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, peft_cfg)
model.print_trainable_parameters()
# trainable: ~1% of total — fits on a single consumer GPU
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

[PyTorch Quantization Documentation <i class="fas fa-external-link-alt"></i>](https://pytorch.org/docs/stable/quantization.html){: target="_blank" }
<span class="annotation">Official guide. PTQ, QAT, the new <code>torch.ao</code> API.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Hinton et al. (2015) — Knowledge Distillation <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1503.02531){: target="_blank" }
<span class="annotation">The original distillation paper. Still readable; the temperature trick is the heart of the recipe.</span>

</li>
<li data-tier="indepth" markdown="1">

[Dettmers et al. (2023) — QLoRA <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2305.14314){: target="_blank" }
<span class="annotation">Fine-tune giant models on a single GPU via 4-bit base + LoRA. Practical revolution for LLM customisation.</span>

</li>
<li data-tier="indepth" markdown="1">

[AutoAWQ <i class="fas fa-external-link-alt"></i>](https://github.com/casper-hansen/AutoAWQ){: target="_blank" }
<span class="annotation">Reference implementation of AWQ for LLM weight quantization. Plays nicely with HuggingFace transformers.</span>

</li>
</ul>

</div>

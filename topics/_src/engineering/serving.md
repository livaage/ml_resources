---
title: Serving — ML Resources Hub
eyebrow_text: ← Engineering · Production
eyebrow_href: {{root}}engineering.html
heading: Serving
lead: FastAPI, BentoML, Triton, vLLM — how a trained model becomes an endpoint your product can call.
active_nav: engineering
prev_href: distributed-pitfalls.html
prev_title: Distributed Training Pitfalls
next_href: monitoring.html
next_title: Monitoring
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Training and serving are different problems.** Training optimises for throughput on a fixed dataset. Serving optimises for latency on one request at a time (or small batches). The right tool, the right format, and the right batching strategy all change. Don't ship a 100-line Flask app; use one of the four standard serving stacks.

</div>

<article class="tldr-body" markdown="1">

**The four standard stacks.** FastAPI + uvicorn for "I want a Python endpoint that calls my model" (small models, hobby projects). BentoML for "I want a packaged, deployable model" (production tabular / sklearn / small DL). Triton for "I want NVIDIA-optimised, dynamic batching, multi-framework". vLLM (or TGI) for "I'm serving an LLM and need throughput".

**Three big serving decisions.** Sync vs async (mostly sync for low-latency, async if you can wait or stream). Single-request vs batched (batching trades latency for throughput). CPU vs GPU (CPU for small models, GPU for transformers / large CNNs).

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Stack by use-case

- **Small Python model**: FastAPI + uvicorn
- **Tabular / sklearn / packaged**: BentoML
- **GPU model with dynamic batching**: Triton
- **LLM**: vLLM, TGI, or a hosted API
- **Edge / mobile**: ONNX Runtime, TFLite, CoreML

</div>

<div class="no" markdown="1">

### Common pitfalls

- Loading the model on every request — load once at startup
- Hardcoded paths — use a registry alias instead
- Synchronous I/O blocking GPU — use async or thread pool
- No request validation — bad inputs crash the server
- No timeout — a slow GPU stalls everything

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch, mlflow

app = FastAPI()

# Load the model once at startup
MODEL = mlflow.pytorch.load_model("models:/credit-risk@production").eval()

class Request(BaseModel):
    features: list[float] = Field(..., min_length=16, max_length=16)

class Response(BaseModel):
    score: float
    model_version: str

@app.post("/predict", response_model=Response)
def predict(req: Request):
    x = torch.tensor(req.features, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        score = MODEL(x).sigmoid().item()
    return Response(score=score, model_version=MODEL.version)

@app.get("/health")
def health(): return {"status": "ok"}
```

</div>

<div class="level-next">
<span>Want dynamic batching, ONNX, & the LLM serving stack?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Latency budget decomposition</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ T_{\text{request}} = T_{\text{net}} + T_{\text{preprocess}} + T_{\text{queue}} + T_{\text{forward}} + T_{\text{postprocess}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Profile each piece — the bottleneck is rarely where you think

</li>
<li markdown="1">

Pre/post-processing often dominates small-model latency

</li>
<li markdown="1">

Queue time is what dynamic batching trades against

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{total request time} \;=\; \text{network} + \text{preprocess} + \text{queue wait} + \text{model forward pass} + \text{postprocess} $$</span>

**In words.** A served prediction's total latency is the sum of several stages. `T` (with various subscripts) just stands for "time spent in this stage". Network time is round-trip on the wire; preprocess is turning JSON into a tensor; queue time is waiting in line behind other requests; the forward pass is the actual model call; postprocess turns the model's output back into a response. Each one is independently measurable; the bottleneck is almost never the one you guess first.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`network`time to send request and response across the wire

</li>
<li markdown="1">

`preprocess`parsing input, validating, converting to tensor

</li>
<li markdown="1">

`queue wait`time spent waiting behind other requests

</li>
<li markdown="1">

`model forward pass`the actual neural-network compute

</li>
<li markdown="1">

`postprocess`converting model output back into a response

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Dynamic batching.** The server accumulates inflight requests up to a small time window (10–50 ms) or batch size; runs the model once on the batch; splits the outputs. Triton, BentoML, and most serving frameworks support this natively. Huge throughput win for GPU-served models with bursty traffic.

**ONNX.** Open neural-network exchange — convert your PyTorch / TF model to a portable graph. Runs on ONNX Runtime (CPU, GPU, edge, mobile). Useful when you want to deploy outside the original training framework, or want graph-level optimisations.

**TorchScript vs torch.compile vs ONNX.** TorchScript: PyTorch's classic graph format. `torch.compile`: modern JIT (PyTorch 2.0+); usually faster for training. ONNX: framework-agnostic. For pure inference on PyTorch, `torch.compile` or ONNX Runtime are usually fastest.

**LLM serving stack.** vLLM (UC Berkeley) and TGI (HuggingFace) are the open-source standards. PagedAttention to avoid memory waste, continuous batching to mix concurrent requests, KV cache management. Hosted APIs (OpenAI, Anthropic, Replicate) abstract all of this.

**Quantisation for serving.** int8 / int4 weights. Often 2–4× smaller and faster with little accuracy loss. `bitsandbytes`, `autoawq`, `gptq` for LLMs. `torch.ao` for tabular / vision. See [Quantization & Distillation](quantization-distillation.html).

**Multi-model serving.** Modern serving frameworks let you load multiple models in one process, share GPU memory, and route requests based on the URL. Useful when you have many small models — or different fine-tunes of one base.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import bentoml, torch

# Export model for BentoML
saved = bentoml.pytorch.save_model("credit-risk", model)

# Service definition
@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 5},
    workers=1,
)
class CreditRiskService:
    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=64,
        max_latency_ms=50,
    )
    async def predict(self, features: list[list[float]]) -> list[float]:
        x = torch.tensor(features)
        with torch.inference_mode():
            scores = self.model(x).sigmoid()
        return scores.tolist()

# bentoml serve credit_risk:CreditRiskService --port 3000
```

</div>

<div class="level-next">
<span>Want LLM serving deep-dive, autoscaling, & SLA engineering?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">LLM serving throughput</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{tok/s} \approx \frac{\text{compute capacity}}{\text{KV cache pressure} + \text{prefill cost}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

KV cache dominates memory for long contexts

</li>
<li markdown="1">

Continuous batching mixes prefill and decode → big throughput gain

</li>
<li markdown="1">

PagedAttention frees fragmented KV memory blocks

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{tokens per second} \;\approx\; \frac{\text{available GPU compute}}{\text{memory tied up in KV cache} + \text{cost of processing the prompt}} $$</span>

**In words.** LLM throughput (tokens generated per second across all requests) is roughly compute divided by overhead. The `≈` says this is approximate. The KV cache is per-request memory that grows with context length — every active request consumes some, leaving less compute for new ones. Prefill is the initial pass over the prompt, which is compute-heavy and competes with the per-token decoding work. Throughput rises when you reduce both — paged KV memory and continuous batching are the two big wins.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`tokens per second`throughput across all concurrent requests

</li>
<li markdown="1">

`available GPU compute`raw matrix-multiply capacity

</li>
<li markdown="1">

`KV cache`per-request attention keys/values, kept around so each new token only computes new K/V

</li>
<li markdown="1">

`prefill cost`the one-time pass to encode the prompt before generation begins

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**PagedAttention (vLLM).** Kwon et al. (2023). KV cache stored in paged blocks (like virtual memory). Avoids the fragmentation of contiguous-allocation schemes. 2–4× throughput improvement.

**Continuous batching.** Don't wait for a batch to finish before starting a new one — feed in new requests as old ones complete. Combined with prefill / decode separation, much higher GPU util than naive batching.

**Speculative decoding.** A small "draft" model proposes *k* tokens; the big "target" model verifies them in one pass. Accept the prefix the target agrees with; resample from where it disagrees. 2× speed-up with no accuracy loss (in expectation). Built into vLLM, TGI, llama.cpp.

**Autoscaling.** KServe, BentoCloud, Modal, RunPod. Scale replicas based on QPS, queue depth, or GPU utilisation. Pre-warmed pools for low cold-start. Scale-to-zero for cost-sensitive workloads.

**SLA engineering.** p50, p95, p99 latency. Saturation thresholds. Error budgets. Most ML APIs have "p99 latency under X ms" as a hard requirement. Monitor + alert on it; load-test with realistic traffic distributions.

**Edge inference.** ONNX Runtime, TFLite, CoreML, MediaPipe. Quantisation is essential (int8 or int4). Many vision and small NLP models run on phones with sub-100ms latency.

**Streaming responses.** For LLMs, stream tokens as they're generated. Server-Sent Events or HTTP chunked transfer. Improves perceived latency dramatically; standard in all production chat APIs.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# vLLM — serve an LLM with PagedAttention + continuous batching
# pip install vllm
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    enable_chunked_prefill=True,
)

sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
outputs = llm.generate([prompt], sampling)
print(outputs[0].outputs[0].text)

# Or run as a server:
# vllm serve meta-llama/Llama-3-8B-Instruct --port 8000
# OpenAI-compatible endpoint at /v1/chat/completions
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

[vLLM Documentation <i class="fas fa-external-link-alt"></i>](https://docs.vllm.ai/){: target="_blank" }
<span class="annotation">The reference open-source LLM serving stack. PagedAttention, continuous batching, OpenAI-compatible API.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[BentoML Documentation <i class="fas fa-external-link-alt"></i>](https://docs.bentoml.com/){: target="_blank" }
<span class="annotation">Excellent practical framework for packaging and deploying any ML model. Multi-framework support, dynamic batching, Yatai for production.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[NVIDIA Triton Inference Server <i class="fas fa-external-link-alt"></i>](https://docs.nvidia.com/deeplearning/triton-inference-server/){: target="_blank" }
<span class="annotation">NVIDIA's production-grade serving framework. Multi-framework, multi-model, dynamic batching, ensemble pipelines.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[ONNX Runtime <i class="fas fa-external-link-alt"></i>](https://onnxruntime.ai/){: target="_blank" }
<span class="annotation">Cross-platform model runtime. The standard answer for "deploy this model somewhere other than where it was trained".</span>

</li>
</ul>

</div>

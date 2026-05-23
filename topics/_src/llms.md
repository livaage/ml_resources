---
title: Foundation Models &amp; LLMs — ML Resources Hub
eyebrow_text: ← Theory · Frontier
eyebrow_href: ../theory.html
heading: Foundation Models &amp; LLMs
lead: Big transformers, trained on most of the internet, that you fine-tune or just prompt for everything.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Train one huge model on huge data with a simple objective; reuse it everywhere.** Next-token prediction on trillions of tokens. The resulting model has absorbed so much structure that you can prompt it to do almost any text task — without retraining. Fine-tune for specific behaviour; RLHF for alignment. The economics of ML changed when this started working.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Watch capability emerge with scale — synthetic abilities sweep across model size on a log scale</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                model scale
                <input id="viz-llm-scale" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-llm-scale-lbl">10⁹ params</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-llm-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-llm-caption"></div>
</div>

<script src="{{root}}js/viz/llms.js"></script>

Stylised capability curves vs model scale. **Memorisation** and **basic syntax** emerge early. **Arithmetic**, **multi-step reasoning**, and **tool use** emerge much later, often with sharp transitions ("emergent abilities"). These curves are caricatures of real benchmark scaling — but the shape is qualitatively right.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**The recipe.** Pre-train a decoder-only transformer on ~trillions of tokens. Tokenise with BPE. Train with next-token cross-entropy. Use AdamW, cosine LR schedule, large batch size. Scale parameters, data, and compute together (Chinchilla scaling).

**Then specialise.** Supervised fine-tune (SFT) on (instruction, response) pairs. Optionally RLHF or DPO with human preference data. Now you have an instruction-following assistant rather than a raw text predictor.

**In-context learning.** Sufficiently large pre-trained models can do new tasks from a few examples in the prompt — no gradient updates. The mechanism is still actively researched; the practical fact is that prompt design is the dominant way teams adapt LLMs.

**Retrieval-augmented generation.** Embed a knowledge base; at query time, retrieve relevant chunks; concatenate into the prompt. Cheaper than fine-tuning, factually grounded, updatable. Standard architecture for "chat-with-your-docs" products.

**Tool use & agents.** LLMs can call external functions — search, code execution, API calls. Loop: prompt → tool call → result → next prompt. Powers everything from web-browsing assistants to coding agents.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### What LLMs do well

- Summarisation, translation, classification, extraction
- Code generation, refactoring, explanation
- Open-ended dialogue, drafting, brainstorming
- Structured outputs (JSON, code, function calls) with the right APIs
- Few-shot in-context learning on unseen tasks

</div>

<div class="no" markdown="1">

### Common failure modes

- Hallucination — confident statements that are simply wrong
- Stale knowledge — train cutoffs mean recent facts may be missing
- Reasoning gaps — multi-step logic, arithmetic, planning
- Prompt injection — adversarial inputs can hijack instructions
- Cost & latency — non-trivial for high-volume real-time use

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from anthropic import Anthropic

client = Anthropic()

# Few-shot in-context learning
prompt = """
Classify the sentiment of each sentence as POSITIVE, NEGATIVE, or NEUTRAL.

Q: "The movie was breathtaking and emotional."
A: POSITIVE

Q: "It was okay, nothing special."
A: NEUTRAL

Q: "I want my two hours back."
A:
""".strip()

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10,
    messages=[{"role": "user", "content": prompt}],
)
print(resp.content[0].text)
```

</div>

<div class="level-next">
<span>Want scaling laws, RLHF, MoE, & the modern stack?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Chinchilla scaling law</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`L(N, D)`achievable test loss given model size *N* and dataset size *D*

</li>
<li markdown="1">

`N`parameters, `D` training tokens

</li>
<li markdown="1">

`E`irreducible-error floor — the lowest possible loss

</li>
<li markdown="1">

`A, B, α, β`fit constants from empirical sweep

</li>
<li markdown="1">

Hoffmann et al. (2022): optimal *N* and *D* grow together — ~20 tokens per parameter

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; \text{floor} \;+\; \frac{A}{(\text{params})^\alpha} \;+\; \frac{B}{(\text{tokens})^\beta} $$</span>

**In words.** The model's loss has three parts: an irreducible **floor** you can never get below (truly random structure in language), a piece that shrinks as you add more **parameters**, and a piece that shrinks as you train on more **tokens**. The exponents `α` and `β` (alpha, beta — fitted constants) are both around 0.3, so doubling params or doubling tokens each cuts the corresponding piece by roughly the same factor. The practical upshot: when you have a fixed compute budget, the loss is lowest when you grow params and tokens *together* — roughly 20 training tokens per parameter.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`loss`cross-entropy loss the model achieves on held-out text

</li>
<li markdown="1">

`floor`the minimum possible loss — irreducible noise in language

</li>
<li markdown="1">

`params`number of model weights; `tokens` number of training tokens

</li>
<li markdown="1">

`α, β`scaling exponents — both ≈ 0.3 in practice

</li>
<li markdown="1">

Take-away: scale model and data together; ~20 tokens per parameter is the Chinchilla-optimal ratio

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Scaling laws.** Kaplan et al. (2020) showed cross-entropy loss is predictable as a power law in compute, params, and data. Hoffmann et al. (2022) corrected the optimal ratio — for a fixed compute budget, train a smaller model on more data than GPT-3 did. The Chinchilla curve is the rule of thumb most modern frontier models follow.

**Emergence.** Wei et al. (2022) — some abilities (3-digit arithmetic, chain-of-thought reasoning) appear sharply at specific scales. Schaeffer et al. (2023) — emergence is partly an artefact of the metrics used; smoother curves under different metrics. Both perspectives are true; the practical takeaway is to plot multiple scales when reporting capability.

**RLHF and alternatives.** Ouyang et al. (2022). Collect human preference pairs over model outputs; train a reward model; optimise the LLM against the reward via PPO. Modern alternatives — DPO (Rafailov et al. 2023), KTO, ORPO — skip the reward model and optimise directly on preferences. Simpler, often comparable.

**Mixture of Experts.** Instead of every token going through every parameter, route tokens to a few "experts". Sparse activation: 8× more parameters at 1× the compute. Used in Mixtral, Switch Transformer, GShard, GPT-4 (rumoured).

**Context length.** Vanilla attention is *O(n²)*. Modern frontier models support 100k–2M token contexts via Flash Attention (kernel-level optimisation), sparse / linear attention, position interpolation (RoPE), and architectural tricks (Mamba, ring attention).

**Inference optimisation.** Quantisation (4-bit, 8-bit), speculative decoding (draft + verify with a small model), KV-cache, FlashAttention. The difference between a model that's 50ms/token and 5s/token is mostly here.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a quantised model — runs on consumer GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,                   # bitsandbytes 4-bit quantisation
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Explain attention to a 10-year-old."},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids    = tok(prompt, return_tensors="pt").to(model.device)
out    = model.generate(**ids, max_new_tokens=200, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=True))
```

</div>

<div class="level-next">
<span>Want alignment, MoE routing, long context, and agentic systems?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">DPO objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\big(\beta \log \tfrac{\pi_\theta(y_w | x)}{\pi_\text{ref}(y_w | x)} - \beta \log \tfrac{\pi_\theta(y_l | x)}{\pi_\text{ref}(y_l | x)}\big)\right] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`x`prompt; `yw, yl` "winning" and "losing" responses from human preference data

</li>
<li markdown="1">

`πθ`current policy (the model being trained)

</li>
<li markdown="1">

`πref`frozen reference policy (usually the SFT model)

</li>
<li markdown="1">

`σ(·)`sigmoid function

</li>
<li markdown="1">

`β`temperature controlling how aggressively to depart from the reference

</li>
<li markdown="1">

No separate reward model, no RL — closed-form preference optimisation

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; -\,\text{avg}\!\left[\log \sigma\!\big(\beta \times \big[\log\tfrac{\text{policy(winner)}}{\text{ref(winner)}} \;-\; \log\tfrac{\text{policy(loser)}}{\text{ref(loser)}}\big]\big)\right] $$</span>

**In words.** For each human-labelled pair of responses (one preferred, one rejected), the loss pushes the model to put more probability on the **winner** and less on the **loser**, relative to a frozen reference model. The log-ratio `log(policy/ref)` measures how much each response's probability has shifted away from the reference. The `σ` (sigmoid) squashes the difference between the two shifts into a probability between 0 and 1; `β` (beta) controls how strongly to depart from the reference. Maximising `log σ(·)` is the same as maximising the probability that the model "agrees" with the human ranking — without needing a separate reward model or RL training loop.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`winner / loser`the preferred and rejected response in a human-labelled pair

</li>
<li markdown="1">

`policy`the model being trained; `ref` a frozen copy (usually the SFT model before preferences)

</li>
<li markdown="1">

`β`temperature — bigger β means more confident departures from the reference

</li>
<li markdown="1">

`σ`sigmoid — maps a real number to a probability between 0 and 1

</li>
<li markdown="1">

Simpler and more stable than RLHF for many applications

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Alignment.** Make the model do what users (and developers) want — instructions, safety, honesty, helpfulness. The toolkit: instruction tuning (SFT), preference optimisation (RLHF / DPO), constitutional AI (rule-based reward), red-teaming, system prompts. Active research area; the right combination is task-dependent.

**Mixture-of-Experts routing.** Each token gets routed to top-*k* experts (out of *N*) based on a learned router. Tradeoffs: load balancing (avoid degenerate routing), expert capacity (how many tokens fit per expert), router stability. Switch Transformer (Fedus et al. 2022) and Mixtral demonstrate the recipe at scale.

**Long context.** Position encoding extensions (RoPE scaling, ALiBi), attention modifications (Flash Attention 2/3, ring attention), state-space models (Mamba). Reading 1M-token context is technically feasible now; whether the model uses it well is another matter ("needle in haystack" evals).

**Agentic systems.** LLMs as the reasoning core of a multi-step process: plan → call tools → observe → iterate. ReAct, Tree of Thoughts, AutoGPT, BabyAGI. The frontier is making these reliable enough for production — currently a brittle research area but improving fast.

**Multi-modal LLMs.** Image, audio, video tokens fed into the same model. CLIP-style encoders for input; sometimes separate decoders for output. GPT-4V, Gemini, Claude 3+ all support image input; some support image / audio output.

**Synthetic data.** Frontier models are increasingly trained on data generated by other LLMs — distillation, self-instruction, synthetic Q-A. Risks: model collapse from training on slop, contamination of benchmarks. Powerful when curated carefully.

**Watermarking and provenance.** Active research into making model outputs identifiable as machine-generated (statistical fingerprints in the sampling distribution). Important for misinformation, attribution, and benchmark integrity.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn.functional as F

# Direct Preference Optimization (DPO) — preference learning without RL
def dpo_loss(policy_logp_chosen, policy_logp_rejected,
             ref_logp_chosen,    ref_logp_rejected, beta=0.1):
    pi_logratio  = policy_logp_chosen - policy_logp_rejected
    ref_logratio = ref_logp_chosen    - ref_logp_rejected
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()

# Speculative decoding — draft + verify
def speculative_generate(draft_model, target_model, prompt, n_draft=4):
    """Draft generates `n_draft` tokens; target verifies them in one pass."""
    ids = prompt
    while len(ids) < max_length:
        draft = draft_model.generate(ids, max_new_tokens=n_draft)
        target_probs = target_model(draft).softmax(-1)
        # Accept tokens where target agrees; resample at the rejection point
        ids = verify_and_resample(draft, target_probs)
    return ids
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

[Hoffmann et al. (2022) — Chinchilla <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2203.15556){: target="_blank" }
<span class="annotation">The paper that corrected GPT-3's compute optimal scaling. The ~20-tokens-per-parameter rule of thumb. Required reading.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Ouyang et al. (2022) — InstructGPT <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2203.02155){: target="_blank" }
<span class="annotation">The RLHF paper that turned GPT-3 into something users would actually want to talk to.</span>

</li>
<li data-tier="indepth" markdown="1">

[Rafailov et al. (2023) — DPO <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2305.18290){: target="_blank" }
<span class="annotation">Direct Preference Optimisation — preference learning without RL. Simpler than RLHF; comparable results.</span>

</li>
<li data-tier="intuition" markdown="1">

[Karpathy — Zero to Hero <i class="fas fa-external-link-alt"></i>](https://karpathy.ai/zero-to-hero.html){: target="_blank" }
<span class="annotation">Karpathy's videos building GPT from scratch. The best hands-on introduction to LLM internals.</span>

</li>
</ul>

</div>

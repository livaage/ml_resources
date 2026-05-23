---
title: Natural Language Processing — ML Resources Hub
eyebrow_text: ← Theory · Applications
eyebrow_href: ../theory.html
heading: Natural Language Processing
lead: How models read, write, translate, summarise, classify — the path from one-hot tokens to GPT.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Words become vectors; sentences become sequences of vectors; models work on those.** The whole stack — tokenisation, embedding, attention, decoding — is about turning fuzzy human-written language into something a model can compute on, then back. Modern NLP is almost entirely transformer-based now, but the older pieces still show up everywhere.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Type a sentence — watch byte-pair tokenisation chunk it and embed each token as a vector</span>
</div>
<div class="viz-classic-controls">
<input id="viz-nlp-text" type="text" value="the quick brown fox jumps over the lazy dog" style="flex:1; min-width: 200px; padding: 0.4rem 0.7rem; border: 1px solid var(--hairline);
                          border-radius: 6px; font-family: var(--font-mono); font-size: 0.85rem;"></input>
<button id="viz-nlp-reset" type="button">Reset</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-nlp-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-nlp-caption"></div>
</div>

<script src="{{root}}js/viz/nlp.js"></script>

A toy tokeniser splits your text into sub-word chunks (some common words stay whole, rarer ones break up). Each token becomes a vector — visualised here as a coloured bar pattern. Real models use 768- or 4096-dim vectors; this is a sketch. The structural point is that *tokens*, not characters or whole words, are what models work with.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Tokenisation.** Split text into reusable chunks. Modern: byte-pair encoding (BPE), WordPiece, SentencePiece — vocabularies of ~30k–100k sub-word units. Common words stay whole; rare words decompose. Crucial detail: tokens, not characters, are the model's atoms.

**Embeddings.** Each token → a learned vector. Same dimension for everything. Vectors with similar meanings end up geometrically close — but the magic is in *relative* structure: "king − man + woman ≈ queen" works in word2vec.

**Transformers.** The current standard. Self-attention lets each token attend to every other, no recurrence. Scales to long sequences (with the right tricks) and trains in parallel. See the [Transformer page](neural-networks/transformer.html) for details.

**Decoder, encoder, or both.** BERT is encoder-only (good for classification, NER). GPT is decoder-only (good for generation). T5 and BART are encoder-decoder (good for translation, summarisation). Modern frontier is mostly decoder-only.

**The fine-tuning stack.** Pre-train on huge unlabeled text → supervised fine-tune on instruction-following pairs → RLHF for alignment. The recipe that turned GPT-3 into ChatGPT.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Classical NLP tasks

- **Classification**: sentiment, topic, spam — fine-tune a BERT-like
- **NER**: tag spans (people, places, products)
- **Translation / summarisation**: T5, BART, or just GPT-4
- **Question answering**: retrieve relevant docs, then read with an LLM
- **Embedding for search**: sentence-transformers, BGE, E5

</div>

<div class="no" markdown="1">

### Watch out

- Tokenisation quirks bite you in unexpected ways (numbers, code, multi-byte chars)
- Context windows are finite — long-doc reading needs chunking
- LLM hallucination is real — pair with retrieval for factual tasks
- Cross-lingual transfer is uneven; resource-poor languages still suffer

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Tokeniser handles every detail of text → token IDs
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tok("the quick brown fox", return_tensors="pt", padding=True, truncation=True)
print(enc.input_ids)            # tensor of subword IDs
print(tok.convert_ids_to_tokens(enc.input_ids[0]))

# Fine-tune for classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
out   = model(**enc, labels=torch.tensor([1]))
out.loss.backward()
```

</div>

<div class="level-next">
<span>Want word embeddings, sub-word tokenisation, and modern architectures?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Cross-entropy on a vocabulary</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L} = -\sum_{t=1}^{T} \log p(w_t \mid w_{1:t-1}) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`wt`the actual token at position *t*

</li>
<li markdown="1">

`w1:t-1`all tokens that came before *t*

</li>
<li markdown="1">

`p(wt | …)`softmax over the vocabulary at position *t*

</li>
<li markdown="1">

Next-token prediction — the universal NLP objective

</li>
<li markdown="1">

Pre-training is exactly this over trillions of tokens

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; -\sum_{t=1}^{T} \log (\text{model's predicted probability of the actual next token}) $$</span>

**In words.** For each position *t* in the training text, the model produces a probability over every word in its vocabulary for what comes next. You look at what the actual next token was, take its predicted probability, and the loss is the negative log of that probability — summed across every position. The `Σ` just means "add up across all positions *t = 1 to T*". The negative log is small when the probability is close to 1 (a confident, correct prediction) and huge when the probability is close to 0 (the model thought the actual word was unlikely). Minimising it makes the model assign higher probability to actual continuations.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`loss`scalar to minimise; lower means better next-token predictions

</li>
<li markdown="1">

`predicted probability`what the model assigns to the actual next token, given everything before

</li>
<li markdown="1">

`log`the natural logarithm — turns multiplied probabilities into added log-probabilities

</li>
<li markdown="1">

This same objective drives pre-training across trillions of tokens for every modern LLM

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Word embeddings (word2vec, GloVe).** Classical: learn one vector per word from co-occurrence statistics or a shallow predictive task. Famous for the analogy structure (king − man + woman ≈ queen). Now subsumed by contextual embeddings from transformers, but the geometric idea persists.

**Contextual embeddings.** ELMo, BERT, RoBERTa — the same word gets different vectors depending on context. "Bank" in "river bank" vs "savings bank" gets different embeddings. Pre-training task: masked token prediction or next-token prediction.

**Subword tokenisation.** BPE (used by GPT) merges most-frequent byte pairs iteratively. WordPiece (BERT) does similar but with a likelihood criterion. SentencePiece treats text as a raw byte stream — language-agnostic. Vocabulary typically 32k–100k. Affects model performance, training speed, and downstream task design.

**Sequence-to-sequence.** Encoder reads input; decoder generates output token-by-token, attending to the encoder's output. Translation, summarisation, dialogue, code generation — all variants.

**Causal vs masked attention.** Causal (decoder-only, GPT): each position only attends to past positions. Masked (encoder-only, BERT): every position sees every other. Different trade-offs for generation vs comprehension; modern frontier favours causal.

**Decoding strategies.** Greedy: pick the argmax. Beam search: keep top-*k* hypotheses. Sampling: temperature, top-*k*, top-*p* (nucleus). Each gives different generation quality; sampling with low temperature is the modern default for assistant-style models.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = "The future of AI is"
ids = tok(prompt, return_tensors="pt").input_ids

# Generate with sampling — top-p (nucleus) decoding
out = model.generate(
    ids,
    do_sample=True,
    temperature=0.8,
    top_p=0.92,
    max_new_tokens=50,
)
print(tok.decode(out[0]))
```

</div>

<div class="level-next">
<span>Want RAG, long-context tricks, multilingual, and structured outputs?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Retrieval-augmented generation</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ p(y \mid x) = \sum_{z \in \mathrm{top}_k(x)} p_\eta(z \mid x) \, p_\theta(y \mid x, z) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`x`user query; `y` generated answer

</li>
<li markdown="1">

`z`retrieved documents; `topk(x)` the *k* documents most relevant to *x*

</li>
<li markdown="1">

`pη(z | x)`retriever's score for document *z* given query *x*

</li>
<li markdown="1">

`pθ(y | x, z)`generator's probability of answer *y* given query and retrieved context

</li>
<li markdown="1">

Decouples knowledge from parameters; fresh facts via a vector store

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ p(\text{answer} \mid \text{query}) \;=\; \sum_{\text{doc} \in \text{top-}k} p(\text{doc} \mid \text{query}) \,\times\, p(\text{answer} \mid \text{query}, \text{doc}) $$</span>

**In words.** The probability of an answer is a weighted sum over the top-*k* retrieved documents. For each candidate document, you multiply two pieces: how relevant the **retriever** thinks the document is to the query, and how likely the **generator** would be to produce the answer given both the query and that document. Add those products up across all *k* retrieved documents (that's what the `Σ` does) and you get the model's overall probability for the answer. In practice the retriever is usually a vector-search system over a knowledge base, and the generator is an LLM — the retrieved chunks just get pasted into the prompt.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`query`the user's question or input

</li>
<li markdown="1">

`doc`a retrieved document chunk

</li>
<li markdown="1">

`top-k`the *k* most relevant documents under the retriever's similarity score

</li>
<li markdown="1">

`p(doc | query)`retriever score: how relevant the document is to the query

</li>
<li markdown="1">

`p(answer | query, doc)`generator: probability of the answer given query plus that document as context

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Retrieval-augmented generation (RAG).** Embed your knowledge base; at query time retrieve top-*k* relevant chunks; concatenate with the prompt; let the LLM answer. Cheaper than fine-tuning, factually grounded, and updatable. The standard production architecture for "chat-with-your-docs" applications.

**Long-context architectures.** Vanilla attention is O(n²) in sequence length — prohibitive past ~8k tokens. Modern approaches: sparse attention (Longformer, BigBird), linear attention (Linear Transformer, Performer), state-space models (Mamba), retrieval-augmented memory (Memorising Transformer). Most production long-context models combine several tricks.

**Tool use.** LLMs that can call external functions: search, code execution, calculators, web browsers, retrieval. The agent loop: prompt → function call → result → prompt with result. Modern frameworks (LangChain, LlamaIndex, OpenAI function calling) standardise this.

**Instruction tuning & RLHF.** Fine-tune a pre-trained LLM on (instruction, response) pairs (SFT), then optimise against a reward model trained from human preferences (RLHF) or against rule-based rewards (RLAIF, Constitutional AI). The reason GPT-3 was a curiosity and GPT-3.5 became a product.

**Multilingual.** Most modern foundation models are heavily English-skewed but include some multilingual capability via the tokeniser's byte fallback. Performance drops sharply on low-resource languages; specialised models (mT5, NLLB, BLOOM) help.

**Structured outputs.** Force the LLM to produce valid JSON, code, or other formats by constraining the decoding step (grammar-constrained generation, JSON mode, JSONSchema). Crucial for production reliability; OpenAI, Anthropic, and others ship native support.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Build a simple RAG retriever
encoder = SentenceTransformer("all-MiniLM-L6-v2")
docs    = ["Paris is the capital of France.",
           "The Eiffel Tower was built in 1889.",
           "Croissants are flaky French pastries.",
           ...]
doc_emb = encoder.encode(docs)

def retrieve(query, k=3):
    q = encoder.encode(query)
    sims = doc_emb @ q
    return [docs[i] for i in sims.argsort()[::-1][:k]]

# Then call the LLM with retrieved context:
context = "\n".join(retrieve("when was the Eiffel Tower built?"))
prompt  = f"Use this context:\n{context}\nQ: when was the Eiffel Tower built?\nA:"
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

[Stanford CS224N — NLP with Deep Learning <i class="fas fa-external-link-alt"></i>](https://web.stanford.edu/class/cs224n/){: target="_blank" }
<span class="annotation">Manning's course. Slides and lectures online; the canonical modern NLP curriculum.</span>

</li>
<li data-tier="intuition" markdown="1">

[Alammar — The Illustrated BERT, ELMo & co. <i class="fas fa-external-link-alt"></i>](https://jalammar.github.io/illustrated-bert/){: target="_blank" }
<span class="annotation">Beautiful visual essays on each modern NLP architecture. Best starting point for the transformer family.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[HuggingFace NLP Course <i class="fas fa-external-link-alt"></i>](https://huggingface.co/learn/nlp-course){: target="_blank" }
<span class="annotation">Practical, code-first introduction to modern NLP. Covers tokenisation, fine-tuning, deployment.</span>

</li>
<li data-tier="indepth" markdown="1">

[Lewis et al. (2020) — RAG <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2005.11401){: target="_blank" }
<span class="annotation">The original RAG paper. The architecture has since become a production default for knowledge-grounded chat.</span>

</li>
</ul>

</div>

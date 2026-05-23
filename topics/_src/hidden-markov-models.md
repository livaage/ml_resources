---
title: Hidden Markov Models — ML Resources Hub
eyebrow_text: ← Theory · Probabilistic Models
eyebrow_href: ../theory.html
heading: Hidden Markov Models
lead: Latent-state sequence models — what RNNs replaced, but still useful when interpretability matters.
prev_href: gaussian-mixture-models.html
prev_title: Gaussian Mixture Models
next_href: bayesian-inference.html
next_title: Bayesian Inference
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**A system you can't see directly, but you can see its outputs.** The state changes over time according to its own rules. You only observe noisy emissions. Given the emissions, you infer the hidden states.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Click an activity icon to flip it — the hidden-state posterior and Viterbi path re-shuffle to explain the new evidence</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Sequence
                <select id="viz-hmm-preset"></select>
</label>
<button id="viz-hmm-reset" type="button">Reset</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-hmm-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-hmm-caption"></div>
</div>

<script src="{{root}}js/viz/hmm.js"></script>

The toy model has three hidden weather states (Sunny / Cloudy / Rainy) that you never observe directly — only the activity I picked that day (Walk / Shop / Clean). Each state has its own emission preference, and they tend to persist (Sunny is more likely to stay Sunny than flip straight to Rainy). The three colour-tinted rows show the forward-backward posterior: how strongly the model believes each state was active given the *entire* sequence. The dark line is the Viterbi *most-likely path* — the single best explanation rather than per-step marginals.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Classic example: speech recognition. The "hidden state" is the phoneme being spoken; the "observation" is the audio signal. You don't see the phoneme directly, only the sound. The model captures (a) which phonemes tend to follow which (transition probabilities) and (b) what each phoneme tends to sound like (emission probabilities).

HMMs have been largely replaced by RNNs and transformers in speech and NLP, but they're still the right tool when (a) the latent states have a real interpretation you care about, (b) data is small, or (c) you want exact inference rather than approximate.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- The hidden states correspond to real-world categories you care about
- Small data — exact inference is feasible and helps
- Biological sequence analysis (gene finding, protein structure)
- You need a probabilistic story you can interrogate

</div>

<div class="no" markdown="1">

### Skip it when

- Lots of data and you don't care about latent-state interpretability — use RNN / transformer
- State transitions depend on long-range context (Markov assumption breaks)
- Emissions are high-dimensional images or text
- You want end-to-end learning of representations

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from hmmlearn import hmm

# 3 hidden states, Gaussian emissions
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.fit(observations)   # shape: (n_samples, n_features)

# Decode: most likely hidden-state sequence (Viterbi)
hidden_states = model.predict(observations)

# Score: log-likelihood of the observed sequence under the model
print(f"log p(obs) = {model.score(observations):.2f}")
```

</div>

<div class="level-next">
<span>Want the three classical problems and the algorithms?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The model</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}) \;=\; \pi(z_1)\,\prod_{t=2}^{T} A(z_t \mid z_{t-1}) \,\prod_{t=1}^{T} B(x_t \mid z_t) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`π`initial state distribution

</li>
<li markdown="1">

`A`state transition matrix

</li>
<li markdown="1">

`B`emission distribution (Gaussian, multinomial, …)

</li>
<li markdown="1">

`zt`hidden state at time *t*

</li>
<li markdown="1">

`xt`observed emission at time *t*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ p(\text{observations}, \text{hidden states}) \;=\; \underbrace{P(z_1)}_{\text{start}} \;\times\; \prod_t \underbrace{P(z_t \mid z_{t-1})}_{\text{transition}} \;\times\; \prod_t \underbrace{P(x_t \mid z_t)}_{\text{emission}} $$</span>

**In words.** The probability of seeing a particular sequence of *observations* together with a particular sequence of *hidden states* factors into three pieces. Start with `π` (pi) — the probability of whatever hidden state you happen to begin in. Then for every following step, multiply in the **transition** probability `A(zt | zt-1)` — how likely it is to move to the new hidden state given the previous one. Finally, for every step, multiply in the **emission** probability `B(xt | zt)` — how likely the observation you saw is given the hidden state you were in. The two big products (`∏`) just say "multiply across all time steps." The whole expression is built on the *Markov assumption*: the future depends on the present only, not the past.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`observations`what you actually see, *x*<sub>1</sub>, *x*<sub>2</sub>, …, *x*<sub>T</sub>

</li>
<li markdown="1">

`hidden states`what's really going on underneath, *z*<sub>1</sub>, *z*<sub>2</sub>, …, *z*<sub>T</sub>

</li>
<li markdown="1">

`start`π — initial state distribution

</li>
<li markdown="1">

`transition`A — probability of moving from one hidden state to the next

</li>
<li markdown="1">

`emission`B — probability of producing each observation from a hidden state

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

Three classical problems, each with its own algorithm:

**1. Evaluation** — given a model, how likely is an observed sequence? Use the *forward algorithm*. Dynamic programming, O(*T·K²*) where *K* is the number of states.

**2. Decoding** — given a model and observations, what's the most likely hidden state sequence? Use the *Viterbi algorithm*. Same complexity as forward, but tracks the argmax path.

**3. Learning** — given observations only, fit the model parameters. Use the *Baum-Welch algorithm* (EM specialized to HMMs). Iterate: E-step computes posterior over latent states via forward-backward; M-step updates π, *A*, *B*.

Without the Markov assumption, all of these would be exponential in *T*. The local conditioning is what makes HMMs tractable.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Tagging tasks where labels follow a Markov chain (POS tagging, NER)
- Acoustic modelling for small-vocabulary speech
- Bioinformatics (CpG islands, gene structure, profile HMMs)
- Activity recognition from sensor sequences

</div>

<div class="no" markdown="1">

### Skip it when

- Long-range dependencies — RNN / transformer captures them better
- Continuous high-dim emissions — better handled by deep generative models
- You need to learn the state space (not just its parameters)
- Labels are observed at every step — use CRF / sequence tagger instead

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from hmmlearn import hmm
import numpy as np

# Multinomial emissions for discrete observations
model = hmm.CategoricalHMM(n_components=3, n_iter=100, random_state=0)
model.fit(X, lengths=lengths)    # lengths splits concatenated sequences

# Posterior probability of each state at each timestep
log_prob, posteriors = model.score_samples(X)

# Viterbi: most likely state sequence
states = model.predict(X)

print("Transition matrix:")
print(np.round(model.transmat_, 3))
```

</div>

<div class="level-next">
<span>Want forward-backward, max-product, and the CRF connection?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Forward recursion</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \alpha_t(j) \;=\; B(x_t \mid z_t{=}j) \;\sum_{i=1}^{K} \alpha_{t-1}(i)\, A(i \to j) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`αt(j)`joint probability of observations 1..t and hidden state *j* at *t*

</li>
<li markdown="1">

Combined with the analogous **backward** recursion *β<sub>t</sub>(j)*, you get the full posterior *γ<sub>t</sub>(j)* at each step

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \alpha_t(j) \;=\; P(\text{observation at } t \mid \text{state } j) \;\times\; \sum_{i} \alpha_{t-1}(i) \cdot P(\text{move } i \to j) $$</span>

**In words.** `αt(j)` (alpha — Greek letter used here as a name for this quantity) is the joint probability of having seen observations 1 through *t* *and* being in hidden state *j* at time *t*. You build it step by step. To get to state *j* at time *t*, you must have been in *some* state *i* at the previous step — so the `Σ` ("sum over all possible previous states *i*") adds up every way of arriving. Multiply by the transition probability of that move, then by the emission probability of seeing `xt` given you're now in state *j*. Running this recursion across *t = 1, 2, …, T* costs O(*T·K²*) — much cheaper than the exponential cost of enumerating every hidden path. The analogous *backward* recursion sweeps right-to-left, and together they give you the per-step posterior over states.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`αt(j)`probability of the first *t* observations *and* ending in state *j* at time *t*

</li>
<li markdown="1">

`K`number of hidden states

</li>
<li markdown="1">

`i`candidate previous hidden state — the sum runs over all of them

</li>
<li markdown="1">

`move i → j`transition from state *i* to state *j* (one entry of the matrix *A*)

</li>
<li markdown="1">

Cost is O(*T·K²*): for each time step, for each pair of states

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Sum-product vs max-product.** Forward-backward and Viterbi are the same dynamic program with different semiring operations: forward sums over paths to compute *p(x)*; Viterbi maxes over paths to find the best one. Both are O(*T·K²*) and both run in log-space in practice to avoid underflow.

**Baum-Welch.** The EM algorithm for HMMs. The E-step uses forward-backward to compute expected state and transition counts; the M-step normalizes those counts to update π, *A*, and the emission parameters. Local optima are common; warm-start with k-means or supervised pre-training if you have it.

**HMMs vs. CRFs.** HMMs are generative (model *p(x, z)*); CRFs are discriminative (model *p(z | x)* directly). CRFs win when you can engineer good features over the observations *and* have labels for training. HMMs win when labels are scarce or you want to do unsupervised structure discovery.

**Generalizations.** HSMMs (hidden semi-Markov) allow non-geometric state durations. Factorial HMMs decompose the state into multiple chains. Input-output HMMs condition transitions and emissions on external inputs. Beyond that, you've reinvented an RNN — at which point you probably should just use one.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You need exact inference and have computational budget for it
- Latent structure interpretation is the goal (e.g. regime detection)
- Probabilistic outputs feed downstream Bayesian computation
- Domain knowledge dictates a specific transition structure

</div>

<div class="no" markdown="1">

### Skip it when

- You can afford to label sequences — train a sequence tagger / transformer instead
- The state space is unknown and continuous
- You need to model variable-rate sequences with strong dependencies
- You want to backprop through the model end-to-end (use a differentiable variant)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

def forward_log(log_pi, log_A, log_B):
    """Forward in log-space. log_B has shape (T, K) — log emission probs."""
    T, K = log_B.shape
    log_alpha = np.empty_like(log_B)
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        # logsumexp trick to stay numerically stable
        log_alpha[t] = log_B[t] + _logsumexp(log_alpha[t-1, :, None] + log_A, axis=0)
    return log_alpha

def _logsumexp(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)
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

[Rabiner (1989) tutorial <i class="fas fa-external-link-alt"></i>](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf){: target="_blank" }
<span class="annotation">The classic tutorial on HMMs. Covers the three problems, all algorithms, applications to speech. Still the best single source.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[hmmlearn documentation <i class="fas fa-external-link-alt"></i>](https://hmmlearn.readthedocs.io/){: target="_blank" }
<span class="annotation">The maintained scikit-learn-style Python implementation. Categorical, Gaussian, and GMM emissions.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Jurafsky & Martin, appendix A <i class="fas fa-external-link-alt"></i>](https://web.stanford.edu/~jurafsky/slp3/A.pdf){: target="_blank" }
<span class="annotation">HMMs for NLP with worked examples (POS tagging). Cleaner notation than Rabiner.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Wikipedia — Viterbi algorithm <i class="fas fa-external-link-alt"></i>](https://en.wikipedia.org/wiki/Viterbi_algorithm){: target="_blank" }
<span class="annotation">Quick reference for the decoding step in isolation. Includes pseudocode.</span>

</li>
</ul>

</div>

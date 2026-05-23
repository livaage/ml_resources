---
title: Learning Paradigms — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Learning Paradigms
lead: Supervised, unsupervised, self-supervised, reinforcement — what each one "sees" during training, and when each one is the right framing.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The paradigm is defined by what the model gets to see.** Supervised has (x, y) pairs. Unsupervised has just x. Self-supervised invents y from x. Reinforcement learning only sees a reward signal. Each tells a model "here's the kind of feedback you'll get" — and that shapes everything else.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Toggle each paradigm — same data, different things the learner gets to see</span>
</div>
<div class="viz-classic-controls">
<button id="viz-lt-sup" type="button" class="active">Supervised</button>
<button id="viz-lt-semi" type="button">Semi-supervised</button>
<button id="viz-lt-self" type="button">Self-supervised</button>
<button id="viz-lt-unsup" type="button">Unsupervised</button>
<button id="viz-lt-rl" type="button">Reinforcement</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-lt-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-lt-caption"></div>
</div>

<script src="{{root}}js/viz/learning-types.js"></script>

The same underlying dataset is presented five ways. Notice how much is "given" to the learner under each paradigm — full labels, partial labels, structure that lets you make up labels (predict the right half from the left half), nothing but the points themselves, or only a numerical reward for an action.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Supervised learning.** Each training example is a pair *(x, y)*. The model learns a function that maps *x* to *y*. Classification (discrete *y*) and regression (continuous *y*) are the two halves. Most "applied ML" problems start here because labels are expensive but tractable.

**Unsupervised learning.** Only *x* — no labels. The model has to find structure on its own: clusters, lower-dimensional manifolds, densities, anomalies. Useful when labels are unavailable or when you want to understand the data before predicting.

**Self-supervised learning.** Labels invented from the input itself: predict the next token from past tokens, predict the masked patch from the visible ones, contrast augmented views of the same image. Powers most modern foundation models — labels are essentially free, and the resulting representations transfer beautifully.

**Semi-supervised learning.** Mostly unlabelled *x*, with a small labelled subset. Often the realistic setting in industry: labels are expensive, unlabelled data is everywhere. Pseudo-labelling, consistency training, and pre-training-then-fine-tuning are the dominant strategies.

**Reinforcement learning.** An agent acts in an environment and receives rewards. No labels — just feedback on whether its actions are working. Used for control (robotics), strategy (games), and increasingly for aligning language models to human preferences.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### When each shines

- **Supervised**: clean labels, the prediction is the product
- **Self-supervised**: huge unlabelled corpora, foundation models
- **Semi-supervised**: small labelled budget, large unlabelled pool
- **Unsupervised**: exploration, segmentation, anomalies
- **RL**: sequential decisions, evaluative feedback only

</div>

<div class="no" markdown="1">

### Where each struggles

- Supervised — needs lots of clean labels
- Self-supervised — pre-training is expensive, distillation is hard
- Semi-supervised — pseudo-label errors can compound
- Unsupervised — evaluation is awkward (what's "right"?)
- RL — sample-inefficient, unstable training, reward hacking

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
# The five paradigms in one breath.

# Supervised — labelled (X, y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Unsupervised — only X
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=3).fit_predict(X)

# Semi-supervised — mostly unlabelled, a few labels
from sklearn.semi_supervised import SelfTrainingClassifier
clf = SelfTrainingClassifier(base_classifier)
clf.fit(X_mixed, y_mixed)         # -1 marks unlabelled

# Self-supervised — labels from the input itself (next-token prediction)
loss = F.cross_entropy(model(tokens[:, :-1]), tokens[:, 1:])

# Reinforcement learning — reward-driven
state, info = env.reset()
for t in range(T):
    action = policy(state)
    state, reward, done, _, _ = env.step(action)
    policy.update(state, action, reward)
```

</div>

<div class="level-next">
<span>Want the formalisms and modern blends?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Empirical risk minimisation</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \hat\theta = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n \ell\!\big(f_\theta(x_i),\, y_i\big) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

The unifying formalism of supervised learning

</li>
<li markdown="1">

*y* can be a true label (supervised), invented from *x* (self-supervised), or partially observed (semi-supervised)

</li>
<li markdown="1">

RL replaces this with a Bellman equation

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{best parameters} \;=\; \text{the } \theta \text{ that minimises the average loss across all training examples} $$</span>

**In words.** "Empirical risk minimisation" just means: try lots of parameter values `θ`, and pick the one that makes the model's *average* mistake on the training set as small as possible. The `arg min` notation means "the value of `θ` that minimises what follows"; the sum-divided-by-`n` is just the average. The function `ℓ` (lowercase L, called the loss) measures how wrong each prediction is. This single formula unifies most paradigms — only the source of *y* changes (true labels, invented labels, or partial labels).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`best params`the parameter values you'll end up using

</li>
<li markdown="1">

`average loss`mean of the model's error across all *n* training examples

</li>
<li markdown="1">

`ℓ`the loss function (squared error, cross-entropy, etc.)

</li>
<li markdown="1">

*y* can be a true label (supervised), invented from *x* (self-supervised), or partially observed (semi-supervised); RL replaces this with a Bellman equation

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Beyond the textbook categories.** The boundaries are fuzzy and mostly historical. A self-supervised model is "supervised" once you invent the labels; a clustering model is "supervised" if you have one labelled example per cluster; pre-training-then-fine-tuning blurs every category. The useful question is what feedback the model gets, when.

**Foundation models & pre-training.** A modern recipe: pre-train a huge model on huge data with a self-supervised objective (next-token prediction, masked image modelling, contrastive learning), then fine-tune on a smaller labelled dataset for a specific task. The pre-training distils general structure; the fine-tuning specialises. Drove the GPT / BERT / DINO families.

**Multi-task and meta-learning.** Multi-task: one model, many tasks; share the early layers, specialise the heads. Meta-learning: "learn to learn" — the training data is a distribution over tasks, and the model learns to adapt quickly to a new one (MAML, ProtoNets). Useful when labels are scarce *per task* but many similar tasks exist.

**Transfer learning & domain adaptation.** Train on one distribution, deploy on another. Most successes in deep learning rely on this — ImageNet pre-training, BERT pre-training, etc. The risk is distribution shift: features useful in the source domain may not transfer.

**Active learning.** The model chooses what to label next — query the most informative example, get a human to label it, repeat. Useful when labels are expensive (medical imaging, expert annotation). Choosing the right "informativeness" criterion is the art.

**Imitation learning.** Learn a policy from expert demonstrations. Avoids RL's exploration problem but requires expert data. Behavior cloning is the simplest form; DAgger, inverse RL, and offline RL are more sophisticated.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn

# Pre-train, then fine-tune — the canonical modern recipe
encoder = nn.Sequential(*backbone_layers)
ssl_head = SSLHead()

# Phase 1: self-supervised pre-training
ssl_loss = contrastive_loss(ssl_head(encoder(x)), positives)
ssl_loss.backward(); ssl_opt.step()

# Phase 2: supervised fine-tuning
encoder.requires_grad_(True)
classifier = nn.Linear(d_model, n_classes)
loss = F.cross_entropy(classifier(encoder(x)), y)
loss.backward(); ft_opt.step()
```

</div>

<div class="level-next">
<span>Want the MDP formalism, offline RL, and weak supervision?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Bellman equation (RL)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ V^\pi(s) = \mathbb{E}_{a \sim \pi}\!\left[ R(s, a) + \gamma\, \mathbb{E}_{s' \sim P}\,V^\pi(s') \right] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Vπ`value of state *s* under policy *π*

</li>
<li markdown="1">

`γ`discount factor (1 = caring about the far future)

</li>
<li markdown="1">

The optimisation problem is over *policies*, not pointwise predictions

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{value of state } s \;=\; \text{expected reward now} \;+\; \gamma \times \text{expected value of next state} $$</span>

**In words.** The "value" of being in a state is the reward you expect to collect by acting from there onward. That breaks recursively into two pieces: the immediate reward from the action your policy `π` picks now, plus the value of wherever you land next, weighted by the discount factor `γ` (gamma, a number between 0 and 1 — closer to 1 means caring more about long-term reward). The *E* symbols are averages: over the actions your policy might take, and over which next state the environment lands you in. Solve this recursive equation and you've solved the RL problem (in principle).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`value`expected total future reward from this state

</li>
<li markdown="1">

`γ`discount factor — closer to 1 means caring about the far future

</li>
<li markdown="1">

`policy π`your rule for choosing actions in each state

</li>
<li markdown="1">

`expected`averaged over the randomness in the policy's actions and the environment's transitions

</li>
<li markdown="1">

The optimisation problem is over *policies*, not pointwise predictions

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**The MDP formalism.** RL studies Markov Decision Processes: states *S*, actions *A*, transitions *P(s' | s, a)*, rewards *R(s, a)*, discount *γ*. The goal is a policy *π(a | s)* that maximises expected discounted reward. Value iteration / policy iteration are the textbook algorithms; modern deep RL approximates the value function or policy with a neural network.

**On-policy vs off-policy.** On-policy (PPO, A2C): learn from data collected by the current policy. Off-policy (Q-learning, SAC, DDPG): learn from any policy's data (replay buffer). Off-policy is more sample-efficient; on-policy is more stable.

**RLHF.** Reinforcement Learning from Human Feedback. Train a reward model from human preferences, then optimise a language model against that reward with PPO. Made instruction-following LLMs possible. The classic three-phase recipe: pre-train → SFT → RLHF.

**Offline RL.** Train an RL agent from a fixed dataset without environment access. Hard because the agent can't recover from bad action selection. Conservative Q-Learning (CQL), Implicit Q-Learning (IQL), Behavior-Regularised Actor-Critic are recent strong approaches.

**Weak supervision.** Instead of clean labels, use multiple noisy, conflicting, or partial labelling functions. Snorkel-style systems combine them via a generative model. Useful when expert labels are unaffordable but heuristics exist.

**Self-training & pseudo-labels.** Train on labelled data, predict on unlabelled, add confident predictions to the training set, iterate. Works surprisingly well in practice — but errors compound, so confidence thresholds and consistency checks are essential.

**Curriculum learning.** Order training examples from easy to hard. Sometimes accelerates convergence dramatically; sometimes does nothing. The "easy first" intuition is solid; the specific curriculum is often domain-specific art.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn.functional as F

# Conservative Q-Learning for offline RL — penalise OOD actions
def cql_loss(q_net, batch, alpha=1.0):
    s, a, r, s_next, done = batch
    q_sa = q_net(s, a)
    with torch.no_grad():
        q_target = r + (1 - done) * gamma * q_net.target(s_next).max(-1).values
    td_loss = F.mse_loss(q_sa, q_target)

    # CQL term: pull down Q for random actions, push up for dataset actions
    q_random  = q_net(s, sample_random_actions(s))
    cql_term  = (q_random.logsumexp(-1) - q_sa).mean()

    return td_loss + alpha * cql_term

# Snorkel-style weak supervision — multiple labelling functions vote
def majority_vote(lf_outputs):
    # lf_outputs: (N, K) where K = number of labelling functions
    valid = lf_outputs != -1                          # -1 means abstain
    votes = (lf_outputs * valid).sum(axis=1)
    return votes / valid.sum(axis=1).clip(min=1)
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

[Elements of Statistical Learning — Ch. 1–2 <i class="fas fa-external-link-alt"></i>](https://web.stanford.edu/~hastie/Papers/ESLII.pdf){: target="_blank" }
<span class="annotation">Hastie, Tibshirani &amp; Friedman lay out the supervised / unsupervised split with classical examples. Free PDF.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Sutton & Barto — Reinforcement Learning: An Introduction <i class="fas fa-external-link-alt"></i>](http://incompleteideas.net/book/the-book-2nd.html){: target="_blank" }
<span class="annotation">The standard RL textbook. Free PDF online. Chapters 1–6 are essential.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[LeCun & Misra — Self-Supervised: The Dark Matter of Intelligence <i class="fas fa-external-link-alt"></i>](https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/){: target="_blank" }
<span class="annotation">Manifesto for why self-supervision is the right framing for foundation models. Worth reading even if you disagree.</span>

</li>
<li data-tier="indepth" markdown="1">

[Lilian Weng — Contrastive Representation Learning <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2021-05-31-contrastive/){: target="_blank" }
<span class="annotation">A thorough tour of self-supervised contrastive learning, from SimCLR to BYOL to DINO. Useful index of the major methods.</span>

</li>
</ul>

</div>

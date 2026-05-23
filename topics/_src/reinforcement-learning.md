---
title: Reinforcement Learning — ML Resources Hub
eyebrow_text: ← Theory · Learning Paradigms
eyebrow_href: ../theory.html
heading: Reinforcement Learning
lead: Learn by doing — no labels, only rewards. The framework that powers game-playing agents, robotics, and RLHF.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**An agent acts in an environment; the environment gives reward.** No labels. No supervised target. The agent has to figure out, from sparse and often delayed reward signals, what good behaviour looks like. The classical loop: observe state, choose action, receive reward and new state, repeat.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Watch tabular Q-learning learn a grid-world policy — Q-values flood backward from the goal as episodes accumulate</span>
</div>
<div class="viz-classic-controls">
<button id="viz-rl-step" type="button">Step (1 episode)</button>
<button id="viz-rl-play" type="button">Play</button>
<button id="viz-rl-reset" type="button">Reset</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                ε (exploration)
                <input id="viz-rl-eps" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-rl-eps-lbl">ε = 0.20</span>
<span class="viz-classic-badge" id="viz-rl-ep-lbl">episode 0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-rl-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-rl-caption"></div>
</div>

<script src="{{root}}js/viz/reinforcement-learning.js"></script>

A 5×5 grid world. The agent (indigo dot) starts top-left and gets +1 for reaching the goal (orange square) and -1 for stepping on a wall (grey). Each cell shows the max Q-value across actions — colour intensity = how good that state is. Watch the "warmth" flood backward from the goal as more episodes play out. Drop ε to greedy (almost no exploration); the agent gets stuck.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**The RL loop.** State *s* → action *a* → reward *r* → next state *s'*. Repeat. The agent learns a policy *π(a | s)* that maximises expected discounted future reward.

**Q-learning.** Learn a value *Q(s, a)* = "expected discounted reward if I take action *a* in state *s* and act greedily afterwards." Update rule: *Q(s, a) ← Q(s, a) + α · (r + γ max<sub>a'</sub> Q(s', a') − Q(s, a))*.

**Exploration vs exploitation.** ε-greedy: with probability ε pick a random action, otherwise pick the best Q. Too much ε → never converges. Too little → stuck in local optima.

**Policy gradient.** Parametrize the policy directly (*π<sub>θ</sub>(a | s)*) and follow the gradient of expected reward. REINFORCE, A2C, PPO. Works when actions are continuous or the action space is huge.

**Modern deep RL.** Replace the Q-table with a neural network → DQN. Add policy networks → A2C, PPO. Add a value-network critic + replay buffer → SAC. The pieces are old; the engineering is what makes them work.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Sequential decisions with delayed reward
- Game-playing, robotics, control
- RLHF — aligning models to human preferences
- No labelled data, but a simulator or rollout mechanism

</div>

<div class="no" markdown="1">

### Limits

- Sample-inefficient — usually needs millions of rollouts
- Unstable training — exploration / exploitation balance is fragile
- Reward design is hard — the agent will exploit whatever you wrote
- Sim-to-real gap — policies that work in simulation often fail on hardware

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, eps = 0.1, 0.95, 0.2

for episode in range(5000):
    s, _ = env.reset()
    done = False
    while not done:
        a = np.random.randint(4) if np.random.random() < eps \
            else Q[s].argmax()
        s_next, r, done, *_ = env.step(a)
        Q[s, a] += alpha * (r + gamma * Q[s_next].max() - Q[s, a])
        s = s_next

# Greedy policy
policy = Q.argmax(axis=1)
```

</div>

<div class="level-next">
<span>Want MDPs, Bellman equations, and actor-critic?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Bellman optimality</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ Q^*(s, a) = \mathbb{E}_{s'}\!\left[\, R(s, a) + \gamma \max_{a'} Q^*(s', a') \,\right] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Q*(s, a)`optimal action-value: best possible expected return from state *s* after taking action *a*

</li>
<li markdown="1">

`R(s, a)`immediate reward for taking *a* in *s*

</li>
<li markdown="1">

`γ`discount factor (0 = myopic; 1 = far-sighted)

</li>
<li markdown="1">

`Es'[·]`average over the next state *s'* drawn from the environment's dynamics

</li>
<li markdown="1">

Q-learning bootstraps this equation iteratively

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ Q^*(\text{state}, \text{action}) \;=\; \text{avg over next states}\!\left[\, \text{reward now} \;+\; \gamma \times \max_{\text{next action}} Q^*(\text{next state}, \text{next action}) \,\right] $$</span>

**In words.** The best possible value of taking an action in a state is whatever reward you collect right now, plus a discounted estimate of the best value you can achieve from *wherever you end up* next. The `Es'` averages over the randomness in where you land. The `max` says you assume you'll act optimally from then on. `γ` (gamma, between 0 and 1) is the discount — it shrinks rewards that are further in the future, so the agent prefers earlier payoffs. This is a self-referential equation: `Q*` appears on both sides. Q-learning solves it by repeatedly nudging `Q` toward the right-hand side as the agent collects experience.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Q*`optimal value of (state, action) — best expected total reward from that point on

</li>
<li markdown="1">

`reward now`the immediate reward signal from taking that action

</li>
<li markdown="1">

`γ`discount factor between 0 and 1 — higher means longer planning horizon

</li>
<li markdown="1">

`avg over next states`average across the random next states the environment can transition to

</li>
<li markdown="1">

`max over next action`assume you'll pick the best action next time, too

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**MDPs.** States *S*, actions *A*, transitions *P(s' | s, a)*, rewards *R(s, a, s')*, discount *γ*. The agent's job is a policy *π* that maximises expected discounted return.

**Value functions.** *V<sup>π</sup>(s)* = expected return from *s* following *π*. *Q<sup>π</sup>(s, a)* = expected return from *s*, taking action *a*, then following *π*. Both satisfy Bellman equations; both can be learned.

**Model-free vs model-based.** Model-free: learn *V* or *Q* directly from experience (Q-learning, SARSA, DQN). Model-based: learn the transition model and plan (Dyna-Q, MuZero). Model-based is more sample-efficient when the model is accurate; less so when it isn't.

**Policy gradient methods.** Parameterise *π<sub>θ</sub>* and take steps in the gradient of expected return: *∇<sub>θ</sub> J = E[Σ<sub>t</sub> ∇<sub>θ</sub> log π<sub>θ</sub>(a<sub>t</sub> | s<sub>t</sub>) · A(s<sub>t</sub>, a<sub>t</sub>)]* where *A* is the advantage. REINFORCE is the simplest form; A2C, A3C, PPO add stability.

**Actor-critic.** Combine policy gradient (actor) with a learned value function (critic). The critic reduces variance; the actor exploits it. Modern RL is mostly actor-critic in some form.

**On-policy vs off-policy.** On-policy (PPO, A2C): learn from data the current policy collected. Stable but sample-inefficient. Off-policy (DQN, SAC, DDPG): learn from any data (replay buffer). More sample-efficient; trickier to make stable.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn, torch.nn.functional as F

# DQN: Q-function approximated by a neural net, replay buffer, target network
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                 nn.Linear(128, n_actions))
    def forward(self, x): return self.net(x)

# One training step
def dqn_step(net, target_net, batch, gamma=0.99):
    s, a, r, s_next, done = batch
    q     = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target_net(s_next).max(dim=1).values
        target = r + gamma * (1 - done) * q_next
    return F.smooth_l1_loss(q, target)
```

</div>

<div class="level-next">
<span>Want PPO, soft actor-critic, RLHF, and the exploration zoo?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">PPO clipped objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \min\!\Big(r_t(\theta) \hat A_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t\Big) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`rt(θ)`importance ratio between new and old policy: *π<sub>θ</sub>(a<sub>t</sub> | s<sub>t</sub>) / π<sub>old</sub>(a<sub>t</sub> | s<sub>t</sub>)*

</li>
<li markdown="1">

`Ât`estimated advantage at step *t* — how much better this action was than average

</li>
<li markdown="1">

`clip(·, 1−ε, 1+ε)`squashes the ratio into a tight band around 1

</li>
<li markdown="1">

`ε`clip range, typically 0.1–0.3

</li>
<li markdown="1">

Clip prevents large policy updates that destabilise training — the default actor-critic in practice

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; \text{avg}_t \min\!\Big(\text{ratio}_t \times \text{advantage}_t, \; \text{clip}(\text{ratio}_t, 1-\epsilon, 1+\epsilon) \times \text{advantage}_t\Big) $$</span>

**In words.** The **ratio** is how much more likely the new policy is to take the action than the old policy was. The **advantage** is how much better than average that action turned out to be. Their product is the natural "policy gradient" term — make good actions more likely, bad actions less. The trick is the `clip`: it caps the ratio inside a tight band around 1 (typically 0.8 to 1.2), and the `min` picks the more pessimistic of the two surrogate objectives. The net effect: PPO can move the policy quickly when the advantage is small, but stops cold when the policy is already very different from the data-collection policy — which is what makes it stable.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`ratio`new policy's probability of the action divided by the old policy's probability

</li>
<li markdown="1">

`advantage`how much better this action was than the average action at that state

</li>
<li markdown="1">

`clip(ratio, 1−ε, 1+ε)`force the ratio into a narrow band — prevents huge updates

</li>
<li markdown="1">

`min(·, ·)`pessimistic choice — caps both upside and downside of the update

</li>
<li markdown="1">

`ε`clip width (commonly 0.1 to 0.3)

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**PPO.** Schulman et al. (2017). The de facto default for continuous control. Clipped ratio objective prevents large policy updates; multiple epochs over the same rollouts; advantage estimation via GAE. Practical, simple, robust.

**SAC.** Haarnoja et al. (2018). Off-policy actor-critic with entropy regularization — encourages exploration by maximising "expected return + policy entropy". Sample-efficient; the right choice for many continuous-action benchmarks.

**MuZero.** Schrittwieser et al. (2020). Learn the dynamics model in a latent space and plan with Monte-Carlo tree search. Achieves AlphaGo-level play without a hand-coded simulator. Beautiful theoretical synthesis.

**Offline RL.** Learn from a fixed dataset without environment access. Conservative Q-Learning (CQL), Implicit Q-Learning (IQL), Decision Transformer reformulate the problem as supervised sequence modelling. Hard because OOD actions can't be evaluated.

**RLHF.** Reinforcement Learning from Human Feedback. Train a reward model from human preferences over pairs of outputs; optimise an LLM against it with PPO. Made instruction-following LLMs possible. The pretrain → SFT → RLHF pipeline is now standard for assistant-style models.

**Exploration.** ε-greedy is the floor. Better: entropy bonuses (SAC), intrinsic motivation (curiosity-driven, ICM), Thompson sampling on the Q-distribution (Bootstrapped DQN), random network distillation (RND). The right method depends on the problem's reward sparsity.

**The deep-RL stability cottage industry.** Modern deep RL is a list of stability tricks: target networks, replay buffers, layer normalization, clipped gradients, learning-rate annealing, normalised observations, advantage normalization, … None alone is magic; together they make things barely work.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn.functional as F

# PPO clipped surrogate loss
def ppo_loss(logp_new, logp_old, advantages, eps=0.2):
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(surr1, surr2).mean()

# Generalised Advantage Estimation
def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(rewards) else 0
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages
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

[Sutton & Barto — Reinforcement Learning: An Introduction <i class="fas fa-external-link-alt"></i>](http://incompleteideas.net/book/the-book-2nd.html){: target="_blank" }
<span class="annotation">The reference textbook. Free PDF online. Chapters 1–6 are essential for any serious RL work.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[OpenAI Spinning Up <i class="fas fa-external-link-alt"></i>](https://spinningup.openai.com/){: target="_blank" }
<span class="annotation">Practical introduction to deep RL with clean reference implementations of REINFORCE, A2C, PPO, DDPG, SAC, TRPO.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — Policy Gradients <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/){: target="_blank" }
<span class="annotation">Tour of all the major policy-gradient methods — REINFORCE, A2C, A3C, TRPO, PPO, SAC — with the math worked out.</span>

</li>
<li data-tier="indepth" markdown="1">

[Schulman et al. (2017) — PPO <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1707.06347){: target="_blank" }
<span class="annotation">The PPO paper. Short and readable; section 3 is the canonical loss derivation.</span>

</li>
</ul>

</div>

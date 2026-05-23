---
title: A/B Testing — ML Resources Hub
eyebrow_text: ← Engineering · Production
eyebrow_href: {{root}}engineering.html
heading: A/B Testing
lead: Comparing models in production — sample size, statistical significance, bandits, and the human factors.
active_nav: engineering
prev_href: quantization-distillation.html
prev_title: Quantization &amp; Distillation
next_href: notebook-to-script.html
next_title: Notebook → Script
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**You can't tell from offline metrics alone whether a model is "better".** Offline AUC up doesn't always mean online conversion up. A/B test in production with a clearly-defined metric, enough sample size for the effect you care about, and pre-registered analysis. Then decide.

</div>

<article class="tldr-body" markdown="1">

**The basic A/B.** Random 50/50 split of users (or sessions). Group A (control) gets the current model; Group B gets the candidate. Compare a primary metric over a fixed evaluation window. Hypothesis test: is the difference larger than chance?

**Sample size.** Smaller effect → larger sample needed. The standard formula: *n ≈ 16 / d²* where *d* is the effect size in standard-deviation units. A 1% relative improvement on a 10%-baseline metric needs ~20 000 users per arm.

**Common gotchas.** Peeking at results and stopping early (inflates false-positive rate). Network effects (the treatment of one user affects another). Novelty effects (users react to anything new, then revert). Seasonality (test on a representative time window).

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### When A/B works well

- Clearly-defined primary metric (CTR, conversion, latency)
- Independent users / sessions (no network effects)
- Effect size estimate from offline data
- Enough traffic to reach significance in days, not months

</div>

<div class="no" markdown="1">

### When it doesn't

- Low traffic — power is too low to detect anything
- Network effects (social, marketplace)
- Slow feedback (months between exposure and outcome)
- Multiple competing metrics with trade-offs — needs a different framework

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from scipy import stats

# Sample size for a 2-proportion test
def required_n(p1, p2, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    se = (2 * p_bar * (1 - p_bar)) ** 0.5
    n = ((z_alpha * se + z_beta * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5)
         / abs(p1 - p2)) ** 2
    return int(n)

print(required_n(0.10, 0.105))    # ~62 000 per arm to detect 5% relative lift

# Two-proportion z-test for results
def ab_test(success_a, n_a, success_b, n_b):
    p_a, p_b = success_a / n_a, success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = (p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) ** 0.5
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"p_a": p_a, "p_b": p_b, "lift": p_b - p_a, "p_value": p_value}
```

</div>

<div class="level-next">
<span>Want sequential testing, multi-armed bandits, & Bayesian A/B?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Statistical power</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ n \approx \frac{2\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*δ* the minimum detectable effect

</li>
<li markdown="1">

*α, β* false-positive and false-negative rates (typically 0.05, 0.20)

</li>
<li markdown="1">

*σ* standard deviation of the metric

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{users per arm} \approx \frac{2 \times (\text{metric spread})^2 \times (\text{z-values})^2}{(\text{minimum effect})^2} $$</span>

**In words.** This is the standard sample-size formula: how many users you need in each arm to reliably detect an effect. The numerator grows with the *variance* of your metric (`σ²`) — noisier metrics need more data. The `z`-values come from your tolerated false-positive rate `α` (typically 5%) and false-negative rate `β` (typically 20%); together they encode "how confident do I want to be?". The denominator is the squared **minimum detectable effect** `δ` — the smaller the lift you want to catch, the more users you need. Halving the effect quadruples the sample.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`users per arm`how many users you need in *each* of A and B

</li>
<li markdown="1">

`metric spread`standard deviation of the outcome metric (σ)

</li>
<li markdown="1">

`z-values`quantiles of the normal distribution at confidence level α and power 1−β

</li>
<li markdown="1">

`minimum effect`smallest lift δ you care about detecting

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Sequential testing.** Fixed-horizon A/B requires committing to a sample size up front. Sequential / always-valid p-values let you peek without inflating false positives. Methods: SPRT, mSPRT, group-sequential designs, e-values. Tools: Optimizely, Statsig, Eppo all implement some variant.

**CUPED (variance reduction).** Microsoft's technique: regress the metric on a pre-experiment covariate (the same user's metric before the test). The residuals have much lower variance, so you need fewer users for the same power. 20–50% sample size reduction is typical.

**Multi-armed bandits.** Instead of fixed splits, allocate more traffic to better-performing arms over time. Thompson sampling is the popular choice. Trade-off: faster learning vs cleaner causal inference. Use bandits for exploitation; A/B for explanation.

**Bayesian A/B.** Compute the posterior probability that B beats A by at least *δ*. More intuitive for stakeholders ("80% chance B is better"). Same data; different interpretation. Doesn't fix multiple-comparisons or peeking by itself.

**Multiple comparisons.** Testing many metrics increases the chance of a false positive somewhere. Pre-register a primary metric; report others as "exploratory". Bonferroni or Benjamini-Hochberg for principled correction.

**Subgroup analysis.** The treatment can help one segment and hurt another. Slice by demographics, geography, device. Watch for Simpson's paradox: aggregate looks fine, every subgroup is worse.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

# CUPED — variance reduction using pre-period covariate
def cuped_ate(y_a, x_a, y_b, x_b):
    """Average Treatment Effect with CUPED-adjusted outcomes."""
    y_all = np.concatenate([y_a, y_b])
    x_all = np.concatenate([x_a, x_b])
    # Optimal theta
    theta = np.cov(y_all, x_all)[0, 1] / np.var(x_all)
    y_adj = y_all - theta * (x_all - x_all.mean())
    y_a_adj, y_b_adj = y_adj[:len(y_a)], y_adj[len(y_a):]
    return y_b_adj.mean() - y_a_adj.mean()

# Thompson sampling bandit — Bernoulli arms
class ThompsonBandit:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms); self.beta = np.ones(n_arms)
    def select(self):
        return np.argmax(np.random.beta(self.alpha, self.beta))
    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm]  += 1 - reward
```

</div>

<div class="level-next">
<span>Want interleaving, off-policy evaluation, & long-term effects?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Inverse propensity scoring</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \hat V = \frac{1}{n} \sum_{i=1}^n \frac{\mathbb{1}\{a_i = \pi(x_i)\}}{p_\mathrm{behaviour}(a_i \mid x_i)} \, r_i $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Estimate the value of a new policy *π* from logs of an old behaviour policy

</li>
<li markdown="1">

Doesn't require deploying *π* to evaluate it

</li>
<li markdown="1">

Variance grows with how different *π* is from behaviour

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{estimated value} \;=\; \text{average over logs of}\; \frac{\text{1 if new policy agrees with log}}{\text{prob old policy picked that action}} \times \text{reward} $$</span>

**In words.** You want to estimate how a new policy `π` (a way of choosing actions) would perform, using only logged data from the old "behaviour" policy. For each logged event, check whether the new policy would have chosen the same action; if yes, keep the reward but *reweight* it by dividing by the probability the old policy assigned to that action. The `𝟙{·}` (indicator) is 1 when the condition holds, 0 otherwise. Dividing by the logging probability corrects for the bias from the old policy picking some actions more than others — actions that were rare under the old policy get up-weighted because we have less data about them. Average across all `n` logged events to get the estimate.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`estimated value`expected reward of the new policy, estimated from old logs

</li>
<li markdown="1">

`new policy`the policy π you want to evaluate (without deploying it)

</li>
<li markdown="1">

`prob old policy picked that action`the propensity score from the logging policy

</li>
<li markdown="1">

`reward`observed outcome (click, conversion, revenue) for that logged event

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Interleaving.** Show both models' results to the same user (e.g., ranked-list problems — interleave A's and B's recommendations). Much higher statistical power per user; works because each user is their own control. Used heavily at Microsoft, Netflix, search engines.

**Off-policy evaluation (OPE).** Estimate how a new policy would perform from logs of a previous one. IPS (inverse propensity scoring), doubly robust estimators, model-based OPE. Standard in recommendation and ad-ranking. Requires the logging policy to have explored — if it always picked the same thing, you can't evaluate alternatives.

**Long-term effects.** Some changes hurt short-term metrics but help long-term (paywalls, ads, content moderation). Different evaluation: long-term holdouts, instrumental variables, or careful causal modelling. Hard; rarely done well.

**Heterogeneous treatment effects (HTE).** The treatment helps some users and hurts others. Estimating *τ(x) = E[Y(1) − Y(0) | X = x]* with causal forests, double ML, or T-/X-learners. Useful for targeted deployment.

**Switchback experiments.** When users can't be split (e.g., ride-share dispatching), switch the treatment on and off over time within the same population. Mitigates network effects.

**Pre-registration & audit.** Write the analysis plan before looking at results. Commit to one primary metric, one stopping rule, one analysis method. Reduces hindsight-driven p-hacking — and makes the test reusable in retros.

**Cost-aware testing.** Each "challenger" model deployment incurs implementation, ramp, and rollback costs. Decision-theoretic framing: expected value of running the test > cost of running it? Often the test isn't worth running because the expected effect is too small.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

# Inverse Propensity Scoring — estimate policy value from logged data
def ips_estimate(actions, rewards, propensities, new_policy):
    """
    actions:      array of taken actions
    rewards:      observed reward for each
    propensities: P(action | context) under the LOGGING policy
    new_policy:   function returning P(action | context) under the NEW policy
    """
    weights = np.array([
        new_policy(a, ctx) / propensities[i]
        for i, (a, ctx) in enumerate(zip(actions, contexts))
    ])
    return (weights * rewards).mean()

# Switchback for network-effect mitigation
def switchback_test(treatment_schedule, observations):
    """Treatment switches on/off in blocks; compare blocks of A vs B."""
    df = pd.DataFrame(observations)
    df["assignment"] = treatment_schedule
    return df.groupby("assignment")["metric"].mean()
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
<li data-tier="indepth" markdown="1">

[Microsoft Experimentation Platform <i class="fas fa-external-link-alt"></i>](https://exp-platform.com/){: target="_blank" }
<span class="annotation">Kohavi et al.'s research site. The "trustworthy A/B testing" papers are required reading for anyone running production experiments.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Statsig — Documentation <i class="fas fa-external-link-alt"></i>](https://www.statsig.com/){: target="_blank" }
<span class="annotation">A modern experimentation platform; their blog covers sequential testing, CUPED, and HTE in accessible language.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Kohavi, Tang & Xu — Trustworthy Online Controlled Experiments <i class="fas fa-external-link-alt"></i>](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264){: target="_blank" }
<span class="annotation">The textbook. Pragmatic, exhaustive; covers the entire A/B testing lifecycle in industry.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Lilian Weng — Multi-Armed Bandits <i class="fas fa-external-link-alt"></i>](https://lilianweng.github.io/posts/2018-10-21-mab/){: target="_blank" }
<span class="annotation">Clear introduction to bandits and their relationship to A/B testing.</span>

</li>
</ul>

</div>

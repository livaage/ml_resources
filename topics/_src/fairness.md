---
title: Fairness, Bias &amp; Ethics — ML Resources Hub
eyebrow_text: ← Theory · Frontier
eyebrow_href: ../theory.html
heading: Fairness, Bias &amp; Ethics
lead: Different subgroups can get different model performance — and that matters. Definitions, metrics, mitigations, and the inherent trade-offs.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**"Accurate on average" hides who pays the error cost.** A model can be 90% accurate overall, 95% on one group, 70% on another. There are several incompatible definitions of "fair" — demographic parity, equal opportunity, equalised odds, calibration — and you generally can't satisfy them all simultaneously. Pick a definition that matches the deployment's actual harms.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Same model, two groups — slide the threshold and watch each fairness metric shift in different directions</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                τ
                <input id="viz-fair-threshold" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-fair-resample" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-fair-thr-lbl">τ = 0.50</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-fair-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-fair-caption"></div>
</div>

<script src="{{root}}js/viz/fairness.js"></script>

Two groups, A and B, with different score distributions but the *same* classifier and threshold. Slide the threshold and watch how each fairness metric shifts: **demographic parity** (equal positive rate), **equal opportunity** (equal TPR), and **predictive parity** (equal precision). You can match one at a time — almost never all three.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Demographic parity / statistical parity.** Each group has the same rate of positive predictions. Useful when the base rate *shouldn't* differ across groups (e.g., advertising), problematic when the base rate genuinely does.

**Equal opportunity.** Each group has the same true positive rate (recall). Useful for "everyone qualified gets a chance" framing — loans, jobs.

**Equalised odds.** Both TPR and FPR are equal across groups. Stricter than equal opportunity.

**Predictive parity.** Each group has the same precision — if the model says positive, the probability of being correct is the same. Useful when downstream consequences are tied to the prediction itself.

**Calibration within groups.** Predicted probabilities mean the same thing for each group — 0.7 means 70% positive rate within each group. A common starting point that's often in tension with the rate-based fairness metrics.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Pick the metric that matches the harm

- "Equal access" → demographic parity
- "Equal chance for qualified people" → equal opportunity
- "Same precision for everyone" → predictive parity
- "Probabilities mean the same thing" → calibration

</div>

<div class="no" markdown="1">

### The impossibility theorems

- Chouldechova (2017): predictive parity + equal FPR + base rate differences → impossible
- Kleinberg, Mullainathan, Raghavan (2017): calibration + balance for FP/FN → impossible (except in trivial cases)
- Translation: you have to choose which fairness criterion you violate

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np

def demographic_parity(y_pred, group):
    return {g: y_pred[group == g].mean() for g in np.unique(group)}

def equal_opportunity(y_true, y_pred, group):
    # TPR = TP / (TP + FN), computed per group
    out = {}
    for g in np.unique(group):
        m = (group == g) & (y_true == 1)
        out[g] = y_pred[m].mean() if m.any() else np.nan
    return out

def predictive_parity(y_true, y_pred, group):
    # Precision = TP / (TP + FP), computed per group
    out = {}
    for g in np.unique(group):
        m = (group == g) & (y_pred == 1)
        out[g] = y_true[m].mean() if m.any() else np.nan
    return out

# A common mitigation: per-group thresholds tuned to satisfy a fairness criterion
def per_group_threshold(scores, group, target_tpr=0.8, y_true=None):
    thresholds = {}
    for g in np.unique(group):
        m = (group == g) & (y_true == 1)
        thresholds[g] = np.quantile(scores[m], 1 - target_tpr)
    return thresholds
```

</div>

<div class="level-next">
<span>Want individual fairness, causal frameworks, & mitigations?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Equalised odds</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ P(\hat Y = 1 \mid Y = y, A = a) = P(\hat Y = 1 \mid Y = y, A = a'),\quad \forall y, a, a' $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*Â* protected attribute, *Y* true label, *Ŷ* prediction

</li>
<li markdown="1">

Both TPR and FPR equal across groups

</li>
<li markdown="1">

Stricter than equal opportunity (which only requires TPR equality)

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{probability of "predict positive"} \;\text{given the true label and the protected group} \;\text{is the same across all groups} $$</span>

**In words.** Equalised odds requires that the model's behaviour on real positives and on real negatives is identical across protected groups. Specifically, the *true positive rate* (chance of being predicted positive when actually positive) and the *false positive rate* (chance of being predicted positive when actually negative) must each match across groups. `Y` is the true label, `Ŷ` is the prediction, and `A` is the protected attribute (e.g. gender, race). This is stricter than "equal opportunity", which only requires the true-positive rate to be equal.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Y`true label (positive or negative)

</li>
<li markdown="1">

`Ŷ`model's prediction

</li>
<li markdown="1">

`A`protected attribute (e.g. group membership)

</li>
<li markdown="1">

Both TPR and FPR equal across groups; stricter than equal opportunity

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Individual fairness.** Similar individuals should get similar predictions. Dwork et al. (2012). Requires a domain-specific similarity metric; rarely deployed in practice but a useful normative anchor.

**Causal fairness.** Define fairness in a structural causal model: counterfactual fairness (would this prediction change if I changed protected attribute, all else equal?), path-specific fairness (some paths through the DAG should be eliminated). Requires explicit causal modelling.

**Mitigation strategies.** *Pre-processing*: re-weight, re-sample, learn fair representations (Zemel et al. 2013). *In-processing*: add fairness constraints to the loss (e.g., adversarial debiasing). *Post-processing*: adjust the threshold per group (Hardt et al. 2016). Each has different trade-offs.

**Bias sources.** Historical bias (data reflects historical inequities); representation bias (some groups undersampled); measurement bias (proxy labels are unequally accurate); aggregation bias (one model for heterogeneous populations); evaluation bias (test sets that miss specific groups).

**"Fair by construction" representations.** Train an encoder so that the protected attribute is unpredictable from the embedding. Risk: overfit to the in-distribution adversary; correlated proxies (e.g., ZIP code, surname) still leak group membership.

**Audit ≠ certify.** Even careful fairness audits can miss failure modes. Subgroup analysis (intersectionality), distribution shift, deployment monitoring. Fairness is an ongoing practice, not a one-time check.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference,
    equalized_odds_difference, selection_rate,
)
from sklearn.metrics import accuracy_score, recall_score

# Group-aware metrics
frame = MetricFrame(
    metrics={"acc": accuracy_score, "tpr": recall_score, "sel_rate": selection_rate},
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=group,
)
print(frame.by_group)
print("Demographic parity diff:", demographic_parity_difference(y_true, y_pred, sensitive_features=group))
print("Equalised odds diff   :",  equalized_odds_difference(y_true, y_pred, sensitive_features=group))

# Post-processing: threshold each group to equalise TPR
from fairlearn.postprocessing import ThresholdOptimizer
to = ThresholdOptimizer(estimator=base, constraints="equalized_odds")
to.fit(X_train, y_train, sensitive_features=group_train)
y_hat = to.predict(X_test, sensitive_features=group_test)
```

</div>

<div class="level-next">
<span>Want counterfactual fairness, mech-interp for bias, & LLM alignment ethics?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Counterfactual fairness</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ P(\hat Y_{A \leftarrow a}(U) = y \mid X = x, A = a) = P(\hat Y_{A \leftarrow a'}(U) = y \mid X = x, A = a), \forall y, a' $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

The prediction shouldn't change in the counterfactual world where the protected attribute was different

</li>
<li markdown="1">

Requires a structural causal model of the data-generating process

</li>
<li markdown="1">

Strong assumption; weak observation guarantees

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{prediction in the actual world} \;=\; \text{prediction in the counterfactual world where only the protected attribute is changed} $$</span>

**In words.** Imagine rewinding the universe and changing *only* the protected attribute `A` (e.g. flipping gender or race) while holding everything else that makes the person who they are constant. Counterfactual fairness says the model's prediction shouldn't differ between the actual and counterfactual person. The notation `A ← a` is "do-calculus": surgically set *A* to value *a*. `U` stands for unobserved background factors. This is a strong philosophical commitment — it requires you to have written down a structural causal model of the data-generating process, which is rarely possible without strong assumptions.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`A ← a`counterfactual intervention: set the protected attribute to value *a*

</li>
<li markdown="1">

`U`unobserved background factors of the individual

</li>
<li markdown="1">

`X`observed features

</li>
<li markdown="1">

Strong assumption; requires a causal model — rarely deployed in practice

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Causal frameworks.** Kusner et al. (2017) — counterfactual fairness via SCMs. Define which causal paths from the protected attribute to the outcome are "fair" (e.g., job-relevant skills) vs "unfair" (e.g., discriminatory hiring). Block the unfair paths while preserving the fair ones. Strong assumptions; controversial.

**Foundation-model bias.** LLMs absorb biases from training data — gender stereotypes, racial associations, geographic skew. Mitigations: filtered pre-training data, instruction tuning ("don't say things like this"), RLHF reward models that penalise biased outputs, post-hoc safety filters. None fully solve it; the surface area is enormous.

**Allocation vs representation harms.** Allocation: who gets the loan / job / treatment? Representation: how are different groups depicted in generated content? Different metrics, different mitigations.

**Mech-interp for bias.** Identify circuits in a network that produce biased outputs; intervene on them. Early work — most interpretability methods aren't precise enough yet, but the conceptual approach is promising.

**Algorithmic harms beyond accuracy.** Privacy, surveillance, agency, opacity. Even a perfectly calibrated, fair-by-every-metric system can cause harm if its mere existence creates a chilling effect or removes meaningful human review. Most production ethics work is structural, not algorithmic.

**The political dimension.** Fairness is normative. Different stakeholders prefer different metrics. The choice between equal opportunity and demographic parity is a policy choice, not a technical one. ML practitioners need to surface these trade-offs to decision-makers, not bury them.

**Audit reports.** Document training data sources, metrics across subgroups, known limitations, monitoring plans. Increasingly required by regulation (EU AI Act, NYC AEDT law). "Model cards" (Mitchell et al. 2019) and "datasheets for datasets" (Gebru et al. 2018) are useful templates.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn.functional as F

# Adversarial debiasing — train an adversary to predict the protected attribute
# from the model's hidden state; gradient-reverse so the model becomes invariant
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

def grad_reverse(x, lam=1.0): return GradReverse.apply(x, lam)

# Training loop
y_hat = model(x)                # main prediction
h = model.hidden(x)             # hidden representation
attr_hat = adv(grad_reverse(h, 1.0))   # adversary predicts the protected attr
loss = F.cross_entropy(y_hat, y) + F.cross_entropy(attr_hat, sensitive_attr)
loss.backward()
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

[Barocas, Hardt & Narayanan — Fairness and Machine Learning <i class="fas fa-external-link-alt"></i>](https://fairmlbook.org/){: target="_blank" }
<span class="annotation">The reference textbook. Free online. Comprehensive treatment of fairness definitions, metrics, and trade-offs.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Hardt, Price & Srebro (2016) — Equal Opportunity <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1610.02413){: target="_blank" }
<span class="annotation">The post-processing-for-fair-thresholds paper. Foundational; readable.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Fairlearn <i class="fas fa-external-link-alt"></i>](https://fairlearn.org/){: target="_blank" }
<span class="annotation">Microsoft's open-source fairness toolkit. Metrics, mitigations, dashboards.</span>

</li>
<li data-tier="indepth" markdown="1">

[Mitchell et al. (2019) — Model Cards <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1810.03993){: target="_blank" }
<span class="annotation">The Google paper proposing standardised model documentation. Now widely adopted by HuggingFace, OpenAI, Anthropic.</span>

</li>
</ul>

</div>

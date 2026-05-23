---
title: Automated Retraining — ML Resources Hub
eyebrow_text: ← Engineering · CI / CD
eyebrow_href: {{root}}engineering.html
heading: Automated Retraining
lead: When to retrain, how to schedule it, and what gates promote a fresh model to production.
active_nav: engineering
prev_href: data-validation.html
prev_title: Data Validation
next_href: model-registries.html
next_title: Model Registries
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Models go stale. Retrain on a schedule or a trigger.** The "schedule" answer is "weekly / monthly / quarterly" and works for most use cases. The "trigger" answer is "drift detected, performance dropped, new data arrived" — more responsive but harder to engineer. Most production systems use both.

</div>

<article class="tldr-body" markdown="1">

Two failure modes to avoid: **never retrain** (the model gradually drifts out of usefulness) and **retrain too eagerly** (every new model destabilises predictions, breaks downstream systems, and introduces churn). The right cadence depends on how fast your data distribution changes.

**When to retrain.** Calendar trigger: every N days/weeks. Drift trigger: data or prediction distribution has changed. Performance trigger: live metrics have dropped. New-data trigger: a significant amount of fresh labelled data is available.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Retraining triggers

- **Calendar**: simplest; weekly / monthly / quarterly
- **Data drift**: PSI / KS thresholds exceeded
- **Performance drift**: live precision / recall drop
- **New data**: K new labelled examples accumulated
- **Manual**: data scientist kicks it off after a discovery

</div>

<div class="no" markdown="1">

### Pitfalls

- Retraining too often → instability + drift in predictions seen by downstream systems
- Retraining never → silent degradation
- No promotion gate → bad models reach production
- No rollback plan → a bad deploy is hard to undo

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```yaml
# A scheduled GitHub Action that retrains weekly
name: Weekly Retrain
on:
  schedule: [{ cron: '0 3 * * 1' }]    # 03:00 UTC Mondays
  workflow_dispatch: {}                 # also runnable on demand

jobs:
  retrain:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Pull fresh data
        run: dvc pull data/
      - name: Validate data
        run: python -m my_project.validate_data data/
      - name: Train
        run: python -m my_project.train --config configs/prod.yaml
      - name: Evaluate against production baseline
        id: eval
        run: python -m my_project.eval --baseline registry://my-model:prod
      - name: Register staging model
        if: ${{ steps.eval.outputs.beats_baseline == 'true' }}
        run: mlflow models register --name my-model --stage Staging
```

</div>

<div class="level-next">
<span>Want the gating logic, rollback, & the human-in-the-loop?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Promotion criteria</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{promote} \;\Leftrightarrow\; \text{new beats baseline by} \geq \delta \;\wedge\; \text{no regression on subgroup metrics} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*δ* = practically meaningful improvement, not statistical noise

</li>
<li markdown="1">

Subgroup regressions = silent harm; gate on them explicitly

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{promote} \;\;\text{if and only if}\;\; (\text{new model beats baseline by at least } \delta) \;\;\text{AND}\;\; (\text{no subgroup gets worse}) $$</span>

**In words.** Two boolean conditions joined by AND — both must hold for the new model to ship. The `⇔` symbol means "if and only if": promotion is exactly equivalent to these conditions. The first gate (`δ`, a small positive threshold) prevents shipping noise: the lift must be large enough to matter, not just barely positive. The second gate (the `∧` in the math is logical AND) protects against silent harm: even if the aggregate metric goes up, you don't promote if any tracked subgroup goes meaningfully down.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`promote`move the candidate model into production

</li>
<li markdown="1">

`δ`minimum meaningful improvement threshold (not just statistical significance)

</li>
<li markdown="1">

`subgroup metrics`per-segment performance (gender, geography, customer tier)

</li>
<li markdown="1">

`AND`both conditions must be true — fail either and you don't ship

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Promotion gates.** Before a new model goes live: aggregate metric must beat baseline by at least δ on a frozen test set. No regression on subgroups (gender, geography, segment) by more than ε. Latency within budget. Cost within budget. All four are concrete, all four can be checked automatically.

**Shadow deployment.** Send live traffic to both the current model and the candidate; only log the candidate's predictions, don't act on them. Compare metrics over a few days; promote if the candidate wins. The lowest-risk way to vet a model.

**Canary deployment.** Send a small fraction (1–5%) of traffic to the candidate. Watch metrics. Gradually ramp up if healthy. Roll back instantly if not. Standard for any high-stakes deployment.

**Rollback discipline.** Every deployment must have a one-click rollback. The model registry should keep the previous N versions. The serving system should re-load on a config change without a full restart.

**Human-in-the-loop.** For high-stakes domains, an actual person reviews the candidate model: looks at predictions on key examples, checks subgroup metrics, signs off. CI gates everything passable; the human catches the rest.

**Champion / challenger.** The current model is the "champion"; the new is the "challenger". Run both, log both, declare a winner after a fixed evaluation window. Production-grade A/B for models.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import mlflow

def should_promote(candidate_run_id, prod_run_id, *,
                   metric="val/auc", min_improvement=0.005,
                   subgroup_metrics=None, max_subgroup_regression=0.01):
    cand = mlflow.get_run(candidate_run_id).data.metrics
    prod = mlflow.get_run(prod_run_id).data.metrics

    if cand[metric] - prod[metric] < min_improvement:
        return False, f"{metric} insufficient improvement"

    for sg in subgroup_metrics or []:
        if cand[sg] < prod[sg] - max_subgroup_regression:
            return False, f"regression on {sg}"

    return True, "promoted"

# In the retraining workflow:
ok, reason = should_promote(new_run_id, prod_run_id,
                            subgroup_metrics=["val/auc_female", "val/auc_male"])
if ok:
    mlflow.transition_model_version_stage(name="my-model", version=v, stage="Production")
```

</div>

<div class="level-next">
<span>Want online learning, multi-armed bandits, & cost-aware retraining?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Retrain decision</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{retrain} \;\Leftrightarrow\; \mathbb{E}[\text{benefit}] > \text{cost} + \text{risk premium} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Benefit: predicted performance gain × business value

</li>
<li markdown="1">

Cost: compute + engineering + downstream disruption

</li>
<li markdown="1">

Risk premium: every deploy can break something

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{retrain} \;\;\text{if and only if}\;\; \text{expected benefit} \;>\; \text{cost} + \text{risk premium} $$</span>

**In words.** A decision-theoretic framing of retraining. The `𝔼[·]` notation in the math version is an *expected value* — the average benefit you'd get if you ran the retrain many times, weighted by how likely each outcome is. You only retrain when the expected payoff is larger than the combined deterministic cost (compute, engineering time, downstream disruption) plus a **risk premium** — extra padding for the possibility that the deploy goes wrong. This is rarely computed explicitly; in practice teams use heuristics, but the framing tells you why "retrain every week regardless" is wasteful in low-drift domains.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`expected benefit`average improvement × business value of that improvement

</li>
<li markdown="1">

`cost`compute + engineering hours + downstream pipeline disruption

</li>
<li markdown="1">

`risk premium`buffer for the possibility the deploy introduces a regression

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Online learning.** Update model weights continuously from streaming data. Different from retraining in that the model is never "redeployed" — its parameters are constantly evolving. Hard: stability, catastrophic forgetting, monitoring. River library + Vowpal Wabbit are reference implementations.

**Cost-aware retraining.** Each retrain costs compute + engineering attention. Multi-armed bandit literature has the right framework: "exploit" the current model unless evidence accumulates that retraining would pay off. Some teams just retrain on a calendar; principled cost trade-offs are rare in production.

**Re-training vs fine-tuning.** Full retrain: throw away the old model, train from scratch on (new + old) data. Fine-tune: warm-start from the old model. Fine-tuning is faster and more stable but accumulates drift; full retrain is the safe baseline.

**Concept drift vs data drift.** Data drift: P(X) changes; the inputs look different. Concept drift: P(Y | X) changes; the relationship between inputs and outputs has changed. Different remedies — data drift can sometimes be ignored; concept drift requires retraining.

**Champion-challenger at scale.** Many candidates competing; the production traffic is split across them by a bandit policy. Wins over time. Useful when you can afford to run multiple models in parallel.

**Federated retraining.** When data can't leave the user device (privacy / bandwidth). Federated averaging: train local updates, aggregate centrally, ship new weights. Brings new failure modes — clients dropping out, malicious updates, non-IID data per client.

**Catastrophe drills.** Periodically simulate a bad deploy and ensure the rollback works. Don't trust untested rollbacks any more than untested backups.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from river import linear_model, optim, preprocessing, metrics

# Online learning — single-example updates, no batch training
model = preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(0.01))
auc = metrics.ROCAUC()

for x, y in streaming_iter():
    y_pred = model.predict_proba_one(x).get(1, 0.5)
    auc.update(y, y_pred)
    model.learn_one(x, y)

    if step % 1000 == 0:
        print(step, auc)
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

[Sato — Continuous Delivery for ML <i class="fas fa-external-link-alt"></i>](https://martinfowler.com/articles/cd4ml.html){: target="_blank" }
<span class="annotation">Same ThoughtWorks essay as on CI for ML — the model-promotion section is the canonical reference.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Chip Huyen — Real-Time ML Challenges <i class="fas fa-external-link-alt"></i>](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html){: target="_blank" }
<span class="annotation">Practical survey of real-time / continual ML systems. Online learning, monitoring, lambda architectures.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[River — Online Learning in Python <i class="fas fa-external-link-alt"></i>](https://riverml.xyz/){: target="_blank" }
<span class="annotation">Reference library for online learning. Streaming-friendly version of scikit-learn.</span>

</li>
<li data-tier="indepth" markdown="1">

[Tecton — Feature Store Architectures <i class="fas fa-external-link-alt"></i>](https://www.tecton.ai/blog/feature-store-vs-ml-platform/){: target="_blank" }
<span class="annotation">Feature stores are central to robust retraining pipelines. This post lays out the architectural choices.</span>

</li>
</ul>

</div>

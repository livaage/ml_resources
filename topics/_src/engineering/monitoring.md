---
title: Monitoring — ML Resources Hub
eyebrow_text: ← Engineering · Production
eyebrow_href: {{root}}engineering.html
heading: Monitoring
lead: Watch your deployed model — for drift, for degradation, for outright failure.
active_nav: engineering
prev_href: serving.html
prev_title: Serving
next_href: quantization-distillation.html
next_title: Quantization &amp; Distillation
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**The model that worked yesterday may not work today.** Inputs drift. Behaviour changes. Edge cases pile up. Monitor four things: *system* health (uptime, latency), *data* drift (input distributions), *prediction* drift (output distributions), and *performance* (where ground truth is available). Each fails differently; each needs its own alerting.

</div>

<article class="tldr-body" markdown="1">

**The four monitoring planes.** System (latency, throughput, errors, GPU util). Data (per-feature drift vs training). Prediction (output distribution drift). Performance (accuracy, calibration where you have feedback). System monitoring is "is the server up". The other three are "is the model still doing its job".

**The hard part.** You usually don't have ground truth at deployment time. You see the prediction; you don't see the answer for hours, days, or never. Inference becomes a guessing game. Workarounds: proxy metrics, A/B against a baseline, periodic labelled audits.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### What to alert on

- **Server**: 5xx rate, p99 latency, queue depth
- **Data**: PSI > 0.25, missing rate up, schema violations
- **Predictions**: positive-rate shift, score distribution shift
- **Performance**: aggregate metric drop where labels available
- **Subgroups**: per-segment regression beyond threshold

</div>

<div class="no" markdown="1">

### Common mistakes

- Only system-level monitoring — the API is "up" but predicting garbage
- Alerting on noise — too many false positives, alerts get ignored
- No baseline — drift comparisons need a reference distribution
- Aggregate-only metrics — subgroup regressions stay hidden

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from prometheus_client import Counter, Histogram, start_http_server

# Standard server-side metrics
REQUESTS = Counter("preds_total", "Total predictions", ["model_version", "status"])
LATENCY  = Histogram("pred_latency_seconds", "Inference latency",
                    buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5])

@app.post("/predict")
def predict(req: Request):
    t0 = time.time()
    try:
        score = MODEL(req.x)
        REQUESTS.labels(model_version=MODEL.version, status="ok").inc()
        return {"score": score}
    except Exception as e:
        REQUESTS.labels(model_version=MODEL.version, status="error").inc()
        raise
    finally:
        LATENCY.observe(time.time() - t0)

start_http_server(9000)        # /metrics endpoint for Prometheus
```

</div>

<div class="level-next">
<span>Want drift detectors, delayed feedback, & alert tuning?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Two ways "broken" looks</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \underbrace{P(X) \;\text{changes}}_{\text{data drift}} \quad \text{or} \quad \underbrace{P(Y \mid X) \;\text{changes}}_{\text{concept drift}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Data drift: inputs look different but the relationship is the same

</li>
<li markdown="1">

Concept drift: same inputs, different optimal answer

</li>
<li markdown="1">

Detecting them requires different signals

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \underbrace{\text{input distribution changes}}_{\text{data drift}} \quad \text{or} \quad \underbrace{\text{right answer for each input changes}}_{\text{concept drift}} $$</span>

**In words.** Two distinct failure modes. `P(X)` is shorthand for "the probability distribution of inputs" — when it changes, your inputs start looking different from what you trained on (e.g. demographics shift, sensor readings drift). `P(Y | X)` is "the probability of the label given the input" — when it changes, the same input now has a different correct answer (e.g. user behaviour evolves, fashion changes). The underbraces (`⏟`) just label which piece is which. Each fails through different signals, so you need separate detectors.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`input distribution`the shape of the inputs your model receives

</li>
<li markdown="1">

`data drift`inputs look different but the input→answer rule is unchanged

</li>
<li markdown="1">

`concept drift`same inputs map to different answers than before

</li>
<li markdown="1">

Detecting them requires different signals

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Data drift detectors.** Compare incoming features against the training distribution. KS for numerical, chi-squared for categorical, PSI for binned numerical. Run on a rolling window; alert when above threshold. See [Data Validation](data-validation.html).

**Prediction drift.** The model's output distribution itself can drift even without obvious input drift — the model is making different decisions. Often the first signal of concept drift. Compare current prediction histograms against a baseline.

**Delayed feedback.** Many real systems have labels weeks later (loan defaults, click-through, fraud). Build a delayed-evaluation pipeline that joins predictions with labels when they arrive; recompute metrics on the lookback window. A dashboard that's behind by 30 days is still useful.

**Proxy metrics.** When no ground truth is available — engagement metrics, click-through, user feedback, downstream conversion. Calibrate proxies against periodic labelled audits.

**Tools.** Evidently, NannyML, Whylogs, Arize, Fiddler. Open-source for the first three; commercial dashboards for the last two. Most do data + prediction drift; Arize and Fiddler add subgroup analysis and explainability.

**Alert hygiene.** Tune thresholds so alerts fire rarely but reliably. Group related signals. Have a runbook ("if drift on feature X, do Y"). Test alerts periodically. False alarms train the team to ignore alerts.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import pandas as pd
from evidently.report import Report
from evidently.metrics import (
    DataDriftPreset, TargetDriftPreset, RegressionPreset,
)

# Daily drift report
report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=df_train, current_data=df_today)
report.save_html(f"drift_{today}.html")

# Programmatic — fetch the drift result, alert if needed
result = report.as_dict()
drift_share = result["metrics"][0]["result"]["dataset_drift"]
if drift_share:
    alert_slack("Data drift detected in production model")
```

</div>

<div class="level-next">
<span>Want shadow models, automated retraining triggers, & observability platforms?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Performance estimation without labels</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{CBPE: } \hat\mu = \frac{1}{n}\sum_i \mathbb{E}_{p(y|\hat y_i, x_i)}\!\big[\,\ell(y, \hat y_i)\big] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

NannyML's "Confidence-Based Performance Estimation"

</li>
<li markdown="1">

Uses the model's own calibrated probabilities to predict its error

</li>
<li markdown="1">

Surprisingly accurate when calibration holds

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{estimated error} \;=\; \text{average over all predictions of}\; \big(\text{expected loss using the model's own probabilities}\big) $$</span>

**In words.** When you don't have ground-truth labels yet, you can still estimate how your model is doing — using its own probabilities. For each prediction `ŷᵢ`, ask: "if the model says 0.8, what loss would I incur on average?" The `𝔼` (expectation, an average) is computed over the model's predicted probability distribution for that input. The `1/n ∑` just averages this expected loss across all `n` predictions. `ℓ` is the loss function (e.g. squared error, cross-entropy). Surprisingly accurate — provided the model's probabilities are calibrated (it really is wrong 20% of the time when it says 80% confidence).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`estimated error`NannyML's "Confidence-Based Performance Estimation"

</li>
<li markdown="1">

`average over predictions`summing then dividing by `n` (total predictions)

</li>
<li markdown="1">

`expected loss`weighting each possible outcome by the model's probability for it

</li>
<li markdown="1">

Surprisingly accurate when calibration holds

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Shadow models.** Run a candidate model alongside production; log both predictions. Compare their distributions and performance (when labels arrive). Identifies regressions before promotion. Standard for high-stakes domains.

**Automatic retraining triggers.** When drift alarms fire, kick off a retraining job. Conservative: alarm only, human decides. Aggressive: auto-retrain + auto-promote (with promotion gates). See [Automated Retraining](automated-retraining.html).

**Per-feature drift attribution.** When overall drift fires, which features are responsible? Greedy: sort features by individual KS statistic. Better: SHAP-on-drift — the contribution of each feature to the model's output shift.

**Calibration drift.** A model can have stable accuracy but drift in its predicted probabilities — confidence rises or falls relative to truth. Track per-bin observed accuracy vs predicted probability over time.

**Subgroup monitoring.** Aggregate metrics hide subgroup harm. Slice by demographic, geography, customer segment. Alert separately. See [Fairness](../fairness.html).

**Logging infrastructure.** Production ML monitoring is essentially "structured logging at scale + dashboards". Kafka for stream ingestion, ClickHouse / DuckDB / Snowflake for analytics, Grafana for dashboards, Alertmanager / PagerDuty for paging.

**Cost monitoring.** Inference is expensive — track $/request, GPU utilisation, request mix. Many shops over-provision GPUs because the costs are invisible at the model-developer level.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import nannyml as nml

# Performance estimation without labels (CBPE) — needs calibrated probabilities
estimator = nml.CBPE(
    y_pred_proba="y_proba",
    y_pred="y_pred",
    y_true="y_true",
    timestamp_column_name="ts",
    metrics=["accuracy", "roc_auc"],
    chunk_period="W",
)
estimator.fit(reference_data=df_reference)
results = estimator.estimate(df_production)
results.plot().show()

# Univariate drift by feature
drift = nml.UnivariateDriftCalculator(
    column_names=feature_cols,
    chunk_period="D",
).fit(reference_data=df_reference).calculate(df_production)
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

[Evidently AI <i class="fas fa-external-link-alt"></i>](https://www.evidentlyai.com/){: target="_blank" }
<span class="annotation">Open-source drift detection and model monitoring. Generates HTML reports; easy to drop into a CI / scheduled job.</span>

</li>
<li data-tier="indepth" markdown="1">

[NannyML <i class="fas fa-external-link-alt"></i>](https://nannyml.readthedocs.io/){: target="_blank" }
<span class="annotation">Specialises in performance estimation without ground truth. The CBPE method is unique and surprisingly effective.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Whylogs <i class="fas fa-external-link-alt"></i>](https://whylabs.ai/whylogs){: target="_blank" }
<span class="annotation">Logging-style data profiling library. Lightweight; designed to capture statistical fingerprints of every batch your model sees.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Chip Huyen — Data Distribution Shifts <i class="fas fa-external-link-alt"></i>](https://huyenchip.com/2022/04/27/data-distribution-shifts.html){: target="_blank" }
<span class="annotation">Excellent overview of the kinds of shift production systems face and how to detect them.</span>

</li>
</ul>

</div>

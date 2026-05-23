---
title: Imbalanced Data — ML Resources Hub
eyebrow_text: ← Theory · Data &amp; Features
eyebrow_href: ../theory.html
heading: Imbalanced Data
lead: When the rare class is the one you care about — fraud, disease, anomalies.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**"99% accuracy" is meaningless when 99% of cases are negative.** A constant predictor wins. The interesting class is the rare one — fraud, cancer, anomalies — and the model needs to actually find it. The fixes: re-weight the loss, resample the data, change the decision threshold, or change the metric you optimise.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide the class ratio · toggle the fix · watch a logistic regression go from "predict everything negative" to "actually find the rare class"</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                positive %
                <input id="viz-imb-ratio" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-imb-none" type="button" class="active">No fix</button>
<button id="viz-imb-weight" type="button">Class weights</button>
<button id="viz-imb-over" type="button">Oversample</button>
<button id="viz-imb-under" type="button">Undersample</button>
<span class="viz-classic-badge" id="viz-imb-r-lbl">5%</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-imb-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-imb-caption"></div>
</div>

<script src="{{root}}js/viz/imbalanced.js"></script>

A logistic-regression boundary on a 2D dataset with adjustable class imbalance. With **no fix** and 5% positives, the model often collapses to a tiny region (or misses the positives entirely) while reporting 95% accuracy. **Class weights** tells the loss "false negatives cost more". **Oversample** duplicates positives to even out the loss. **Undersample** drops negatives. All three pull the boundary back into useful territory.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Re-weight the loss.** Multiply each positive example's contribution by *N_neg / N_pos* (or use `class_weight="balanced"` in sklearn). Cheapest fix; usually the first thing to try.

**Oversample the minority.** Duplicate (or SMOTE-synthesize) positive examples until the classes are balanced. Risk: with naive duplication, the model can memorize the few real positives. SMOTE interpolates between nearest neighbours, less risky.

**Undersample the majority.** Drop most negatives. Fast, but you lose information. Useful when negatives are abundant and computation is the bottleneck.

**Change the threshold.** Train the model normally, then choose the operating point that maximises your real metric (F1, recall at fixed precision, expected cost). Often the simplest fix and the only one that matters at deployment time.

**Change the metric.** Stop reporting accuracy. Use F1, precision-recall AUC, or a cost-sensitive metric that reflects your real-world preferences over FP vs FN.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- One class is < 10% of the data
- False negatives are much more expensive than false positives (or vice versa)
- The model is reporting high accuracy but useless recall
- Cost-sensitive deployment (medical, financial, security)

</div>

<div class="no" markdown="1">

### Watch out

- Resampling inside training data only — never the validation/test set
- SMOTE-ing tabular categories produces nonsense — use only on continuous features
- Class weights interact oddly with calibration — recalibrate after
- Heavy oversampling can lead to overfitting on the duplicated points

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Easiest: balanced class weights
clf = LogisticRegression(class_weight="balanced", max_iter=1000)

# Pipeline that resamples ONLY the training fold (imblearn handles this)
pipe = Pipeline([
    ("smote", SMOTE(sampling_strategy=0.5, k_neighbors=5)),
    ("clf",   LogisticRegression(max_iter=1000)),
])

# Cost-sensitive threshold — pick at deployment
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_val, p_val)
f1 = 2 * p * r / (p + r + 1e-12)
tau_best = thresholds[f1.argmax()]
```

</div>

<div class="level-next">
<span>Want SMOTE variants, focal loss, and one-class methods?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Cost-sensitive Bayes classifier</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \hat y = 1 \iff p(y = 1 \mid x) > \frac{c_{FP}}{c_{FP} + c_{FN}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`cFP`cost of a false positive; `cFN` of a false negative

</li>
<li markdown="1">

If *c<sub>FN</sub>* is high (catching fraud), threshold drops well below 0.5

</li>
<li markdown="1">

Optimal decisions just need calibrated probabilities + costs

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{predict positive when} \quad p(\text{positive} \mid x) \;>\; \frac{\text{cost of false positive}}{\text{cost of false positive} \;+\; \text{cost of false negative}} $$</span>

**In words.** The optimal decision rule when you know how expensive each kind of mistake is. Compute the probability that the example is positive; predict positive only if that probability beats a threshold determined entirely by the cost ratio. If false negatives are much worse than false positives (e.g. cancer screening — missing a case is catastrophic), the threshold drops well below 0.5 and you catch more positives at the price of more false alarms. Calibrated probabilities and explicit costs replace ad-hoc threshold tuning.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`p(positive | x)`calibrated probability that the example is positive

</li>
<li markdown="1">

`cost of false positive`how bad it is to wrongly flag a negative as positive

</li>
<li markdown="1">

`cost of false negative`how bad it is to miss a real positive

</li>
<li markdown="1">

If missing positives is costly, the threshold drops well below 0.5

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**SMOTE and its descendants.** SMOTE interpolates between minority points and their k-nearest minority neighbours. Borderline-SMOTE focuses on the decision boundary; ADASYN weights synthetic samples toward harder cases. All assume continuous, meaningful Euclidean distance — they don't work for categorical or text features.

**Focal loss.** Lin et al. (2017). Multiplies cross-entropy by *(1 − p_t)^γ*, which down-weights easy examples. Lets the model concentrate gradient on hard cases. Originally for object detection; useful for any extreme imbalance.

**Two-stage cascades.** Train a fast, conservative classifier to filter out obvious negatives; pass the rest to an expensive precise model. Lets you use different fixes at each stage. Standard in real-world fraud and anomaly detection.

**Anomaly detection framing.** When positives are very rare (< 0.1%), classification is the wrong frame. Treat positives as anomalies — train on negatives only, flag points the model can't reconstruct or that have low density. See the Anomaly Detection page.

**Beware naive metrics.** ROC-AUC can stay high even when a model is useless on the minority class — flip to PR-AUC, which is sensitive to base-rate shifts. F1 weighs precision and recall equally; F2 weights recall more (use it when missing positives is worse).

**Calibration after resampling.** Oversampling and class weights change the model's base rate and its output probabilities are biased toward the resampled ratio. Post-hoc calibration on a held-out fold (Platt or isotonic) fixes this.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn.functional as F

# Focal loss for binary classification with extreme imbalance
def focal_loss(logits, y, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p   = torch.sigmoid(logits)
    pt  = y * p + (1 - y) * (1 - p)
    w   = alpha * y + (1 - alpha) * (1 - y)
    return (w * (1 - pt).pow(gamma) * bce).mean()

# Calibrate after resampling — Platt on a held-out fold
from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(resampled_model, method="isotonic", cv="prefit")
cal.fit(X_calib, y_calib)
```

</div>

<div class="level-next">
<span>Want recursive sub-sampling, MWMOTE, one-class, and synthetic data?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Posterior under resampling</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ p_{\text{true}}(y=1 \mid x) = \frac{p_{\text{resamp}}(y=1\mid x)\,\pi_{\text{true}}}{p_{\text{resamp}}(y=1\mid x)\,\pi_{\text{true}} + (1 - p_{\text{resamp}}(y=1\mid x))\,(1 - \pi_{\text{true}})} \cdot \frac{1}{\pi_{\text{resamp}}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`π`base rate (true vs resampled)

</li>
<li markdown="1">

The posterior under resampling needs a Bayes correction to recover true probabilities

</li>
<li markdown="1">

This is why models trained on resampled data are mis-calibrated by default

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{true probability of positive} \;=\; \text{Bayes-corrected version of the resampled probability, using the true and resampled base rates} $$</span>

**In words.** When you train on a resampled dataset (e.g. 50/50 after oversampling positives) the model learns probabilities that match the 50/50 world it saw — not the true 1/99 world your test set comes from. The formula gives the Bayes correction: combine the resampled probability with the *true* base rate `π_true` and the *resampled* base rate `π_resamp` to recover what the model would have said had it seen the real ratio. This is why models trained on resampled data are mis-calibrated by default, and why calibration on a held-out fold with the natural ratio is essential.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`p_resamp`probability the model outputs when trained on resampled data

</li>
<li markdown="1">

`π_true`true base rate (e.g. 1% positives in the wild)

</li>
<li markdown="1">

`π_resamp`base rate in the resampled training set (e.g. 50%)

</li>
<li markdown="1">

Recovering the true probability requires a Bayes correction; otherwise the model is mis-calibrated

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Beyond SMOTE.** Variants like Borderline-SMOTE-1/2, MWMOTE (majority-weighted minority), and SVMSMOTE focus synthetic examples near the boundary or where the model is uncertain. All share the same Euclidean assumption.

**Cluster-based methods.** Cluster the majority class, keep only representatives from each cluster (e.g. NearMiss, TomekLinks). Reduces dataset size without losing as much information as random undersampling.

**Cost-sensitive learning.** Build the cost matrix directly into the loss / split criterion. MetaCost (Domingos, 1999) and instance-weighted boosting are classical examples; modern deep nets use class-weighted or cost-sensitive loss functions.

**One-class learning.** When positives are essentially absent at training time, model only the negatives and flag anything that doesn't fit. One-class SVM, Isolation Forest, deep autoencoders for reconstruction error — all standard anomaly-detection tools.

**Synthetic / generative data.** When real positives are rare and expensive, GANs or diffusion models trained on the small positive set can supply additional training data. Quality is hard to verify; the practical danger is generating subtly off-distribution samples that the model overfits to.

**Subgroup imbalance.** A model can be well-calibrated overall but mis-calibrated on a subgroup defined by sensitive attributes. Imbalance compounds with fairness concerns; both have to be analysed together (see the Fairness page).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Borderline-SMOTE — interpolate only near the decision boundary
b = BorderlineSMOTE(kind="borderline-2", k_neighbors=5)
X_bal, y_bal = b.fit_resample(X_train, y_train)

# SMOTE + Tomek-link cleaning — generate synthetic, then drop noisy pairs
st = SMOTETomek(sampling_strategy="auto")
X_clean, y_clean = st.fit_resample(X_train, y_train)

# One-class anomaly detection
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(contamination=0.01, random_state=0)
y_score = iforest.fit(X_neg).score_samples(X_test)
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

[imbalanced-learn <i class="fas fa-external-link-alt"></i>](https://imbalanced-learn.org/){: target="_blank" }
<span class="annotation">The reference Python library for resampling. SMOTE, ADASYN, NearMiss, combined methods — with sklearn-compatible API.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Chawla et al. (2002) — SMOTE <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1106.1813){: target="_blank" }
<span class="annotation">The original SMOTE paper. Surprisingly readable, with empirical results across many datasets.</span>

</li>
<li data-tier="indepth" markdown="1">

[Lin et al. (2017) — Focal Loss <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1708.02002){: target="_blank" }
<span class="annotation">Focal loss for object detection — same paper referenced from Loss Functions. The motivation chapter is the best intuition pump for class imbalance.</span>

</li>
<li data-tier="intuition" markdown="1">

[Brownlee — Tactics to Combat Imbalanced Classes <i class="fas fa-external-link-alt"></i>](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/){: target="_blank" }
<span class="annotation">Practical recipe-style guide that's a good first read for getting unstuck on a real problem.</span>

</li>
</ul>

</div>

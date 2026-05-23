---
title: Classification Metrics — ML Resources Hub
eyebrow_text: ← Theory · Core Concepts
eyebrow_href: ../theory.html
heading: Classification Metrics
lead: Accuracy, precision, recall, F1, ROC, AUC — when each one is the right thing to report.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Accuracy alone hides too much.** On a 95% / 5% imbalanced dataset, a model that predicts the majority class always is 95% accurate — and useless. You need at least two numbers: precision (when I say positive, am I right?) and recall (of all the positives, how many did I find?). Threshold matters too: slide it and every metric moves.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide the threshold — confusion matrix, precision, recall, F1, and the operating point on the ROC all move together</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                threshold
                <input id="viz-clm-threshold" class="viz-classic-slider" type="range"></input>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Imbalance
                <select id="viz-clm-imbalance"></select>
</label>
<button id="viz-clm-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-clm-thr-lbl">τ = 0.50</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-clm-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-clm-caption"></div>
</div>

<script src="{{root}}js/viz/classification-metrics.js"></script>

Left: predicted score distributions for the two classes. The vertical line is the threshold — everything to its right is predicted positive. Top-right: the confusion matrix. Bottom-right: the ROC curve with the current operating point marked. Slide τ and watch everything move — drop τ to catch more positives (higher recall, lower precision); raise τ for the opposite. Try the *10% imbalance* preset and notice how the optimal F1 threshold drifts away from 0.5.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Accuracy** = (TP + TN) / total. Useful when classes are balanced and costs are equal. Often misleading otherwise.

**Precision** = TP / (TP + FP). "When I predict positive, how often am I right?" Care when false positives are expensive (spam → important email in junk).

**Recall** (sensitivity) = TP / (TP + FN). "Of all the actual positives, how many did I find?" Care when false negatives are expensive (cancer screening, fraud).

**F1** = harmonic mean of precision and recall. Single number; punishes the worse of the two. Default for imbalanced classification reports.

**ROC-AUC** = probability that a random positive is scored higher than a random negative. Threshold-independent; useful for ranking and overall discriminative power. Use PR-AUC for highly imbalanced data — ROC-AUC can stay high even when the model is useless on the minority class.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- **Accuracy**: balanced classes, equal costs
- **Precision / Recall**: care about FP vs FN asymmetry
- **F1**: imbalanced classes, single-number summary
- **ROC-AUC**: ranking quality, threshold not fixed
- **PR-AUC**: highly imbalanced classes (1% positives)

</div>

<div class="no" markdown="1">

### Watch out when

- Accuracy on imbalanced data is misleading
- ROC-AUC on very imbalanced data is misleading
- F1 gives equal weight to precision and recall — use Fβ if you don't
- Macro vs micro vs weighted matters for multi-class

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix,
)

# One-number summaries
acc = accuracy_score(y_true, y_pred)
p   = precision_score(y_true, y_pred)
r   = recall_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred)

# Threshold-independent (need probabilities, not labels)
auc      = roc_auc_score(y_true, y_prob)
ap       = average_precision_score(y_true, y_prob)   # PR-AUC; better for imbalance

# The big picture
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# Multi-class: macro vs micro vs weighted
f1_macro    = f1_score(y_true, y_pred, average="macro")     # mean of per-class F1
f1_micro    = f1_score(y_true, y_pred, average="micro")     # = accuracy for multiclass
f1_weighted = f1_score(y_true, y_pred, average="weighted")  # by support
```

</div>

<div class="level-next">
<span>Want PR-curves, cost-sensitive thresholds, and multi-class metrics?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Fβ score</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`β = 1`standard F1 — equal weight

</li>
<li markdown="1">

`β > 1`weights recall more (e.g. F2)

</li>
<li markdown="1">

`β < 1`weights precision more (e.g. F0.5)

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ F_\beta \;=\; (1 + \beta^2) \times \frac{\text{precision} \times \text{recall}}{\beta^2 \times \text{precision} \;+\; \text{recall}} $$</span>

**In words.** Fβ is a weighted blend of precision and recall into a single number. The parameter `β` (beta) controls which side you care about more: `β = 1` gives equal weight (this is just F1); a bigger β tilts toward recall (you really don't want to miss positives); a smaller β tilts toward precision (you really don't want to cry wolf). The slightly odd weighted form is the *harmonic* mean — it punishes the worse of the two scores, so you can't trade away one entirely.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`precision`fraction of "predicted positive" that are actually positive

</li>
<li markdown="1">

`recall`fraction of actual positives the model caught

</li>
<li markdown="1">

`β = 1`standard F1 — equal weight

</li>
<li markdown="1">

`β > 1`weights recall more (e.g. F2 for medical screening)

</li>
<li markdown="1">

`β < 1`weights precision more (e.g. F0.5 for spam filtering)

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Picking the right threshold.** The model outputs a probability; you choose where to cut it. 0.5 is rarely optimal — pick the threshold that maximises F1 (or Fβ for asymmetric costs) on a validation set. For deployment, freeze that threshold and report performance at it.

**Cost-sensitive thresholds.** If FP costs *c<sub>FP</sub>* and FN costs *c<sub>FN</sub>*, the optimal threshold (under a Bayes-decision argument) is τ* = c<sub>FP</sub> / (c<sub>FP</sub> + c<sub>FN</sub>). For cancer screening (FN is catastrophic), τ shifts way below 0.5.

**PR vs ROC.** ROC plots TPR vs FPR; PR plots precision vs recall. Both span the same threshold range. On heavily imbalanced data, ROC can be deceptive — a high AUC can hide poor minority-class performance. PR-AUC is the right summary there.

**Multi-class.**
*Macro*-averaged F1 = mean of per-class F1 — treats every class equally.
            *Micro*-averaged F1 = aggregate TP/FP/FN across classes first — dominated by big classes.
            *Weighted* = macro but weighted by class support.
            Pick macro if you care equally about every class (rare-class performance matters); micro if you don't (mass matters).

**Multi-label vs multi-class.** Multi-class: one label per example, K options. Multi-label: any subset of K. Metrics generalise differently — Hamming loss, subset accuracy, per-label F1.

**Top-k accuracy.** Was the right class in the top k predictions? Standard for ImageNet (top-5). Useful when there's structural ambiguity ("dog breed" could be any of several reasonable guesses).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

# Find the threshold that maximises F1 on validation
p, r, thresholds = precision_recall_curve(y_val, p_val)
f1_arr           = 2 * p * r / (p + r + 1e-12)
best_idx         = np.argmax(f1_arr)
tau_best         = thresholds[best_idx]
print(f"Best τ = {tau_best:.2f}  F1 = {f1_arr[best_idx]:.3f}")

# Cost-sensitive choice — FN costs 10x FP
c_fp, c_fn = 1, 10
tau_cost   = c_fp / (c_fp + c_fn)             # → 0.09
y_hat_cost = (p_test >= tau_cost).astype(int)
```

</div>

<div class="level-next">
<span>Want expected calibration, Brier decomposition, & proper scoring?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Brier score decomposition</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{Brier} = \underbrace{\text{Reliability}}_{\text{miscalibration}} - \underbrace{\text{Resolution}}_{\text{discrimination}} + \underbrace{\text{Uncertainty}}_{\text{class entropy}} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Lower Brier score is better

</li>
<li markdown="1">

Reliability ↓ if predicted probabilities match observed frequencies

</li>
<li markdown="1">

Resolution ↑ if the model assigns different probabilities to different cases

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{Brier score} \;=\; \text{miscalibration} \;-\; \text{discrimination} \;+\; \text{class uncertainty} $$</span>

**In words.** The Brier score (mean squared error between predicted probabilities and actual labels) breaks cleanly into three pieces. *Reliability* is how well your stated probabilities match observed frequencies — when you say "70% likely", do 70% of those cases actually happen? Lower is better. *Resolution* is how varied your predictions are — a model that always says "50%" has no resolution. Higher is better, which is why it's subtracted. *Uncertainty* is the irreducible difficulty of the prediction (entropy of the labels). Lower Brier overall is better — and the decomposition tells you whether to fix your calibration or your discrimination.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Brier score`average squared error between predicted probabilities and labels

</li>
<li markdown="1">

`miscalibration`how badly your probabilities match observed frequencies

</li>
<li markdown="1">

`discrimination`how much your model varies its predictions across cases

</li>
<li markdown="1">

`class uncertainty`baseline entropy of the labels — unavoidable

</li>
<li markdown="1">

Lower Brier is better; the decomposition tells you which piece needs fixing

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Brier score & proper scoring.** Brier = mean squared error between predicted probabilities and labels (one-hot). A "proper" scoring rule is one whose expected value is minimised by the true probability — Brier and log-loss are both proper; accuracy isn't. Always optimise a proper scoring rule.

**Expected calibration error (ECE).** Bin predictions by predicted probability; in each bin compare the average predicted probability to the fraction actually positive. Average the gap. Useful — but its dependence on binning makes it brittle; alternatives include MCE (max calibration error), and calibration-aware scoring rules.

**Calibration vs discrimination.** A model can have great AUC but terrible calibration (good ranking but wrong probabilities) and vice versa. They're orthogonal. Post-hoc calibration (Platt scaling, isotonic regression) fixes calibration without changing the model's discrimination — useful for deploying a model whose scores you want to interpret.

**Class-conditional metrics.** For very imbalanced problems, look at per-class precision / recall / F1 separately. The overall macro-F1 hides cases where one class is great and another is awful.

**Threshold-independent metrics under shift.** AUC and AP are invariant to monotone score transformations — useful for comparing models that score differently. But neither is invariant to base-rate shift. If the production prior changes from training, post-hoc re-calibration is required.

**Cohen's κ and MCC.** Cohen's kappa adjusts accuracy for the chance level. Matthews Correlation Coefficient is a balanced metric on the confusion matrix; symmetric, robust to imbalance. Both are sometimes preferred for biological / medical applications.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import numpy as np
from sklearn.metrics import brier_score_loss, matthews_corrcoef

# Brier score — lower is better, proper, sensitive to calibration
brier = brier_score_loss(y_true, p_pred)

# Matthews correlation coefficient — balanced, robust to imbalance
mcc = matthews_corrcoef(y_true, y_pred)

# Expected calibration error (ECE) — quick implementation
def ece(y_true, p_pred, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0: continue
        conf = p_pred[mask].mean()                 # average predicted prob
        acc  = (y_true[mask] == 1).mean()          # observed positive rate
        e += (mask.sum() / len(y_true)) * abs(conf - acc)
    return e
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

[scikit-learn — Model Evaluation Guide <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/model_evaluation.html){: target="_blank" }
<span class="annotation">Best practical reference for which metric to use when. Has well-chosen examples and pitfalls.</span>

</li>
<li data-tier="intuition" markdown="1">

[Kaggle — Plotting a Confusion Matrix <i class="fas fa-external-link-alt"></i>](https://www.kaggle.com/code/grfiv4/plot-a-confusion-matrix){: target="_blank" }
<span class="annotation">Pragmatic primer with multi-class examples — useful for teaching new colleagues.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Frank Harrell — Classification vs Prediction <i class="fas fa-external-link-alt"></i>](https://www.fharrell.com/post/classification/){: target="_blank" }
<span class="annotation">Opinionated argument that we should report calibrated probabilities, not thresholded classifications. Worth reading even (especially) if you disagree.</span>

</li>
<li data-tier="indepth" markdown="1">

[Guo et al. (2017) — Calibration of Modern Neural Nets <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1706.04599){: target="_blank" }
<span class="annotation">Showed modern networks are miscalibrated despite high accuracy, and recommended temperature scaling as a fix.</span>

</li>
</ul>

</div>

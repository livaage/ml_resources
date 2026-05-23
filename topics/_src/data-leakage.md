---
title: Data Leakage — ML Resources Hub
eyebrow_text: ← Theory · Data &amp; Features
eyebrow_href: ../theory.html
heading: Data Leakage
lead: The silent killer — when the model "knows" something it couldn't possibly know at deployment.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**If your validation score is suspiciously good, something probably leaked.** Data leakage is when information from outside the training set sneaks into training — the test set's mean used in preprocessing, a feature only available *after* the prediction time, a near-duplicate of the target. The model learns to exploit it; deployment fails silently.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Compare two pipelines doing the "same" thing — one leaks the test statistics through the scaler</span>
</div>
<div class="viz-classic-controls">
<button id="viz-leak-wrong" type="button" class="active">Naive pipeline (leaks)</button>
<button id="viz-leak-right" type="button">Pipeline-correct (no leak)</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-leak-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-leak-caption"></div>
</div>

<script src="{{root}}js/viz/leakage.js"></script>

Both panels run the same operations: fit a scaler, fit a model, predict on a held-out set. The **naive** pipeline fits the scaler on *all* the data before splitting — so the test set's mean and std are baked into training. The **correct** pipeline fits the scaler only on train. The reported test accuracy differs by several percentage points; in real deployment the naive number is fiction.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Preprocessing on the full dataset.** The most common leak. Fitting `StandardScaler`, `SimpleImputer`, `PCA`, or any encoder on the full dataset (including val/test) before splitting leaks statistics. Always fit on train only; use scikit-learn `Pipeline` to enforce it.

**Target leakage.** A feature that's a near-duplicate of the target. Often subtle — "was the loan paid off?" as a feature when predicting whether the loan is good. The model gets perfect train and val scores, then the feature is unavailable in production.

**Temporal leakage.** Random splits on time-series data put the future in train. The model "learns" to use future information. Always use time-based splits for any data with a time dimension.

**Group leakage.** Multiple records per entity (patient, user, session) split across train and val. The model memorises entity-specific quirks and reports good val performance — but a new entity at deployment is fundamentally unseen.

**Hyperparameter leakage.** Tuning extensively on the test set. After enough trials, you've selected for performance on that test set; the number is no longer an unbiased estimate of true performance.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Common sources of leakage

- Scaler / imputer / encoder fit on full dataset
- Target encoding without CV
- Rolling aggregates that include the prediction time
- "Did the customer churn this month?" as a feature for predicting churn next month
- Random splits on time-series
- De-duplication that misses near-duplicates across train/test

</div>

<div class="no" markdown="1">

### Defences

- Use scikit-learn `Pipeline` + `ColumnTransformer`
- Time-based splits for any temporal data
- GroupKFold for non-iid records
- Audit features for "information unavailable at prediction time"
- Look at top features by importance — anything suspicious?
- Try a "shuffled-labels" sanity check — should give ~baseline performance

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold

# WRONG — scaler sees test data
X_scaled = StandardScaler().fit_transform(X)            # leak!
X_train, X_test, ... = train_test_split(X_scaled, y, ...)

# RIGHT — pipeline fits inside each CV fold
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression()),
])
scores = cross_val_score(pipe, X, y, cv=5)

# RIGHT — group split for non-iid data
gkf = GroupKFold(n_splits=5)
cross_val_score(pipe, X, y, groups=groups, cv=gkf)
```

</div>

<div class="level-next">
<span>Want target leakage detection, temporal joins, and adversarial validation?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Information-available-at-prediction-time</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ x_i^{(t)} \;\text{is valid iff}\; x_i^{(t)} \in \mathcal{F}_{t^-} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

A feature is valid only if it's measurable in the σ-algebra of past information

</li>
<li markdown="1">

Every "rolling 30-day average" feature must end strictly before the target window starts

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{feature at time } t \;\text{is valid only if it can be computed from data strictly before } t $$</span>

**In words.** The rule of thumb for avoiding temporal leakage: ask, "could this feature be computed at prediction time, using only data already observed?" The mathematical version uses *σ-algebras* — a formal way to say "the set of all things knowable up to a given time" — but the intuition is what counts. Any rolling average, any aggregation, any join must end strictly before the target's timestamp. If you can't compute the feature at deployment from then-available data, you can't use it.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`x at time t`a feature value associated with prediction time *t*

</li>
<li markdown="1">

`data before t`only data that was already observed at time *t*

</li>
<li markdown="1">

Every "rolling 30-day average" feature must end strictly before the target window starts

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Temporal leakage in joins.** When joining tables, every aggregation must use only data available *before* the target event. "Total spend per user" computed once on the full dataset leaks future transactions. Point-in-time correctness — every feature has a timestamp, every join respects it.

**Target encoding done right.** Replace each category with its mean target value, but compute the mean *inside CV folds*. Without that, the encoding for the validation rows includes the validation rows' own targets.

**Adversarial validation.** Train a binary classifier to distinguish train rows from test rows. If accuracy is much above 0.5, your train and test are systematically different (distribution shift). Look at which features the classifier uses — those are the leaks or the shifts.

**Near-duplicates.** Two rows that are very close in feature space but assigned to train and test will inflate accuracy. Common in scraped web data, image datasets, and any dataset with multiple records per entity. De-duplicate carefully (perceptual hashes for images, fuzzy matching for text).

**Leakage detection.** Inspect top features by importance. If one is suspiciously informative ("status field" predicts "did the application succeed"), that's a leak. Permute the suspect feature — if performance crashes, the model relies on it; check whether it's available at prediction time.

**Shuffled-label baselines.** Train a model on shuffled labels. If performance is meaningfully above the base rate, your features carry information they shouldn't (data leakage or a bug in the shuffling).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Adversarial validation — train vs test as a binary classifier
combined = np.r_[X_train, X_test]
is_test  = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
auc = cross_val_score(RandomForestClassifier(), combined, is_test,
                       scoring="roc_auc", cv=5).mean()
print(f"Train-vs-test AUC: {auc:.3f}")
# > 0.6 ⇒ distribution shift / leakage worth investigating

# Shuffled-label sanity check — should be at chance
import numpy as np
y_shuf = np.random.permutation(y_train)
print(f"Shuffled: {cross_val_score(model, X_train, y_shuf, cv=5).mean():.3f}")
# Above base rate ⇒ leakage somewhere
```

</div>

<div class="level-next">
<span>Want competitions, post-mortems, and production leakage?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Leakage detection via permutation importance</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ I(x_j) = \text{err}(\text{shuffle}_j(X), y) - \text{err}(X, y) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Shuffle feature *j*; if the error spikes, the model depends heavily on it

</li>
<li markdown="1">

Combined with a "should this feature even be here" check, finds many leaks

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{importance of feature } j \;=\; \text{(error with feature } j \text{ shuffled)} \;-\; \text{(error with feature } j \text{ intact)} $$</span>

**In words.** A feature's importance is "how much worse does the model get if I scramble that feature?" Randomly permute the values of column *j* across rows — that breaks any association between the feature and the target — and re-evaluate. If the error jumps a lot, the model was really relying on that feature; if the error barely moves, the feature was decorative. Combined with a domain check ("is this feature even available at prediction time?"), this finds many leaks: a suspiciously powerful feature that wouldn't exist in production is almost always a leak.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`feature j`the column you're probing

</li>
<li markdown="1">

`shuffle`randomly permute the column across rows, breaking its link to the target

</li>
<li markdown="1">

`error`any metric of how badly the model does (loss, 1 − accuracy, etc.)

</li>
<li markdown="1">

A suspiciously important feature is often a leak — check whether it could exist at prediction time

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Famous competition leaks.** Heritage Provider Network: zip code leaked hospital outcomes. KDD Cup 2008: medical-record metadata had identifiers correlating with diagnosis. The pattern is the same — a benign-looking feature carries information that wouldn't exist at deployment.

**Production leakage.** Train-serve skew is essentially distributional leakage at deploy time. A feature pipeline that worked on the offline join can fail to be replicated online — features get computed differently. Feature stores (Feast, Tecton) exist to enforce consistency.

**Pre-training leakage.** Foundation models pre-trained on huge corpora can have seen benchmark test sets during pre-training. Reported zero-shot or few-shot scores on those benchmarks may be inflated. Decontamination procedures (matching benchmark items against pre-training data and removing them) are increasingly standard but imperfect.

**Differential privacy as a defence.** If your training loss is differentially private (bounded influence of any single example), it's bounded leakage by construction. Comes at an accuracy cost; practical for some sensitive domains.

**Leakage vs. concept drift.** Both cause "validated model performs worse in production". Distinguish them: leakage gives inflated *validation* performance (impossible to detect from validation alone); drift gives valid validation but degraded production. Different remedies.

**Auditing for hidden labels.** When you suspect a leak, ablate each feature in turn and see which ones, when removed, fix the suspiciously-high score. Combined with domain knowledge ("could this feature be observed at decision time?"), this isolates the leak.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)
for i in result.importances_mean.argsort()[::-1][:10]:
    print(f"{X_val.columns[i]:40s} {result.importances_mean[i]:.3f}"
          f" ± {result.importances_std[i]:.3f}")
# Inspect top features: which ones could not exist at prediction time?
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
<li data-tier="intuition" markdown="1">

[Becker — Data Leakage Mini-Course (Kaggle) <i class="fas fa-external-link-alt"></i>](https://www.kaggle.com/code/dansbecker/data-leakage){: target="_blank" }
<span class="annotation">Compact, practical introduction with real competition examples. Excellent first read.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Brownlee — Data Leakage in Machine Learning <i class="fas fa-external-link-alt"></i>](https://machinelearningmastery.com/data-leakage-machine-learning/){: target="_blank" }
<span class="annotation">Cookbook-style article covering the common leakage patterns and detection strategies.</span>

</li>
<li data-tier="indepth" markdown="1">

[Kapoor & Narayanan — Leakage and the Reproducibility Crisis in ML <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2210.01437){: target="_blank" }
<span class="annotation">Large-scale empirical study showing how widespread leakage is in published ML research across many fields.</span>

</li>
<li data-tier="indepth" markdown="1">

[Feast — Feature Store <i class="fas fa-external-link-alt"></i>](https://www.feast.dev/){: target="_blank" }
<span class="annotation">Reference open-source feature store. Point-in-time correctness is a first-class feature — eliminates a whole class of production leakage.</span>

</li>
</ul>

</div>

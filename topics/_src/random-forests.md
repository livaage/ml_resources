---
title: Random Forests — ML Resources Hub
eyebrow_text: ← Theory · Tree-Based
eyebrow_href: ../theory.html
heading: Random Forests
lead: An ensemble of decorrelated decision trees, averaged or voted.
prev_href: decision-trees.html
prev_title: Decision Trees
next_href: gradient-boosting.html
next_title: Gradient Boosting
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Ask a crowd of decision trees, then take the majority vote.** Each tree alone is shaky and makes its own mistakes, but the trees disagree in different places — so when you average their answers, the errors cancel out and the agreement points to the truth.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide N up — watch the ensemble boundary smooth out compared to the brittle individual trees below</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Data
                <select id="viz-rf-data"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Trees
                <input id="viz-rf-n" class="viz-classic-slider" type="range"></input>
</label>
<select id="viz-rf-depth" aria-label="depth"></select>
<button id="viz-rf-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-rf-n-lbl">N = 20</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-rf-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-rf-caption"></div>
</div>

<script src="{{root}}js/viz/random-forests.js"></script>

Each tree below was trained on a different bootstrap sample of the data with a random feature subset at every split — so they're all biased, all brittle, all wrong in different ways. But the big panel up top is the *average* of their votes. The wrongness cancels; the agreement reinforces. Slide N from 1 to 80 and watch the boundary go from jagged-and-overconfident to smooth-and-stable.
{: .viz-intro }

<article class="tldr-body" markdown="1">

A **decision tree** is a flowchart of yes/no questions ("Is age > 30?" → "Is income > 50k?" → …) that ends in a prediction. It's intuitive, but it's also brittle: small changes in your training data can produce a wildly different tree.

A **random forest** builds many such trees, each on a slightly different sample of your data, and lets them vote. It's a bit like asking 100 doctors for a second opinion instead of trusting one — the consensus is more reliable than any individual.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Your data is in rows and columns (a spreadsheet)
- You want something that "just works" with minimal setup
- You're not sure where to start
- You want a reasonable accuracy estimate without extra effort

</div>

<div class="no" markdown="1">

### Skip it when

- You need to explain why a specific prediction came out a certain way
- Your data is images, text, or sequences (use neural networks)
- You need predictions outside the range you trained on

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.ensemble import RandomForestClassifier

# Train: just give it labelled data
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

</div>

<div class="level-next">
<span>Want to know <em>how</em> it actually works?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Key idea</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \hat{y}(x) \;=\; \frac{1}{B} \sum_{b=1}^{B} T_b(x) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`ŷ(x)`the forest's prediction for input *x*

</li>
<li markdown="1">

`B`the number of trees (typically 100–500)

</li>
<li markdown="1">

`Tb(x)`the prediction from the *b*-th tree

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{forest prediction} \;=\; \text{average of all tree predictions} $$</span>

**In words.** Ask every tree what it thinks about input *x*, then take the average of their answers (for classification, that's a majority vote; for regression, a numeric mean). `B` is how many trees you have — usually a few hundred. Each individual tree `Tb` was trained on a bootstrap re-sample of the data while only considering a random subset of features at every split, so the trees are deliberately decorrelated. Averaging cancels out their independent errors while preserving the signal they all agree on.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`forest prediction`the ensemble's final answer at input *x*

</li>
<li markdown="1">

`tree predictions`each individual tree's vote or value for *x*

</li>
<li markdown="1">

`B`number of trees in the forest (typically 100–500)

</li>
</ul>

</div>

Each tree is trained on a **bootstrap sample** with a **random feature subset** at each split. Both sources of randomness decorrelate the trees so the average cuts variance.

</div>

<article class="tldr-body" markdown="1">

A single decision tree is high-variance — small changes to the training data produce very different trees. Random forests fight this by training many trees on slightly different views of the data, then averaging their predictions.

Two knobs do the work. **Bootstrap sampling** means each tree sees a different random sample (with replacement) of the training set. **Feature subsetting** means each split only considers a random subset of features (typically √*p* for classification, *p*/3 for regression). Together they ensure the trees disagree in independent ways.

Bonus: each bootstrap leaves out about ⅓ of the data per tree. Aggregating the trees that excluded each point gives an **out-of-bag (OOB) estimate** of generalization error — no cross-validation needed.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Tabular data with mixed feature types
- You need a strong baseline with almost no tuning
- Robustness matters more than the last 2% of accuracy
- You want a free OOB error estimate

</div>

<div class="no" markdown="1">

### Skip it when

- Extrapolation outside the training range is required
- Monotonicity constraints are required
- You need per-prediction interpretability
- High-signal tabular — gradient boosting usually wins

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",   # √p features considered per split
    oob_score=True,        # free generalization estimate
    n_jobs=-1,
    random_state=0,
)
clf.fit(X_train, y_train)

print(f"OOB accuracy: {clf.oob_score_:.3f}")
for name, imp in sorted(
    zip(X_train.columns, clf.feature_importances_),
    key=lambda x: -x[1],
)[:5]:
    print(f"  {name:20s} {imp:.3f}")
```

</div>

<div class="level-next">
<span>Want the bias-variance derivation and honest diagnostics?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Variance reduction</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathrm{Var}(\hat{y}) \;=\; \rho\,\sigma^2 \;+\; \frac{1-\rho}{B}\,\sigma^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`σ²`variance of an individual tree's prediction

</li>
<li markdown="1">

`ρ`pairwise correlation between trees on the same input

</li>
<li markdown="1">

`B`number of trees

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{forest variance} \;=\; \text{correlation} \times \text{tree variance} \;+\; \frac{1 - \text{correlation}}{\text{number of trees}} \times \text{tree variance} $$</span>

**In words.** The variance of the averaged forecast splits into two pieces. The first is `correlation` (called `ρ`, rho — a number between 0 and 1 measuring how similarly trees vote on the same input) times the variance of a single tree. The second piece shrinks with the number of trees in the denominator — so adding more trees only reduces the second piece, while the first piece is a hard floor you can't escape no matter how many trees you grow. The whole point of feature subsetting is to push correlation `ρ` closer to zero so that floor is lower.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`forest variance`spread of the ensemble's prediction at a fixed input across re-fits

</li>
<li markdown="1">

`tree variance`spread of a single tree's prediction at the same input

</li>
<li markdown="1">

`correlation`how similarly any two trees vote on the same input — lower is better

</li>
<li markdown="1">

`number of trees`more trees only shrink the second term; the first sets the floor

</li>
</ul>

</div>

As *B* → ∞ the second term vanishes — variance is floored at `ρσ²`. The whole game is to **drive ρ down** via per-split feature subsets without making each tree too weak (which inflates σ²).

</div>

<article class="tldr-body" markdown="1">

Random forests are bagging applied to trees, with one twist: at each split, only a random subset of *m* features (out of *p*) is considered. This injects an extra source of decorrelation beyond bootstrap sampling alone, addressing the fact that standard bagging produces highly correlated trees whenever a few features dominate the splits.

The `max_features` hyperparameter trades off correlation (low *m* → low ρ) against individual tree strength (low *m* → higher σ²). Empirical defaults — √*p* for classification, *p*/3 for regression — work remarkably well across domains.

**Out-of-bag error.** Each bootstrap omits a fraction (1 − 1/N)<sup>N</sup> → e<sup>−1</sup> ≈ 36.8% of points. Averaging predictions from trees that excluded each point gives an estimate of generalization error that is asymptotically equivalent to leave-one-out CV, at no extra training cost.

**Feature importance.** Impurity-based importance (`feature_importances_`) is biased toward high-cardinality features. Prefer `permutation_importance` on a held-out set for honest estimates.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You want a robust baseline before investing in boosting
- You need uncertainty estimates (via per-tree predictions)
- OOB is desirable (small datasets, expensive CV)
- The signal is heterogeneous — feature interactions vary across regimes

</div>

<div class="no" markdown="1">

### Skip it when

- You need calibrated probabilities without post-hoc calibration
- Memory is tight (forests are heavyweight at inference)
- Smooth function approximation matters (forests are piecewise-constant)
- The variable of interest is monotone in inputs and you need that respected

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

clf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=1,
    oob_score=True,
    bootstrap=True,
    n_jobs=-1,
    random_state=0,
).fit(X_train, y_train)

# Honest importance via permutation on held-out set
perm = permutation_importance(
    clf, X_test, y_test, n_repeats=10, n_jobs=-1, random_state=0
)
ranked = sorted(zip(X_train.columns, perm.importances_mean), key=lambda x: -x[1])

print(f"OOB: {clf.oob_score_:.3f}   Test: {clf.score(X_test, y_test):.3f}")
for name, imp in ranked[:10]:
    print(f"  {name:20s} {imp:+.4f}")
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

[scikit-learn user guide <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/ensemble.html#random-forests){: target="_blank" }
<span class="annotation">Practical reference for hyperparameters — what each one does and how to set it.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Breiman (2001) <i class="fas fa-external-link-alt"></i>](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf){: target="_blank" }
<span class="annotation">The original paper. 30 pages, surprisingly readable. The clearest explanation of <em>why</em> RFs work.</span>

</li>
<li data-tier="indepth" markdown="1">

[ESL, chapter 15 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">Full theoretical treatment with the bias–variance derivation. Read after Breiman.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Wikipedia <i class="fas fa-external-link-alt"></i>](https://en.wikipedia.org/wiki/Random_forest){: target="_blank" }
<span class="annotation">Solid quick reference with pointers to extensions (Extra Trees, isolation forests).</span>

</li>
</ul>

</div>

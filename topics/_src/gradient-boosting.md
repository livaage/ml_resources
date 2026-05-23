---
title: Gradient Boosting — ML Resources Hub
eyebrow_text: ← Theory · Tree-Based
eyebrow_href: ../theory.html
heading: Gradient Boosting
lead: Trees that learn sequentially — each one corrects the mistakes of the ones before.
prev_href: random-forests.html
prev_title: Random Forests
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**A relay race of trees.** Train a small tree, see where it makes mistakes, then train the next tree to fix those mistakes. Repeat. Sum them up. The resulting model is usually the strongest thing you can put on tabular data.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Each Step fits one depth-2 tree to the current residuals — watch the orange fit chase the dashed target</span>
</div>
<div class="viz-classic-controls">
<button id="viz-boost-step" type="button">Step</button>
<button id="viz-boost-auto" type="button">Auto</button>
<button id="viz-boost-reset" type="button">Reset</button>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Target
                <select id="viz-boost-data"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                η
                <input id="viz-boost-lr" class="viz-classic-slider" type="range"></input>
</label>
<span class="viz-classic-badge" id="viz-boost-lr-lbl">η = 0.40</span>
<span class="viz-classic-badge" id="viz-boost-counter">Tree 0</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-boost-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-boost-caption"></div>
</div>

<script src="{{root}}js/viz/boosting.js"></script>

Round 0 is just the mean — the bottom panel shows every point's residual from it. Each Step fits a depth-2 tree (4 leaves) to those residuals and adds it (scaled by η) to the ensemble. After ~20 trees on the Sine target the fit is essentially perfect; the residuals shrink toward zero. The *Step function* target is a great demo of why trees are excellent on non-smooth targets — boosting nails the jumps where smooth methods would oscillate.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Random forests train trees in parallel and average them. Gradient boosting trains them *in sequence*: tree 2 is trained on the errors tree 1 made, tree 3 is trained on the errors trees 1+2 made, and so on. Each new tree is small (just a few splits) but specifically targeted at what's still going wrong.

The libraries you've heard of — **XGBoost**, **LightGBM**, **CatBoost** — are heavily-optimized implementations of this idea. They're still the default winning approach on Kaggle tabular competitions.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Tabular data and you want the best possible accuracy
- You have time to tune hyperparameters (learning rate, depth, etc.)
- Mixed numeric / categorical features
- You're competing on Kaggle

</div>

<div class="no" markdown="1">

### Skip it when

- You need a quick zero-tuning baseline (random forests are friendlier)
- Data is images, text, or audio (use neural networks)
- Tiny data — overfits easily without careful tuning
- You need a fully interpretable model

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=200, random_state=0)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

</div>

<div class="level-next">
<span>Want to see how the gradients work?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Key idea</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ F_m(x) \;=\; F_{m-1}(x) \;+\; \nu \cdot h_m(x), \qquad h_m \approx -\nabla_F \mathcal{L} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`Fm`the ensemble after *m* trees

</li>
<li markdown="1">

`hm`the new tree at step *m*, fit to the negative gradient of the loss

</li>
<li markdown="1">

`ν`learning rate (shrinkage), typically 0.01–0.1

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{new ensemble} \;=\; \text{old ensemble} \;+\; \nu \times \text{new tree fit to residuals} $$</span>

**In words.** Start with whatever ensemble you have so far (initially: just the mean). Compute its residuals — for squared loss, that's literally *actual − predicted*; for other losses it's the negative gradient of the loss, sometimes called the "pseudo-residual". Fit one small tree to those residuals so the tree predicts what's still missing, then add that tree to the ensemble after multiplying by a small learning rate `ν` (nu, the Greek letter for "shrinkage", typically 0.01–0.1). Smaller `ν` means slower learning but better generalisation. Repeat until validation error stops improving.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`old ensemble`sum of all the trees you've added so far

</li>
<li markdown="1">

`new tree`a small tree (a few splits) trained on the current residuals

</li>
<li markdown="1">

`residuals`what the current ensemble is still getting wrong — the negative gradient of the loss

</li>
<li markdown="1">

`ν`learning rate (shrinkage) — small ν trades training speed for generalisation

</li>
</ul>

</div>

At each step, fit a small tree to the pseudo-residuals (negative gradient of the loss w.r.t. current predictions). Add it to the ensemble with a small learning rate. Repeat.

</div>

<article class="tldr-body" markdown="1">

The "gradient" in gradient boosting is the derivative of the loss with respect to the predictions. For squared loss, the gradient is just *y − F(x)* — the residuals — so each new tree is trained to predict the leftover error. For other losses (log loss, hinge), the pseudo-residuals differ but the recipe is the same.

Three hyperparameters do most of the work: `n_estimators` (more trees → more capacity), `max_depth` (deeper trees → more interaction modelling), and `learning_rate` (smaller → slower but better generalization). Early stopping on a validation set is the standard way to choose `n_estimators`.

Unlike random forests, gradient boosting is *not* bagged — every tree sees all the data. The decorrelation comes from each tree fitting a different residual signal, not from data sampling.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You want SOTA accuracy on tabular data
- You can afford hyperparameter tuning
- You need a flexible loss (regression, classification, ranking)
- Feature importance / SHAP values are useful

</div>

<div class="no" markdown="1">

### Skip it when

- Data is non-tabular
- You're under-resourced — sklearn's default GB is slow; reach for LightGBM
- You need probability calibration without post-hoc adjustment
- Strict latency budget at inference (forests can be heavy)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import lightgbm as lgb

train_ds = lgb.Dataset(X_train, label=y_train)
val_ds   = lgb.Dataset(X_val,   label=y_val, reference=train_ds)

params = {
    "objective":     "binary",
    "metric":        "auc",
    "learning_rate": 0.05,
    "num_leaves":    63,
    "feature_fraction": 0.9,
    "verbosity":    -1,
}

model = lgb.train(
    params, train_ds, num_boost_round=2000,
    valid_sets=[val_ds],
    callbacks=[lgb.early_stopping(50)],
)
```

</div>

<div class="level-next">
<span>Want the functional-gradient view and the XGBoost regularizer?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Regularized objective (XGBoost)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L}^{(m)} \;=\; \sum_{i} \ell\!\left(y_i, F_{m-1}(x_i) + h_m(x_i)\right) \;+\; \Omega(h_m), \quad \Omega(h) = \gamma\,T + \tfrac{1}{2}\lambda \|w\|^2 $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`T`number of leaves in the tree

</li>
<li markdown="1">

`w`vector of leaf weights

</li>
<li markdown="1">

`γ, λ`regularization strengths on leaf count and leaf magnitude

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{step } m \text{ objective} \;=\; \text{loss after adding new tree} \;+\; \text{tree complexity penalty}, \quad \text{penalty} = \gamma \times (\text{leaves}) + \tfrac{1}{2}\lambda \times \text{(sum of squared leaf weights)} $$</span>

**In words.** When choosing the next tree, don't just minimise training loss — also charge the tree for being complex. The first term sums the per-point loss after adding the new tree's prediction to the current ensemble. The second term, the `complexity penalty`, has two pieces: `γ` (gamma) charges per leaf — making trees prefer fewer leaves; and `λ` (lambda) penalises the squared magnitude of each leaf's output value — keeping leaf predictions modest. Together they let you grow only the splits whose loss reduction outweighs their complexity cost.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`loss after adding new tree`training error once the new tree's predictions are added to the running ensemble

</li>
<li markdown="1">

`tree complexity penalty`extra cost for big or splashy trees — discourages overfitting

</li>
<li markdown="1">

`γ`price per leaf — bigger γ means fewer splits

</li>
<li markdown="1">

`λ`penalty on leaf magnitudes — bigger λ shrinks each leaf's predicted value

</li>
<li markdown="1">

`leaf weights`the value each leaf predicts; the penalty sums their squares

</li>
</ul>

</div>

Gradient boosting can be viewed as **functional gradient descent in function space**: at each step, move *F* in the direction that most decreases the loss, where the "direction" is parameterized as a regression tree. XGBoost adds explicit regularization on tree complexity.

</div>

<article class="tldr-body" markdown="1">

Friedman's original formulation (2001) interprets boosting as gradient descent in function space, with each weak learner an approximate steepest-descent step. The Newton-style variant used by XGBoost expands the loss to second order, fitting trees to *g/h* instead of just *g* — this is what the "XGBoost gain" formula computes at every candidate split.

**Why LightGBM is faster.** Histogram binning (bucket features into ~256 bins before considering splits) reduces split-finding from O(N) to O(bins). Leaf-wise (not level-wise) growth picks the highest-loss-reducing split anywhere in the tree, which converges faster but overfits if depth is uncapped.

**CatBoost.** Categorical features get ordered target statistics computed on permuted prefixes to avoid target leakage. Often dominates when you have lots of categorical features and not much time to tune.

**Common failure mode.** Default learning rate (0.1) is often too high. Halve it and double n_estimators — usually a free generalization gain.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Heterogeneous tabular features (numeric + categorical + missing)
- You need monotone constraints (LightGBM / XGBoost support them)
- You're confident enough to tune and have a held-out set
- Quantile / ranking objectives matter (NDCG, Huber, etc.)

</div>

<div class="no" markdown="1">

### Skip it when

- You're chasing the last 0.5% — try CatBoost-vs-LightGBM ensemble first
- Data has strong sequential / spatial structure (use specialized nets)
- You can't tune — random forests are more forgiving defaults
- Tiny labelled set — boosting overfits faster than RF

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    "objective":      "binary:logistic",
    "eval_metric":    "auc",
    "eta":            0.05,
    "max_depth":      6,
    "min_child_weight": 1,
    "gamma":          0.1,    # γ in Ω(h)
    "reg_lambda":     1.0,    # λ in Ω(h)
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "tree_method":    "hist",
}

booster = xgb.train(
    params, dtrain, num_boost_round=2000,
    evals=[(dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)
print(f"Best AUC: {booster.best_score:.4f} at round {booster.best_iteration}")
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

[XGBoost docs — introduction <i class="fas fa-external-link-alt"></i>](https://xgboost.readthedocs.io/en/stable/tutorials/model.html){: target="_blank" }
<span class="annotation">Walks through the second-order Taylor expansion and the split-gain formula. Best concise derivation.</span>

</li>
<li data-tier="indepth" markdown="1">

[Friedman (2001) <i class="fas fa-external-link-alt"></i>](https://jerryfriedman.su.domains/ftp/trebst.pdf){: target="_blank" }
<span class="annotation">The original "Greedy Function Approximation" paper. Where the functional-gradient view comes from.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[LightGBM features <i class="fas fa-external-link-alt"></i>](https://lightgbm.readthedocs.io/en/latest/Features.html){: target="_blank" }
<span class="annotation">Histogram binning, leaf-wise growth, GOSS sampling — the tricks that make LightGBM fast.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[ESL, chapter 10 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">Full chapter on boosting from a statistical perspective.</span>

</li>
</ul>

</div>

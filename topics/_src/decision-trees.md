---
title: Decision Trees — ML Resources Hub
eyebrow_text: ← Theory · Tree-Based
eyebrow_href: {{root}}theory.html
heading: Decision Trees
lead: A flowchart of yes/no questions, learned automatically from data.
next_href: random-forests.html
next_title: Random Forests
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**A series of yes/no questions that funnel data to a prediction.** "Is income > £30k?" → "Yes" → "Is age > 40?" → "Yes" → predict "high credit". The algorithm picks the questions and the order, learning them from training data.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide the depth — watch the tree carve axis-aligned regions and the right-hand diagram grow</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Dataset
                <select id="viz-tree-data"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Depth
                <input id="viz-tree-depth" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-tree-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-tree-depth-lbl">max depth = 3</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-tree-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-tree-caption"></div>
</div>

<script src="{{root}}js/viz/decision-trees.js"></script>

Every split is axis-aligned — so a single tree can only carve feature space into rectangles. That's fine for the *Linear boundary* dataset (one split does it) but watch what happens on *XOR*: depth 1 can't separate the classes at all, but depth 2 nails it with two perpendicular splits. Crank to depth 8 on a noisy dataset and you'll see overfitting — every training point gets its own tiny region.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Decision trees are perhaps the most intuitive ML model — you can read one off and explain exactly why it made each prediction. They handle mixed feature types (numeric + categorical), don't need normalisation, and don't make assumptions about the shape of the data.

The big downside: a single tree is brittle. Small changes in the training data produce very different trees, so they overfit easily. The fix is ensembles ([random forests]({{root}}topics/random-forests.html), [gradient boosting]({{root}}topics/gradient-boosting.html)) — many trees combined. But you have to understand a single tree first.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You need a model you can read off and explain
- Quick baseline with no feature engineering
- Mixed numeric / categorical features
- You'll feed it into a random forest or boosted ensemble

</div>

<div class="no" markdown="1">

### Skip it when

- You want best possible accuracy — use random forests / GBM instead
- Smooth function approximation matters (trees are piecewise-constant)
- Linear relationships dominate — a linear model is more elegant
- Very high-dim or sparse data

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf.fit(X_train, y_train)

# Visualise the actual decisions
plot_tree(clf, feature_names=X_train.columns, filled=True)
plt.show()
```

</div>

<div class="level-next">
<span>Want the splitting math?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Splitting criterion (Gini impurity)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ G(S) \;=\; 1 \;-\; \sum_{k=1}^{K} p_k^2, \qquad \Delta G \;=\; G(\text{parent}) - \sum_{\text{children}} \tfrac{|S_i|}{|S|}\,G(S_i) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`S`set of training points at a node

</li>
<li markdown="1">

`pk`fraction of class *k* in *S*

</li>
<li markdown="1">

`ΔG`impurity decrease from a candidate split — pick the split that maximises this

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{impurity} \;=\; 1 \;-\; \text{sum of (class fraction)}^2, \qquad \text{gain} \;=\; \text{parent impurity} \;-\; \text{weighted child impurity} $$</span>

**In words.** At a node, look at the points there and compute each class's fraction; `impurity` is one minus the sum of those fractions squared (it's zero when every point is one class, and large when classes are evenly mixed). For any candidate split, compute the impurity of each child, weight each by the share of points it gets, and subtract that weighted sum from the parent's impurity. That difference is the `gain` — pick whichever split maximises it. The algorithm tries every feature and every threshold and keeps the winner.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`impurity`how mixed the classes are at a node — zero for pure, larger for mixed

</li>
<li markdown="1">

`class fraction`share of points belonging to each class at the node

</li>
<li markdown="1">

`gain`how much the split reduces impurity — bigger is better

</li>
<li markdown="1">

`weighted child impurity`each child's impurity multiplied by its share of points, then summed

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**CART (Classification And Regression Trees)** is the canonical algorithm. Greedy: at each node, search over all features and all thresholds for the split that most decreases impurity. Recurse on each child. Stop when a depth limit is reached or impurity decrease is too small.

**Splitting criteria.** For classification: *Gini* (default in sklearn) or *entropy* (information gain). They're nearly equivalent in practice. For regression: *variance reduction* — pick the split that most reduces within-node variance of *y*.

**Controlling overfitting.** Unpruned trees memorise the training set. Three knobs: `max_depth` (hard limit), `min_samples_leaf` (refuse splits that produce tiny leaves), `min_impurity_decrease` (refuse splits below a threshold). Cost-complexity pruning (CCP) is the principled approach: grow a deep tree, then prune back nodes whose contribution to accuracy is too small relative to their complexity.

**Categorical features.** sklearn's trees don't natively handle categoricals — they expect numeric inputs (use one-hot or target encoding). LightGBM and XGBoost handle them natively, often via partition-based splits which are more efficient than one-hot.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Need an interpretable baseline you can show non-technical stakeholders
- Quick exploration on a new dataset (no scaling needed)
- Stage 1 of a tree-based pipeline (RF / boosting build on this)
- Decisions that should be a literal flowchart

</div>

<div class="no" markdown="1">

### Skip it when

- Continuous target where smooth predictions matter
- You'll use an ensemble anyway — go straight to that
- High-dim sparse text — linear models or NB win
- Strong distributional assumptions you'd rather encode explicitly

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Reasonable hyperparameter sweep
params = {
    "max_depth":        [None, 4, 6, 8, 10],
    "min_samples_leaf": [1, 5, 10, 20],
    "criterion":        ["gini", "entropy"],
}
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=0),
    params, cv=5, scoring="f1_macro", n_jobs=-1,
).fit(X_train, y_train)

print(f"Best params: {gs.best_params_}")
print(f"Best CV F1:  {gs.best_score_:.3f}")
```

</div>

<div class="level-next">
<span>Want cost-complexity pruning and the bias side of the story?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Cost-complexity pruning</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ R_\alpha(T) \;=\; R(T) \;+\; \alpha\,|T| $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`R(T)`misclassification cost of tree *T*

</li>
<li markdown="1">

`|T|`number of leaves

</li>
<li markdown="1">

`α`regularization strength — sweep this to trace out a sequence of pruned subtrees

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{penalised cost} \;=\; \text{misclassification cost} \;+\; \alpha \times \text{number of leaves} $$</span>

**In words.** Instead of judging a tree only by how often it gets training points wrong, also charge it for how many leaves it has. `α` (alpha, a small positive number) is the price-per-leaf — at `α = 0` you keep the whole tree, and as α grows you progressively prune branches whose accuracy gain doesn't justify their leaf cost. Sweeping α from 0 upwards traces out a nested sequence of pruned subtrees; pick whichever one has the best validation score (often the smallest within one standard error of the best, as in the code below).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`penalised cost`the quantity being minimised — fit plus complexity

</li>
<li markdown="1">

`misclassification cost`fraction of training points the tree gets wrong

</li>
<li markdown="1">

`number of leaves`count of terminal nodes — a proxy for tree size

</li>
<li markdown="1">

`α`regularization strength — bigger α means harsher penalty per leaf

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Greedy splits are myopic.** CART picks the best *local* split at each node, which isn't always the globally optimal tree. Finding the optimal tree is NP-hard. Modern interpretable-ML methods (OCT, MurTree) use mathematical programming to find globally optimal trees for small problems — useful for high-stakes applications where you want to defend the structure.

**Bias and feature importance.** Impurity-based importance (sklearn's `feature_importances_`) is biased toward high-cardinality features — they offer more split candidates. *Permutation importance* on a held-out set gives more honest estimates. For random forests this matters even more — see the discussion on the [RF page]({{root}}topics/random-forests.html).

**Missing values.** CART originally used *surrogate splits* — alternate splits learned to handle missing values gracefully. sklearn doesn't implement this; LightGBM/XGBoost route missing values to whichever child improves the loss most. The simplest workaround in sklearn: impute first, or use `HistGradientBoostingClassifier` which handles missing natively.

**Monotonicity constraints.** Sometimes you need the model to be monotone in a feature (e.g. credit risk should not decrease with income). XGBoost and LightGBM both support monotone constraints; vanilla CART doesn't. When this matters, switch frameworks.

**Where trees go to die.** Decision trees alone are rarely state-of-the-art today. They're foundations for ensembles (RF, GBM). The interesting research is now in: differentiable trees (soft decision trees, tree-MLPs), neural-tree hybrids (NODE, TabNet), and globally-optimal trees for interpretability-critical settings.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Regulatory / interpretability requirements that need a literal flowchart
- Cost-complexity pruning gives you a principled accuracy/complexity trade-off
- You're building or debugging an ensemble and need to inspect base learners
- Globally-optimal trees on small problems — for defensibility

</div>

<div class="no" markdown="1">

### Skip it when

- You're chasing accuracy — ensembles dominate
- Smooth function approximation needed
- Adversarial robustness — trees have known attack surfaces
- Online / streaming setting — incremental tree algorithms exist but are niche

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Cost-complexity pruning path
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
path = tree.cost_complexity_pruning_path(X_train, y_train)
alphas, impurities = path.ccp_alphas, path.impurities

# Train one tree per alpha; pick the smallest tree within 1 SE of the best
trees = [
    DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X_train, y_train)
    for a in alphas
]
val_scores = np.array([t.score(X_val, y_val) for t in trees])

best_i  = val_scores.argmax()
se      = val_scores.std() / np.sqrt(len(val_scores))
within  = np.where(val_scores >= val_scores[best_i] - se)[0]
chosen  = trees[within[-1]]   # simplest tree within 1 SE — Occam's razor

print(f"Chose tree with {chosen.get_n_leaves()} leaves, "
      f"val acc = {val_scores[within[-1]]:.3f}")
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

[scikit-learn — Decision trees <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/tree.html){: target="_blank" }
<span class="annotation">Practical reference with examples for classification, regression, and multi-output. The "Tips on practical use" section is gold.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[ESL, chapter 9 <i class="fas fa-external-link-alt"></i>](https://hastie.su.domains/ElemStatLearn/){: target="_blank" }
<span class="annotation">Statistical-learning treatment of CART, splitting criteria, and pruning. Read sections 9.2 for the foundations.</span>

</li>
<li data-tier="indepth" markdown="1">

[Breiman et al. — CART (book) <i class="fas fa-external-link-alt"></i>](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-richard-olshen-charles-stone){: target="_blank" }
<span class="annotation">The 1984 monograph that introduced CART. Out of print but PDFs circulate. The original source on cost-complexity pruning and surrogate splits.</span>

</li>
<li data-tier="intuition" markdown="1">

[MLU-Explain — Decision Trees <i class="fas fa-external-link-alt"></i>](https://mlu-explain.github.io/decision-tree/){: target="_blank" }
<span class="annotation">Interactive visualization of how splits work. Best intuition builder for "what does the algorithm <em>do</em>?"</span>

</li>
</ul>

</div>

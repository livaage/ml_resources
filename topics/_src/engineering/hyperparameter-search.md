---
title: Hyperparameter Search — ML Resources Hub
eyebrow_text: ← Engineering · Development Loop
eyebrow_href: {{root}}engineering.html
heading: Hyperparameter Search
lead: Grid, random, Bayesian, Hyperband — how to find good hyperparameters without burning your compute budget.
active_nav: engineering
prev_href: experiment-tracking.html
prev_title: Experiment Tracking
next_href: testing-ml-code.html
next_title: Testing ML Code
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Hyperparameter search is itself an optimisation problem with a budget.** Random search beats grid search in > 3 dimensions. Bayesian optimisation (Optuna, scikit-optimize) is much better when each trial is expensive. Hyperband / ASHA aggressively prune unpromising trials. The trick is matching the search strategy to the cost of a trial.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Grid search vs random search — same number of trials, very different coverage of the important dimensions</span>
</div>
<div class="viz-classic-controls">
<button id="viz-hp-grid" type="button" class="active">Grid 5×5</button>
<button id="viz-hp-random" type="button">Random 25</button>
<button id="viz-hp-bayes" type="button">Bayesian (10 random + 15 informed)</button>
<button id="viz-hp-reset" type="button">Re-sample</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-hp-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-hp-caption"></div>
</div>

<script src="{{root}}js/viz/hp-search.js"></script>

An objective *f(lr, wd)* with an "important" dimension (left-right) and an "unimportant" one (up-down). Grid wastes most of its budget on the unimportant axis; random hits more unique values of the important one; Bayesian quickly finds the high-value region after a few exploratory probes.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Grid search.** The default beginner choice. Wasteful — most hyperparameters aren't equally important, but grid spends the same effort on each axis. Useful only with ≤ 3 hyperparameters.

**Random search.** Sample N points uniformly. Bergstra & Bengio (2012) showed this is strictly better than grid in high dimensions because the search distributes budget across the relevant axes. Surprisingly hard to beat in practice when N is large enough.

**Bayesian optimisation.** Build a surrogate model (Gaussian process or random forest) of the objective; pick the next trial to maximise expected improvement. Wins when each trial is expensive. Tools: Optuna, scikit-optimize, BoTorch.

**Hyperband / ASHA.** Allocate many trials, but kill the bad ones early. Train at low compute; promote the top-k to more compute; iterate. Often the most cost-efficient strategy for deep learning.

**Population-based training (PBT).** Run many models in parallel; periodically copy the best ones' weights and perturb their hyperparameters. Adapts hyperparameters during training.

</article>

<div class="use-cases neutral" markdown="1">

<div class="yes" markdown="1">

### Pick by trial cost

- **Cheap trial (< minutes)**: random search
- **Moderate trial (~hour)**: Bayesian via Optuna
- **Expensive trial (~day)**: Hyperband / ASHA
- **Continuous training**: PBT
- **≤ 3 hyperparameters with discrete values**: grid is fine

</div>

<div class="no" markdown="1">

### Pitfalls

- Searching too many parameters — most don't matter; profile importances first
- Search range too narrow — you can't find what you don't include
- Comparing trials with different seeds — variance dominates
- Picking by val score from a single seed

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import optuna

def objective(trial):
    lr  = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd  = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
    bs  = trial.suggest_categorical("batch_size", [32, 64, 128])
    act = trial.suggest_categorical("activation", ["relu", "gelu", "swish"])

    val_loss = train_and_validate(lr, wd, bs, act)

    # Report intermediate values for pruning
    trial.report(val_loss, step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return val_loss

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
)
study.optimize(objective, n_trials=50, timeout=3600 * 6)
print(study.best_params, study.best_value)
```

</div>

<div class="level-next">
<span>Want the math behind Bayesian optimisation, & ASHA scheduling?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Expected Improvement</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathrm{EI}(x) = \mathbb{E}\!\left[\max(0,\, f^* - f(x))\right] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`f*`best value seen so far

</li>
<li markdown="1">

Expectation over the surrogate's posterior — high for promising unexplored regions

</li>
<li markdown="1">

Acquisition function that drives Bayesian optimisation

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{expected improvement at } x \;=\; \text{average of }\max(0,\; \text{best so far} - f(x)) $$</span>

**In words.** An *acquisition function* that tells Bayesian optimisation which hyperparameter point to try next. The `𝔼[·]` in the math is an *expected value* — averaged over the surrogate model's uncertainty about `f(x)` (e.g., a Gaussian process posterior). For each candidate `x`, you ask: "if I were to evaluate it, how much could it beat my best score so far?" The `max(0, …)` means you only count *improvements*, never penalising regions where the model thinks `x` is worse. Points with high EI are either predicted to be better, or have lots of uncertainty (so there's a chance they're great).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`x`a candidate hyperparameter configuration

</li>
<li markdown="1">

`f(x)`predicted score (uncertain — from the surrogate model)

</li>
<li markdown="1">

`best so far`the best score seen across all trials evaluated

</li>
<li markdown="1">

`max(0, …)`only count improvements; ignore "would be worse"

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Surrogate models.** Bayesian optimisation builds a probabilistic model of *f(hyperparams)*. Gaussian processes are the textbook choice — good with few trials, slow when many. Tree-structured Parzen Estimator (TPE, used by Optuna by default) scales better and handles categoricals.

**Acquisition functions.** Expected Improvement, Upper Confidence Bound, Probability of Improvement, Knowledge Gradient. EI is the standard; UCB is parameterised by an exploration knob. Most libraries default to EI; pick UCB when you want explicit exploration control.

**Asynchronous Successive Halving (ASHA).** Li et al. (2020). Promote the top η<sup>-1</sup> fraction of trials at each rung; trials that survive rung r get η× the compute. Asynchronous (no synchronisation barrier) makes it scale to hundreds of workers.

**Multi-fidelity methods.** Use cheap proxies (subsampled data, fewer epochs) to estimate the expensive metric. Hyperband + BOHB combine ASHA's early pruning with Bayesian optimisation of which trials to try. The strongest general-purpose approach for deep learning.

**Pruning.** Median pruner: kill a trial if its intermediate value is below the median at the same step. Patient pruner: like median but with a grace period. Hyperband pruner: structured promotion schedule.

**Search space design.** Use log-uniform for learning rate and weight decay. Use uniform for layer counts and dropout. Use categorical for activation choice and optimizer name. Conditioning ("if optimizer == sgd, also tune momentum") is supported by Optuna's `trial.suggest_*` inside if-blocks.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import optuna

# Multi-fidelity with HyperbandPruner — train cheap, promote winners
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(multivariate=True),
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
)

def objective(trial):
    cfg = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "model": trial.suggest_categorical("model", ["resnet18", "resnet50", "vit"]),
    }
    for epoch in range(100):
        val_loss = train_one_epoch(cfg)
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
    return val_loss

study.optimize(objective, n_trials=200, n_jobs=4)
```

</div>

<div class="level-next">
<span>Want PBT, neural architecture search, & multi-objective?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Population-Based Training</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \theta_i^{(t+1)} \leftarrow \begin{cases} \theta_j^{(t)} \;\text{(copy)} & \text{if rank}(i) < \text{bottom k} \\ \theta_i^{(t)} \;\text{(continue)} & \text{otherwise} \end{cases} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Bottom performers copy top performers' weights

</li>
<li markdown="1">

… then perturb the hyperparameters

</li>
<li markdown="1">

Adapts hyperparameters during training — useful when the optimal LR schedule isn't known a priori

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{agent $i$ next step} \;=\; \begin{cases} \text{copy weights from a top agent} & \text{if agent $i$ is in the bottom } k \\ \text{keep current weights} & \text{otherwise} \end{cases} $$</span>

**In words.** Run many training runs ("agents") in parallel, each with different hyperparameters. `θ` (theta) denotes the model weights; the subscript `i` is the agent index and superscript `t` is the training step. Periodically, rank agents by validation score: the bottom `k` copy the weights of a top performer, then *perturb* their hyperparameters (e.g., multiply LR by 1.2 or 0.8). High performers continue unchanged. Over time, the surviving hyperparameter *trajectory* becomes an evolved schedule — possibly better than any fixed schedule a human would pick.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`θ`model weights (parameters)

</li>
<li markdown="1">

`i`agent index (which run in the population)

</li>
<li markdown="1">

`t`training step / generation

</li>
<li markdown="1">

`bottom k`the worst-performing fraction at this checkpoint

</li>
<li markdown="1">

`copy`literally copy the weights from a randomly chosen top agent

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**PBT.** Jaderberg et al. (2017). Run *N* agents in parallel with random hyperparameters; periodically exploit (bad agents copy good agents' weights) and explore (perturb hyperparameters). The hyperparameter trajectory becomes a "schedule" rather than a fixed value. Used at DeepMind for many production-grade hyperparameter sweeps.

**Neural Architecture Search (NAS).** Hyperparameter search for architectures. Once-popular; now mostly subsumed by foundation-model fine-tuning. Differentiable NAS (DARTS) and gradient-based methods are the modern face.

**Multi-objective optimisation.** Trade off accuracy vs latency, accuracy vs cost, etc. Pareto fronts; NSGA-II for evolutionary, qNEHVI for Bayesian. Optuna has built-in support.

**Cost-aware search.** Each trial has a known cost; the budget is constrained. Cost-EI, Cost-LCB. Useful when trials vary enormously in compute (batch size, model size).

**Continual / online hyperparameter tuning.** Production models drift; what was the best LR a year ago may not be now. Periodic re-tuning, possibly with PBT-style methods running on production.

**Beware overfitting the search.** If your search trains on training data and selects on val, you're fine. If you tune the validation set itself (e.g., picking which features to engineer based on val performance), you're overfitting your search procedure. Hold out a final test set.

**Reporting.** Always report your search budget (number of trials, total GPU-hours). Without this, a "we found 95% accuracy" is meaningless — was that 5 trials or 50 000?

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import optuna

# Multi-objective — maximise accuracy AND minimise inference latency
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(),
)
def objective(trial):
    width = trial.suggest_int("width", 32, 512, log=True)
    depth = trial.suggest_int("depth", 2, 8)
    acc, latency_ms = train_and_benchmark(width, depth)
    return acc, latency_ms

study.optimize(objective, n_trials=100)

# Inspect the Pareto front
for trial in study.best_trials:
    print(trial.values, trial.params)
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

[Optuna — Documentation <i class="fas fa-external-link-alt"></i>](https://optuna.org/){: target="_blank" }
<span class="annotation">The reference Python hyperparameter optimisation library. TPE, Hyperband pruner, multi-objective, sklearn / Lightning / huggingface integrations.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Bergstra & Bengio (2012) — Random Search <i class="fas fa-external-link-alt"></i>](https://www.jmlr.org/papers/v13/bergstra12a.html){: target="_blank" }
<span class="annotation">The empirical paper showing random search beats grid search in high dimensions. Short, readable, foundational.</span>

</li>
<li data-tier="indepth" markdown="1">

[Li et al. (2020) — ASHA <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1810.05934){: target="_blank" }
<span class="annotation">Asynchronous Successive Halving — the basis of modern multi-fidelity tuning.</span>

</li>
<li data-tier="indepth" markdown="1">

[Jaderberg et al. (2017) — PBT <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1711.09846){: target="_blank" }
<span class="annotation">DeepMind's Population-Based Training paper. Hyperparameters as a learned schedule, not a fixed value.</span>

</li>
</ul>

</div>

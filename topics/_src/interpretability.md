---
title: Interpretability &amp; Explainability — ML Resources Hub
eyebrow_text: ← Theory · Frontier
eyebrow_href: ../theory.html
heading: Interpretability &amp; Explainability
lead: Open the black box — feature importance, attribution, mechanistic interpretability, and the limits of "why did the model do that?"
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Two flavours. Per-prediction explanation: "why did this input get this answer?". Model-wide understanding: "what does this model actually compute?".** The first is feature-attribution (SHAP, LIME, Integrated Gradients). The second is mechanistic interpretability (probing, circuits, sparse autoencoders). Both have real successes and real limits.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">SHAP-style local explanation — which features pushed this prediction up, which pushed it down?</span>
</div>
<div class="viz-classic-controls">
<button id="viz-int-resample" type="button">Re-sample input</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-int-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-int-caption"></div>
</div>

<script src="{{root}}js/viz/interpretability.js"></script>

A single tabular prediction with per-feature SHAP-style contributions. Each bar shows how much that feature pushed the prediction up (orange) or down (indigo) relative to the average. Sum them all up + the model's average prediction = this specific prediction. SHAP values guarantee this additivity by construction.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Feature importance (model-wide).** Permutation importance: shuffle a feature; how much does the score drop? Tree importance: how often a feature appears in splits, weighted by gain. Both are simple and useful but can be misled by correlated features.

**SHAP & LIME (per-prediction).** SHAP (Lundberg & Lee, 2017) — game-theoretic Shapley values, additivity by construction, slow but principled. LIME — fit a local linear surrogate around a single prediction, fast but unstable. Both give "this feature contributed +X to this prediction".

**Saliency maps for images.** Where in the image did the model "look"? Gradient × input, Integrated Gradients, GradCAM, SmoothGrad. All are easy to compute, but easy to mislead — a saliency map can highlight the right region for the wrong reason.

**Attention as explanation?** Tempting but controversial. Attention weights don't directly correspond to feature importance, even in transformers. Use as a starting hypothesis, not an explanation. Jain & Wallace (2019): "Attention is not Explanation".

**Mechanistic interpretability.** Find specific circuits inside a neural network that implement a specific behaviour. Anthropic's induction heads, OpenAI's sparse autoencoders. Still early — the field is producing real results but on toy models.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Regulated domain (finance, healthcare, insurance) — explanations are mandatory
- Debugging — why is this prediction wrong?
- Audit — does the model rely on a feature it shouldn't?
- Trust — communicating model behaviour to non-ML stakeholders

</div>

<div class="no" markdown="1">

### Real limits

- Explanations can mislead — humans accept plausible-but-wrong narratives
- Correlated features confound most attribution methods
- "Why" questions presume a causal model that the attribution doesn't have
- Mechanistic understanding is hard work and rarely complete

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import shap
import xgboost as xgb

# Train and explain with SHAP
model = xgb.XGBClassifier().fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Force plot — one prediction's attributions
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Summary plot — global feature importance with direction
shap.summary_plot(shap_values, X_test)

# Permutation importance — model-agnostic
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)
for i in result.importances_mean.argsort()[::-1][:5]:
    print(f"{X_val.columns[i]:30s}  {result.importances_mean[i]:.3f}")
```

</div>

<div class="level-next">
<span>Want global vs local, axiomatic guarantees, & mechanistic methods?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Shapley value</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!} \big[\, f(S \cup \{i\}) - f(S)\,\big] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Marginal contribution of feature *i*, averaged over all subsets

</li>
<li markdown="1">

Unique attribution method satisfying efficiency, symmetry, dummy, additivity

</li>
<li markdown="1">

Exact computation is exponential — SHAP uses tractable approximations

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \phi_i \;=\; \text{weighted average across all feature subsets } S \text{ of}\; \big[\,\text{prediction with } i \text{ added to } S \;-\; \text{prediction with just } S\,\big] $$</span>

**In words.** The Shapley value for feature *i* answers "how much did feature *i* contribute, on average, across every possible context?" For each subset *S* of the other features, you ask: "what does the prediction become if I add feature *i* to this subset?" That difference is feature *i*'s marginal contribution given context *S*. Average those marginals across every possible context, with the specific weighting in the formula (which counts each "size of context" equally), and you get the unique attribution satisfying four natural axioms (efficiency, symmetry, dummy, additivity). Computing exactly is exponential in feature count; SHAP uses clever approximations.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`φᵢ`Shapley value (attribution) for feature *i*

</li>
<li markdown="1">

`S`a subset of features *not including i*

</li>
<li markdown="1">

`f(S ∪ {i}) − f(S)`marginal contribution of *i* given the context *S*

</li>
<li markdown="1">

The factorial weights ensure each "context size" is averaged equally

</li>
<li markdown="1">

Unique method satisfying efficiency, symmetry, dummy, additivity

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Axiomatic attribution.** SHAP, Integrated Gradients, and LIME each obey specific axioms — efficiency, sensitivity, implementation invariance. The choice of axioms determines the method. There's no "right" axiom set; pick what matches your need.

**Local vs global explanations.** Local: "why this specific prediction?" — SHAP-per-instance, IG, LIME. Global: "what does the model rely on overall?" — permutation importance, mean |SHAP|, partial dependence plots. Both are useful for different questions.

**Partial dependence plots (PDP) and ICE.** PDP: average the model's prediction over the training distribution, varying one feature. ICE: do the same per-instance. Useful for visualising the marginal effect of one feature on the model's prediction.

**Counterfactual explanations.** "What's the smallest change to the input that flips the prediction?" Useful for actionable explanations: "to get the loan, raise income by $5k". Tools: DiCE, Alibi, scikit-learn-compat libraries.

**Probing classifiers.** Train a small linear classifier on a model's internal representations to predict some property. Reveals what information is linearly decodable from the model's intermediate states. Used heavily in interpretability of language models.

**Concept-based methods.** TCAV (Kim et al. 2018) — measure how sensitive a prediction is to a high-level concept (defined by examples). Useful for "does the model use this concept?" rather than "this raw feature?".

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from captum.attr import IntegratedGradients, LayerGradCam

# Integrated Gradients — axiomatic attribution for neural networks
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baselines=baseline, return_convergence_delta=True)

# GradCAM — class-discriminative heatmap from convolutional layers
cam = LayerGradCam(model, model.layer4[-1])
heat = cam.attribute(input, target=pred_class)
```

</div>

<div class="level-next">
<span>Want mechanistic interpretability, circuits, & sparse autoencoders?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Sparse autoencoder objective</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \mathcal{L} = \big\lVert x - \hat x \big\rVert^2 + \lambda\,\lVert z \rVert_1, \quad z = \sigma(W_\text{enc} x + b) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Learn a sparse code *z* over the network's hidden state

</li>
<li markdown="1">

Activations of single *z*-dimensions tend to correspond to interpretable features

</li>
<li markdown="1">

Anthropic / DeepMind's flagship mech-interp tool

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{loss} \;=\; \text{(squared reconstruction error)} \;+\; \lambda \times \text{(sum of absolute values of the sparse code)} $$</span>

**In words.** A sparse autoencoder learns to compress a network's hidden state *x* into a sparse code *z* and then reconstruct *x* from *z*. The loss has two competing terms: the first measures how badly the reconstruction *x̂* misses *x* (squared error); the second is an L1 penalty that pressures the code *z* to be mostly zero. `λ` (lambda) controls how aggressively to sparsify. After training, individual dimensions of *z* tend to light up for interpretable concepts — one for "fruit", one for "lawsuit", one for "Python code". `σ` here is a nonlinearity (typically ReLU).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`x`the original hidden state of the network

</li>
<li markdown="1">

`x̂`the reconstruction from the sparse code

</li>
<li markdown="1">

`z`the sparse code (mostly zeros after training)

</li>
<li markdown="1">

`λ`sparsity strength — bigger λ forces more zeros

</li>
<li markdown="1">

Single *z*-dimensions tend to correspond to interpretable features

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Mechanistic interpretability.** Aim: understand a neural network the way you'd understand a compiled program — find the algorithms it implements. Olah et al.'s Circuits work (Distill 2020+), Anthropic's induction heads (2022), Nanda's modular arithmetic (2023). Real progress on toy and small-real models.

**Probing & representational similarity.** Probe with a small classifier; representational similarity analysis (RSA); centered kernel alignment (CKA). All are ways of asking "what information is in this layer?" without intervening on the model.

**Causal interpretability.** Patch activations between forward passes; ablate specific neurons or attention heads; measure the effect. Activation patching (Meng et al. 2022), path patching (Goldowsky-Dill 2023). The most rigorous way to attribute behaviour to components.

**Sparse autoencoders for feature discovery.** Train a sparse autoencoder on a model's hidden states. The sparse codes' dimensions tend to correspond to interpretable features ("this neuron fires for fruit"). Bricken et al. 2023 (Anthropic), Cunningham et al. 2023.

**Why "attention is not explanation" matters.** Attention weights are *part of* the computation but they don't tell you what the model is computing. Two attention distributions can lead to the same output; high attention to a token doesn't mean the model needs that token. Use attention as a hypothesis generator, not an answer.

**Limits and dangers of explanations.** Even faithful explanations can mislead — humans confabulate. "Right answer for the wrong reason" is hard to detect. Adversarial explanations exist (Slack et al. 2020 — fool LIME / SHAP). Treat explanation as a debugging tool, not as truth.

**Mechanistic for LLMs.** The frontier — induction heads, name mover heads, indirect object identification, modular arithmetic circuits, the "in-context learning" mechanism. Anthropic and OpenAI have dedicated interpretability teams; results are technical and incremental.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torch.nn as nn

# Sparse autoencoder over a transformer's hidden states
class SAE(nn.Module):
    def __init__(self, d_model, d_dict, lam=1e-3):
        super().__init__()
        self.W_enc = nn.Linear(d_model, d_dict)
        self.W_dec = nn.Linear(d_dict, d_model, bias=False)
        self.lam = lam

    def forward(self, x):
        z = torch.relu(self.W_enc(x))
        x_hat = self.W_dec(z)
        loss = (x - x_hat).pow(2).mean() + self.lam * z.abs().mean()
        return x_hat, z, loss

# Train on hidden activations harvested from a transformer pass
hidden = transformer(text).hidden_states[layer]
x_hat, z, loss = sae(hidden)
# z has thousands of dimensions, most of which are zero at any given input.
# Look at what each non-zero feature fires on across many examples.
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

[Molnar — Interpretable Machine Learning <i class="fas fa-external-link-alt"></i>](https://christophm.github.io/interpretable-ml-book/){: target="_blank" }
<span class="annotation">Free, comprehensive textbook on practical interpretability. SHAP, LIME, PDP, ICE, counterfactuals — all worked examples.</span>

</li>
<li data-tier="intuition" markdown="1">

[Distill — Circuits Thread <i class="fas fa-external-link-alt"></i>](https://distill.pub/2020/circuits/){: target="_blank" }
<span class="annotation">Olah et al.'s series on mechanistic interpretability of convolutional networks. The visual style still sets the standard.</span>

</li>
<li data-tier="indepth" markdown="1">

[Anthropic — Transformer Circuits <i class="fas fa-external-link-alt"></i>](https://transformer-circuits.pub/){: target="_blank" }
<span class="annotation">Anthropic's mech-interp blog. Induction heads, sparse autoencoders, and superposition. The frontier of "what's inside transformers".</span>

</li>
<li data-tier="fundamentals" markdown="1">

[SHAP — Documentation <i class="fas fa-external-link-alt"></i>](https://shap.readthedocs.io/){: target="_blank" }
<span class="annotation">The reference SHAP library. Practical tutorials for trees, deep models, kernels.</span>

</li>
</ul>

</div>

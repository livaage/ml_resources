---
title: Naive Bayes — ML Resources Hub
eyebrow_text: ← Theory · Probabilistic Models
eyebrow_href: ../theory.html
heading: Naive Bayes
lead: A surprisingly effective classifier built on a deliberately naive independence assumption.
next_href: gaussian-mixture-models.html
next_title: Gaussian Mixture Models
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Pretend each feature is independent of the others, then multiply probabilities.** The assumption is almost always wrong, but the math becomes so simple — and so fast — that the model is surprisingly competitive on text and other discrete-feature problems.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Each ellipse is axis-aligned — that's the "naive" assumption — and the heatmap is the posterior class probability</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Data
                <select id="viz-nb-data"></select>
</label>
<button id="viz-nb-reset" type="button">Re-sample</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-nb-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-nb-caption"></div>
</div>

<script src="{{root}}js/viz/naive-bayes.js"></script>

Naive Bayes models each class with a diagonal-covariance Gaussian — features are assumed independent within each class. That means the ellipses are always parallel to the axes, and the decision boundary is a conic section. Try the *Tilted* dataset to see the assumption fail: the true direction of variation is diagonal, but NB can't see it. Despite that, on high-dimensional sparse data (text, bag-of-words) the simplicity often wins anyway.
{: .viz-intro }

<article class="tldr-body" markdown="1">

Naive Bayes asks: given the words in this email, how likely is each class (spam vs. not spam)? It computes that by pretending each word is independent of every other word — which is obviously false in real language — but then it argmaxes over classes, and the wrong probabilities often still rank the classes correctly.

It's fast, trains in a single pass over the data, handles thousands of features without breaking a sweat, and is still the default baseline for many text-classification problems.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Text classification (spam, sentiment, topic)
- You need a fast baseline before trying anything fancier
- Training data is small but you have many features
- You need online / streaming training

</div>

<div class="no" markdown="1">

### Skip it when

- Feature interactions matter (the independence assumption costs too much)
- Continuous features without natural Gaussian structure
- You need calibrated probability estimates
- You're modelling images or sequences

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = CountVectorizer()
X = vec.fit_transform(texts)

clf = MultinomialNB().fit(X, labels)
print("Top spam words:", vec.get_feature_names_out()[clf.feature_log_prob_[1].argsort()[-10:]])
```

</div>

<div class="level-next">
<span>Want the actual Bayes' rule?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Key idea</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ P(y \mid \mathbf{x}) \;\propto\; P(y)\,\prod_{i=1}^{d} P(x_i \mid y) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`P(y)`class prior — proportion of each class in training data

</li>
<li markdown="1">

`P(xi|y)`per-feature likelihood, modelled by a simple distribution

</li>
<li markdown="1">

The "naive" part is the product — features assumed independent given *y*

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{score}(y) \;=\; P(y) \;\times\; P(\text{feature}_1 \mid y) \;\times\; P(\text{feature}_2 \mid y) \;\times\; \cdots \;\times\; P(\text{feature}_d \mid y) $$</span>

**In words.** To classify a new example, compute a score for each candidate class `y` and pick the biggest. The score is the class's *prior* probability (how common that class is) multiplied by the probability of each feature value *given* that class. The big product `∏` just means "multiply the feature likelihoods together across all features 1 through *d*" — the "naive" trick is that we pretend the features don't depend on each other once you fix the class. The `∝` sign means "proportional to": we've dropped the denominator from Bayes' rule because it's the same for every class and doesn't affect which class wins.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`score(y)`unnormalised posterior for class *y* — argmax over this gives the prediction

</li>
<li markdown="1">

`P(y)`class prior — fraction of training examples in class *y*

</li>
<li markdown="1">

`P(featurei | y)`likelihood of feature *i*'s value, fit independently per class

</li>
<li markdown="1">

Multiply all feature likelihoods together — that's the "naive" independence assumption

</li>
</ul>

</div>

Bayes' rule plus the conditional-independence assumption. The denominator `P(x)` doesn't depend on *y*, so it drops out when comparing classes.

</div>

<article class="tldr-body" markdown="1">

**Variants by likelihood.** *Multinomial* NB models word counts and is the classic choice for text. *Bernoulli* NB models presence / absence of words (useful for short documents). *Gaussian* NB models continuous features as Gaussian per class. The structure is the same; only the per-feature density changes.

**Laplace smoothing.** If a word never appeared with class *y* in training, *P(x<sub>i</sub> | y) = 0* would zero the entire product. Add-one (Laplace) smoothing replaces zeros with a small prior count, which is essentially a Dirichlet prior on the categorical likelihood.

**Why it works despite being wrong.** Naive Bayes is a biased probability estimator but a competitive *ranker*: the argmax over classes is often correct even when the individual probabilities are miscalibrated. For decision-making it's fine; for confidence estimates it isn't.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Bag-of-words text classification
- Document filtering with online updates
- You want a transparent baseline you can read off
- Class priors shift between train and test (you can re-prior easily)

</div>

<div class="no" markdown="1">

### Skip it when

- Probability calibration matters — the over-confidence is notorious
- Features are heavily correlated and you can't decorrelate them
- Continuous features that are multi-modal per class
- Sequence structure matters (use a sequence model)

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1, 2))),
    ("nb",    MultinomialNB(alpha=0.1)),    # Laplace smoothing
])
pipe.fit(X_train_text, y_train)

y_pred = pipe.predict(X_test_text)
print(classification_report(y_test, y_pred))
```

</div>

<div class="level-next">
<span>Want the connection to logistic regression and the failure modes?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Log-posterior (linear in features)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \log P(y \mid \mathbf{x}) \;=\; \log P(y) \;+\; \sum_{i=1}^{d} \log P(x_i \mid y) \;+\; \text{const} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

The log-posterior is **linear** in the features (for many likelihood families)

</li>
<li markdown="1">

Naive Bayes is therefore a linear classifier in disguise

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \log(\text{score}(y)) \;=\; \log P(y) \;+\; \big[\log P(\text{feat}_1 \mid y) + \log P(\text{feat}_2 \mid y) + \cdots\big] \;+\; \text{const} $$</span>

**In words.** Take the logarithm of both sides of the Fundamentals formula and the product turns into a *sum* — the `Σ` symbol just means "add up across every feature *i*." That's a huge practical convenience: multiplying lots of tiny probabilities underflows to zero on a computer, but adding their logs is numerically stable. The "const" is just `log P(x)`, which is the same for every class and so doesn't affect argmax. The deeper point: this expression is *linear* in the (logged) feature counts — exactly the shape of a linear classifier like logistic regression.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`log score(y)`log of the unnormalised posterior — bigger means more likely

</li>
<li markdown="1">

`log P(y)`log of the class prior — a per-class bias

</li>
<li markdown="1">

`log P(feati | y)`log-likelihood of feature *i* under class *y* — a per-feature weight

</li>
<li markdown="1">

Linear in the features → naive Bayes is a linear classifier in disguise

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**NB vs. logistic regression.** Both produce linear decision boundaries in the log-likelihood representation. NB is a *generative* model: it models *P(x, y) = P(y)·P(x|y)*. Logistic regression is *discriminative*: it models *P(y | x)* directly. Ng & Jordan (2001) showed that NB reaches its (worse) asymptotic error faster — so NB wins at small *N*, LR wins at large *N*.

**Complement NB** (Rennie et al. 2003). For unbalanced text classification, modeling *P(x | ¬y)* instead of *P(x | y)* and flipping signs reduces the bias toward the majority class. Implemented in sklearn as `ComplementNB` — often beats MultinomialNB on real text.

**Calibration.** NB's probability estimates are systematically over-confident because the independence assumption underestimates correlated evidence (the same signal gets counted multiple times). Post-hoc calibration (Platt scaling, isotonic regression) on a held-out set can fix the probabilities without changing the classifier.

**Connection to language modelling.** A unigram language model with class-conditional unigrams *is* multinomial NB. n-gram smoothing techniques (Good-Turing, Kneser-Ney) directly carry over.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Limited labelled data, many features — NB's asymptotic-error trade-off wins
- Streaming setting with class priors that drift
- Interpretable baseline alongside a heavier model for ablation
- You need very fast inference (e.g. online filtering)

</div>

<div class="no" markdown="1">

### Skip it when

- Long-form text or sequence structure — n-gram NB caps at small windows
- Continuous features are multi-modal — Gaussian NB will get badly wrong
- You need probabilities used in downstream Bayesian computation
- Adversarial inputs — easily exploited via crafted feature combinations

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ComplementNB for imbalanced text + isotonic calibration for honest probabilities
base = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=5, sublinear_tf=True, ngram_range=(1, 2))),
    ("nb",    ComplementNB(alpha=0.1)),
])
clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
clf.fit(X_train, y_train)

# Now clf.predict_proba is well-calibrated on held-out data
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

[scikit-learn user guide <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/naive_bayes.html){: target="_blank" }
<span class="annotation">Practical comparison of all four NB variants (Gaussian / Multinomial / Complement / Bernoulli).</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Jurafsky & Martin, ch. 4 <i class="fas fa-external-link-alt"></i>](https://web.stanford.edu/~jurafsky/slp3/4.pdf){: target="_blank" }
<span class="annotation">NB applied to text classification with worked examples — the clearest intro for NLP context.</span>

</li>
<li data-tier="indepth" markdown="1">

[Ng & Jordan (2001) <i class="fas fa-external-link-alt"></i>](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf){: target="_blank" }
<span class="annotation">"On Discriminative vs. Generative Classifiers" — when does NB beat logistic regression? Short and clear.</span>

</li>
<li data-tier="indepth" markdown="1">

[Rennie et al. (2003) <i class="fas fa-external-link-alt"></i>](https://www.aaai.org/Papers/ICML/2003/ICML03-081.pdf){: target="_blank" }
<span class="annotation">"Tackling the Poor Assumptions of Naive Bayes" — introduces Complement NB and explains practical failure modes.</span>

</li>
</ul>

</div>

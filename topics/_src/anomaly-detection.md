---
title: Anomaly Detection — ML Resources Hub
eyebrow_text: ← Theory · Model Families
eyebrow_href: ../theory.html
heading: Anomaly Detection
lead: Find the points that don't look like the rest — fraud, defects, intrusions, the rare and weird.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Learn what "normal" looks like; flag what doesn't.** Anomaly detection is unsupervised — you usually don't have labelled anomalies, just the assumption that most of your data is normal and a few rare points are not.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Slide the threshold — every point below the cutoff density gets flagged in terracotta</span>
</div>
<div class="viz-classic-controls">
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                Data
                <select id="viz-anom-data"></select>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                τ
                <input id="viz-anom-threshold" class="viz-classic-slider" type="range"></input>
</label>
<label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
                h
                <input id="viz-anom-bw" class="viz-classic-slider" type="range"></input>
</label>
<button id="viz-anom-reset" type="button">Re-sample</button>
<span class="viz-classic-badge" id="viz-anom-threshold-lbl">τ = 0.25</span>
<span class="viz-classic-badge" id="viz-anom-bw-lbl">h = 0.12</span>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-anom-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-anom-caption"></div>
</div>

<script src="{{root}}js/viz/anomaly.js"></script>

The cream-to-indigo heatmap is a kernel density estimate of "where normal lives". Points are flagged as anomalies when their local density score drops below the threshold τ. Drop the bandwidth *h* and the model becomes sensitive to local quirks (over-fits); raise it and only the most isolated points stand out. The *Ring* dataset is a classic — the genuinely anomalous points live *inside* the ring, where most distance-based methods would happily call them "central and normal".
{: .viz-intro }

<article class="tldr-body" markdown="1">

Three classical strategies. **Density-based**: a point with low probability under a model of "normal" is anomalous. **Distance-based**: a point far from its nearest neighbours is anomalous. **Reconstruction-based**: train a model to compress and reconstruct normal data; points it reconstructs badly are anomalous.

The right choice depends on what "anomalous" means in your domain — a fraudster looks different from a manufacturing defect looks different from a network intrusion.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- Fraud / intrusion / defect detection
- You have plenty of normal data but few or no labelled anomalies
- Monitoring sensor data for unusual patterns
- Cleaning a dataset of outliers before modelling

</div>

<div class="no" markdown="1">

### Skip it when

- You have labels for both classes — train a regular classifier (with class weights)
- Anomalies are common enough to balance — it's just classification
- "Anomalous" isn't well-defined and changes over time
- You need to explain why a specific point was flagged

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=0).fit(X_train)

# -1 = anomaly, 1 = normal
labels = iso.predict(X_test)
scores = iso.score_samples(X_test)   # lower = more anomalous
```

</div>

<div class="level-next">
<span>Want the actual algorithms?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Three strategies</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{score}(x) \;\in\; \big\{\, \underbrace{-\log p(x)}_{\text{density}},\; \underbrace{d(x, \mathrm{NN}_k(x))}_{\text{distance}},\; \underbrace{\|x - \hat{x}\|}_{\text{reconstruction}} \,\big\} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Each strategy assigns a continuous score — threshold to decide normal vs. anomalous

</li>
<li markdown="1">

Choice of strategy = assumption about what makes anomalies "weird"

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{anomaly score}(x) \;=\; \text{one of:}\;\big\{\,\text{low density},\;\text{far from neighbours},\;\text{poor reconstruction}\,\big\} $$</span>

**In words.** Three ways to assign a "weirdness" score to a point *x*. *Density*: how (im)probable is *x* under a model of normal data? `−log p(x)` is the "surprise" — bigger means rarer. *Distance*: how far is *x* from its *k* nearest neighbours (written `NNk(x)`)? Lone points are anomalous. *Reconstruction*: train a compressor on normal data; if it can't faithfully reconstruct *x*, then *x* doesn't look normal. Each strategy picks a different definition of "weird"; pick the one that matches how your anomalies actually behave.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`density`negative log probability under a model of normal data — bigger = rarer

</li>
<li markdown="1">

`distance`how far *x* is from its *k* nearest neighbours

</li>
<li markdown="1">

`reconstruction`how badly an autoencoder reconstructs *x*

</li>
<li markdown="1">

Each strategy gives a continuous score; threshold to decide normal vs. anomalous

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Isolation Forest.** Builds random trees that split features at random thresholds. Anomalies — being in sparse regions — get isolated in fewer splits. The score is average path length to isolation. Cheap, scales well, and works in moderate dimensions. Default for tabular anomaly detection.

**One-Class SVM.** Fits a decision boundary around the "normal" data in a kernel feature space, treating the origin as the "anomaly side". Good with small data and a sensible kernel; doesn't scale to big data.

**Local Outlier Factor (LOF).** Compares each point's local density to its neighbours' local densities. Catches anomalies in heterogeneous-density data that global methods miss. Score > 1 means lower density than neighbours.

**Autoencoder reconstruction.** Train an autoencoder on normal data; at inference time, flag points with high reconstruction error. Scales to images and high-dim data where other methods struggle.

**Threshold setting.** All methods give continuous scores; the threshold is a business decision (precision vs. recall trade-off). With no labels, use a quantile of training scores; with some labels, calibrate against the validation set.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- **Isolation Forest:** moderate-dim tabular — strong default
- **One-Class SVM:** small / clean dataset with kernel intuition
- **LOF:** heterogeneous local density (clusters of different sizes)
- **Autoencoder:** images, time series, high-dim structured data

</div>

<div class="no" markdown="1">

### Skip it when

- "Anomalous" depends on labelled examples and you have plenty of them
- The score distribution is bi-modal and no threshold cleanly separates classes
- Anomalies arrive in groups (contextual) — single-point methods miss the group
- The data-generating process drifts over time — model becomes stale

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

X_scaled = StandardScaler().fit_transform(X)

methods = {
    "iforest":  IsolationForest(contamination=0.05, random_state=0),
    "lof":      LocalOutlierFactor(n_neighbors=20, contamination=0.05),
    "ocsvm":    OneClassSVM(nu=0.05, kernel="rbf", gamma="scale"),
}

for name, m in methods.items():
    if hasattr(m, "fit_predict"):
        labels = m.fit_predict(X_scaled)         # for LOF
    else:
        labels = m.fit(X_scaled).predict(X_scaled)
    print(f"{name:10s} flagged {(labels == -1).sum():,} points as anomalous")
```

</div>

<div class="level-next">
<span>Want deep methods and contextual / collective anomalies?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Three flavours of anomaly</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{point} \;\subset\; \text{contextual} \;\subset\; \text{collective} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

**Point**: one observation is anomalous regardless of context

</li>
<li markdown="1">

**Contextual**: an observation is anomalous given a context (time, location)

</li>
<li markdown="1">

**Collective**: a sequence / group is anomalous as a whole

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{point anomaly} \;\subset\; \text{contextual anomaly} \;\subset\; \text{collective anomaly} $$</span>

**In words.** Anomalies come in three nested types of increasing subtlety. A *point anomaly* is a single observation that's odd regardless of anything around it (a credit card charge in a country you've never visited). A *contextual anomaly* is odd only in the right context — 25°C is normal in summer, anomalous in winter; you need to know the season to flag it. A *collective anomaly* requires looking at a whole group together — no single heart-rate reading is unusual, but the whole sequence drifting upward over an hour is. Each type subsumes the previous and is harder to detect.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`point`one observation odd on its own

</li>
<li markdown="1">

`contextual`odd given the context (time of day, location, season)

</li>
<li markdown="1">

`collective`a sequence or group is anomalous as a whole, even if each piece looks normal

</li>
<li markdown="1">

Each type is harder to detect than the previous

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Deep one-class methods.** Deep SVDD (Ruff et al.) replaces the kernel feature map with a learned neural network and shrinks the data into a small hypersphere. Trained end-to-end; works for images and time series. Watch for representation collapse (every input maps to the centre).

**Generative anomaly detection.** Train a generative model (GAN, normalizing flow, diffusion) on normal data; anomalies have low likelihood or low-quality reconstructions. State of the art on industrial defect detection (MVTec AD). Caveat: deep generative models *do not* reliably assign low likelihood to out-of-distribution inputs — see Nalisnick et al. 2019.

**Sequence and time-series.** Forecast-based methods flag points where the prediction error exceeds a threshold (ARIMA residuals, Prophet, deep forecast models). Reconstruction-based methods (LSTM autoencoders, transformer denoisers) work for collective anomalies — flag a window whose reconstruction is poor.

**Calibration.** Anomaly scores are not probabilities. Convert via percentile-based mapping or fit a tail distribution (Generalized Pareto). PR-AUC is the right summary metric in the heavily-imbalanced regime; ROC-AUC overstates performance.

**Evaluation pitfalls.** Without labels, you're estimating performance from synthetic anomalies — which often don't match real ones. With labels, watch out for label leakage from the threshold-setting process. Always evaluate on a held-out period for time-series.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- **Deep SVDD:** images, learned representations, end-to-end pipeline
- **Normalizing flows:** need calibrated densities, not just scores
- **LSTM / transformer reconstruction:** sequential data with structure
- **Density-ratio:** compare against a known reference distribution

</div>

<div class="no" markdown="1">

### Skip it when

- You truly have labels — use supervised methods with class weights / focal loss
- "Normal" is multi-modal and rare — single-class methods overfit one mode
- Anomalies must be human-interpretable — deep methods are opaque
- You can't retrain regularly and the data drifts

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn

class AutoencoderAD(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, 16))
        self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d_in))
    def forward(self, x):
        return self.dec(self.enc(x))

# Train ONLY on normal data
model = AutoencoderAD(d_in=X_normal.shape[1])
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(100):
    opt.zero_grad()
    loss = ((model(X_normal) - X_normal) ** 2).mean()
    loss.backward(); opt.step()

# Anomaly score = per-sample reconstruction error
with torch.no_grad():
    err = ((model(X_test) - X_test) ** 2).mean(dim=1)
threshold = err[y_test == 0].quantile(0.99)    # top 1% of normal training errors
flagged = err > threshold
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

[scikit-learn — Outlier detection <i class="fas fa-external-link-alt"></i>](https://scikit-learn.org/stable/modules/outlier_detection.html){: target="_blank" }
<span class="annotation">Side-by-side comparison of Isolation Forest, One-Class SVM, LOF, and Elliptic Envelope on toy data. Start here.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[PyOD library <i class="fas fa-external-link-alt"></i>](https://pyod.readthedocs.io/){: target="_blank" }
<span class="annotation">Unified API across ~40 anomaly-detection algorithms — classical, deep, and ensemble. Includes benchmarks.</span>

</li>
<li data-tier="indepth" markdown="1">

[Ruff et al. — Deep AD survey (2021) <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2007.02500){: target="_blank" }
<span class="annotation">"A Unifying Review of Deep and Shallow Anomaly Detection" — best modern survey. Covers SVDD, generative, self-supervised methods.</span>

</li>
<li data-tier="indepth" markdown="1">

[MVTec AD benchmark <i class="fas fa-external-link-alt"></i>](https://www.mvtec.com/company/research/datasets/mvtec-ad){: target="_blank" }
<span class="annotation">Industrial defect-detection benchmark. The de-facto standard for evaluating image-based anomaly methods.</span>

</li>
</ul>

</div>

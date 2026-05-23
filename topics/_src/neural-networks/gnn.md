---
title: Graph Neural Networks (GNN) — ML Resources Hub
eyebrow_text: ← Theory · Neural Networks
eyebrow_href: {{root}}theory.html
heading: Graph Neural Networks
lead: Networks that process graph-structured data — molecules, social networks, knowledge graphs, code.
prev_href: transformer.html
prev_title: Transformers
next_href: autoencoder.html
next_title: Autoencoders & VAEs
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">
<span class="key-idea-label">Key idea</span>

**Every node updates itself by listening to its neighbours.** Like a social network of nodes passing notes — each round, every node reads what its neighbours just said, blends it with its own state, and updates. Do this *K* times and even far-away nodes have rippled in.
</div>

<div class="viz-embed viz-gnn" data-fig="gnn">
    <div class="viz-task-banner">
        <span class="viz-task-label">Task</span>
        <span class="viz-task-text">
            Each node updates itself by <strong>aggregating messages</strong> from its neighbours; repeat for <em>K</em> steps and every node carries information from its <em>K</em>-hop neighbourhood. Pick a source, then watch how a node's representation evolves as messages propagate.
        </span>
    </div>
    <div class="viz-embed-header">
        <span class="viz-embed-title">Click any node to make it the source · then "Step" or "Auto" to propagate</span>
    </div>
    <div class="viz-gnn-controls">
        <button id="viz-gnn-step"  type="button">Step</button>
        <button id="viz-gnn-reset" type="button">Reset</button>
        <button id="viz-gnn-play"  type="button">Auto</button>
        <label style="display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.85rem;">
            Aggregation
            <select id="viz-gnn-agg"></select>
        </label>
        <span class="viz-gnn-round" id="viz-gnn-round">Round 0</span>
    </div>
    <div class="viz-gnn-canvas-wrap">
        <canvas id="viz-gnn-canvas"></canvas>
    </div>
    <div class="viz-gnn-caption" id="viz-gnn-caption"></div>
</div>
<script src="{{root}}js/viz/gnn.js"></script>

<article class="tldr-body" markdown="1">

### Why graph structure can't just be flattened

A graph has no canonical ordering — relabel the nodes and it's the *same* graph. Flatten it into a vector and you've baked in a fake order; train an MLP on that and the model will treat node 3 differently from node 7 even when they play identical structural roles. On top of that, **nodes have variable numbers of neighbours** — atom 1 might bond to two others, atom 2 to four — so you can't even use a fixed-size input vector. GNNs solve both: the aggregation step is *permutation-invariant* (sum / mean / max don't care about order), and it works on any neighbourhood size.

### How message passing works intuitively

Picture each node holding a small vector — its current "state". One round of message passing is:

1. Every node sends its state along each outgoing edge.
2. Every node collects everything it received and combines it (sum, mean, max, or learned attention).
3. Every node mixes that aggregate with its old state to produce a new state.

Repeat *K* times. After one round, a node knows about its immediate neighbours. After two, its neighbours' neighbours. After *K*, its entire *K*-hop neighbourhood. Click a node on the edge of the figure and count how many rounds the signal needs to cross the graph.

### Why depth matters — but also hurts

More layers means longer-range information, which sounds good. The catch is **over-smoothing**: keep averaging neighbours into yourself and after enough rounds, every node converges to the same blurry mean of the whole graph. You've lost the very distinctions you were trying to learn. That's why most GNNs use only 2–4 layers, with skip connections or jumping-knowledge tricks if you need more.

### Node, edge, or graph-level tasks

The same backbone serves three task families:

- **Node-level** — one prediction per node (which user will churn? what role does this protein play?).
- **Edge-level** — predict whether an edge should exist or what kind it is (link prediction, knowledge-graph completion).
- **Graph-level** — one prediction for the whole graph, formed by pooling all node embeddings into a single vector (will this molecule bind to the target?).

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- The edges *carry meaning* — bonds in a molecule, friendships, citations, road segments
- Molecular property prediction or drug discovery
- Recommendation on bipartite user-item graphs
- Knowledge-graph completion or reasoning over relations

</div>
<div class="no" markdown="1">

### Skip it when

- Your data isn't naturally a graph (don't force it)
- The graph is huge but the structure is incidental — try a baseline that ignores it first
- You only have node features and no informative edges — an MLP will do
- Long-range dependencies dominate — a graph transformer may beat a plain GNN

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
```

</div>

<div class="level-next">
    <span>Want the message-passing math?</span>
    <button data-go-to="fundamentals" type="button">Switch to Fundamentals →</button>
</div>

</section>


<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">Message passing</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \mathbf{h}_v^{(\ell+1)} \;=\; \phi\!\left(\mathbf{h}_v^{(\ell)},\; \bigoplus_{u \in \mathcal{N}(v)} \psi(\mathbf{h}_u^{(\ell)}, \mathbf{h}_v^{(\ell)}, \mathbf{e}_{uv})\right) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1"><code>h<sub>v</sub><sup>(ℓ)</sup></code>feature vector of node *v* at layer *ℓ*</li>
<li markdown="1"><code>𝒩(v)</code>neighbours of node *v*</li>
<li markdown="1"><code>ψ, φ</code>learned message and update functions (usually small MLPs)</li>
<li markdown="1"><code>⊕</code>permutation-invariant aggregation — sum, mean, max, or attention</li>
<li markdown="1"><code>e<sub>uv</sub></code>(optional) feature vector on the edge from *u* to *v*</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{new features of } v \;=\; \text{update}\!\left(\text{old features of } v,\; \text{aggregate}\big[\,\text{message}(u, v, \text{edge})\,\big]_{u \in \text{neighbours}(v)}\right) $$</span>

**In words.** Every node carries a feature vector. To update node *v*, do three things in order. First, for each **neighbour** *u*, compute a **message** — a small learned function of *u*'s features, *v*'s features, and the edge between them. Second, **aggregate** those messages with a permutation-invariant combiner (usually sum, mean, or max) so the result doesn't depend on which order the neighbours showed up in. Third, combine the aggregate with *v*'s previous features using another learned function (the **update**). Stack a few layers and information ripples outward. The funny `⊕` symbol is just a placeholder for whichever order-free aggregator you chose.
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**The aggregate-then-update recipe.** Most GNN papers boil down to picking a *message function*, an *aggregator*, and an *update function*. Everything is differentiable end-to-end, so you train the whole stack with the loss of your downstream task (node classification, link prediction, graph regression).

**GCN vs. GraphSAGE vs. GAT.** Three classic choices, each making a different trade-off:

- **GCN** ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)) — a single linear transform plus a degree-normalized sum over neighbours. Cheapest baseline; works surprisingly often.
- **GraphSAGE** ([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)) — samples a fixed number of neighbours per node so you can train on huge graphs without loading the whole adjacency. Aggregator is configurable (mean, LSTM, pool).
- **GAT** ([Veličković et al., 2018](https://arxiv.org/abs/1710.10903)) — replaces the fixed neighbour weights with **attention** computed on each edge, so the model learns which neighbours matter for which node. Multi-head attention stabilises training.

**Node, edge, and graph features.** Edges can carry their own learned features (bond type, friendship strength), which the message function takes as a third argument. Graph-level tasks need a **readout / pooling** step that collapses all node embeddings into one vector — mean and sum pooling are the workhorses; attention pooling and hierarchical pooling (DiffPool) help on harder tasks.

**Batching graphs.** Unlike images, graphs in a batch have different node counts and edge counts. Frameworks like PyTorch Geometric handle this by concatenating all graphs into one big disconnected supergraph and keeping a `batch` vector that records which node belongs to which graph. Pooling layers read that vector to keep predictions per-graph.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- Molecule or protein property prediction with explicit chemical structure
- Recommendation systems with user-item bipartite graphs
- Citation / co-authorship / social-network analysis
- Traffic and road-network forecasting where topology matters

</div>
<div class="no" markdown="1">

### Skip it when

- A graph-blind baseline already wins — never deploy a GNN you didn't sanity-check against an MLP
- Very long-range dependencies — vanilla message passing over-smooths; reach for a graph transformer
- The graph changes faster than you can train — use streaming or incremental methods
- You need calibrated uncertainty — plain GNNs are over-confident, reach for a Bayesian or ensemble variant

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class MolPropertyGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1)
        self.head  = torch.nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        # Pool per graph (batch holds graph index for each node)
        x = global_mean_pool(x, batch)
        return self.head(x)
```

</div>

<div class="level-next">
    <span>Want the Weisfeiler-Leman ceiling, graph transformers, and how the field actually scales?</span>
    <button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>


<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">
<span class="key-idea-label">WL-1 expressiveness ceiling</span>

<div class="notation-standard" markdown="1">
<span class="formula">$$ \text{standard message-passing GNNs} \;\preceq\; \text{1-WL test}\;<\; \text{graph isomorphism} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">Any standard message-passing GNN can only distinguish graphs that the 1-dimensional Weisfeiler-Leman colour-refinement test can distinguish</li>
<li markdown="1">Some non-isomorphic graphs share WL-1 colourings, so no standard GNN can tell them apart</li>
<li markdown="1">GIN ([Xu et al., 2019](https://arxiv.org/abs/1810.00826)) — sum aggregation + an MLP — *achieves* the WL-1 bound; weaker aggregators (mean, max) fall below it</li>
</ul>

</div>

<div class="notation-plain" markdown="1">
<span class="formula">$$ \text{power of standard GNNs} \;\le\; \text{power of the 1-WL test} \;<\; \text{telling all non-identical graphs apart} $$</span>

**In words.** There's a classical algorithm called the **1-WL test** (1-dimensional Weisfeiler-Leman) that iteratively recolours each node by hashing the multiset of its neighbours' colours. It's a fast heuristic for telling whether two graphs are the same up to relabelling ("isomorphic"). [Xu et al. (2019)](https://arxiv.org/abs/1810.00826) proved that *any* standard message-passing GNN is at most as discriminative as this test — and the test itself fails on some genuinely different graph pairs (e.g. two regular graphs with the same degree sequence). So there are graphs no plain GNN can ever tell apart, no matter how deep or wide. The `≼` symbol means "no more powerful than".
{: .formula-narration }

</div>

</div>

<article class="tldr-body" markdown="1">

**Over-smoothing & over-squashing.** Two failure modes that bite as you stack more layers. **Over-smoothing**: each round mixes neighbours into self, so after enough rounds every node converges to the same vector — the graph mean. **Over-squashing**: information from a distant node has to pass through narrow bottlenecks (low-degree cut edges) and gets crushed into a fixed-size message. Modern fixes: skip / jumping-knowledge connections, normalization, **graph rewiring** (add long-range shortcuts), or switch to attention.

**Expressive power & the Weisfeiler-Leman test.** Standard message-passing GNNs are capped at WL-1 (above). To break the ceiling you need: **higher-order GNNs** (operate on *k*-tuples of nodes — gain expressiveness at *O(N<sup>k</sup>)* cost), **subgraph methods** (SUN, GNN-AK — extract a local subgraph around each node and process it separately), or **equivariant GNNs** like EGNN that use 3D coordinates and stay invariant under rotation / translation — essential for molecules and physics.

**Graph transformers vs. GNNs.** Treat the graph as fully connected and let attention figure out the structure, but inject the graph as an inductive bias — distance encodings, edge encodings, or Laplacian-eigenvector positional encodings ([GraphGPS](https://arxiv.org/abs/2205.12454), [Graphormer](https://arxiv.org/abs/2106.05234)). On small / medium graphs with long-range interactions they tend to beat plain GNNs; on huge sparse graphs the quadratic attention cost is the bottleneck and classical GNNs still win.

**Scaling to web-scale graphs.** Billions of edges don't fit in memory. The three dominant strategies all sample subgraphs:

- **Neighbour sampling** (GraphSAGE) — pick a fixed number of neighbours per node per layer; gradient is unbiased but variance grows with depth.
- **Cluster sampling** (Cluster-GCN) — partition the graph into dense clusters, train one cluster per batch.
- **Layer / subgraph sampling** (GraphSAINT) — sample a connected induced subgraph and normalize importances so the expected loss matches full-batch.

**Real-world wins.** [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2)'s *evoformer* is essentially a graph transformer over residue pairs — message passing on a protein's contact graph is what makes structure prediction work. Pinterest's [PinSage](https://arxiv.org/abs/1806.01973) uses GraphSAGE-style sampling over 3 billion nodes for recommendation. Uber Eats and Google Maps both use GNNs for ETA and routing. In drug discovery, GNN-based property predictors are now standard for screening molecule libraries before any lab work.

</article>

<div class="use-cases" markdown="1">
<div class="yes" markdown="1">

### Reach for it when

- **EGNN / equivariant** — 3D structure-aware tasks (molecules, proteins, particle physics)
- **Graph transformers** — long-range dependencies on small or medium graphs
- **Cluster-GCN / GraphSAGE / SAINT** — web-scale graphs that don't fit in memory
- **Higher-order or subgraph methods** — when WL-1 isn't enough and you can pay the compute

</div>
<div class="no" markdown="1">

### Skip it when

- You're chasing the WL hierarchy but the dataset doesn't actually need it
- Tabular or time-series data dressed up as a graph — use the native model
- You need bounded inference latency on a constantly growing graph
- A handful of hand-crafted structural features plus an MLP already solves the task

</div>
</div>

<div class="code-snippet" markdown="1">
<button class="copy-btn" type="button">Copy</button>

```python
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

# A bare-bones expressive GNN layer (GIN-style) — implemented from scratch.
# Sum-aggregate neighbours, then MLP — provably matches the WL-1 ceiling.
class GINLayer(nn.Module):
    def __init__(self, d_in, d_out, eps=0.0, learn_eps=True):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps)) if learn_eps else eps
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        agg = scatter(x[src], dst, dim=0, dim_size=x.size(0), reduce="sum")
        return self.mlp((1 + self.eps) * x + agg)
```

</div>

<div class="level-next">
    <span>Want the picture instead?</span>
    <button data-go-to="intuition" type="button">← Back to Intuition</button>
</div>

</section>


<!-- TOPIC SIDEBAR -->

<div class="fig-explainer" data-fig="gnn" markdown="1">

### Watching messages propagate

The **circles** are nodes; the **lines** are edges. Click any node to make it the *source* — it lights up with a signal that hasn't yet reached anyone else. Each press of **Step** runs one round of message passing: every node aggregates whatever its neighbours sent last round and updates its colour.

After one step, only the source's immediate neighbours have heard from it. After two, two-hop neighbours. The **round counter** tracks *K*. Switch the **aggregation** dropdown between sum / mean / max and watch how the signal spreads differently — sum amplifies through high-degree hubs, mean averages it out, max keeps only the loudest.

Pick a node on the perimeter and count how many rounds it takes for the whole graph to light up — that's the *diameter* of the graph, and the minimum depth a GNN would need to let any node see any other.

</div>

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="intuition" markdown="1">
[Distill — A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The clearest visual introduction to graph networks anywhere. Interactive demos for every concept — start here.</span>
</li>
<li data-tier="intuition" markdown="1">
[Distill — Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Companion piece to the gentle intro — focuses on *why* graph convolutions are the natural generalisation of image convolutions.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Petar Veličković — GNN lectures & talks](https://petar-v.com/talks/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">From one of the authors of GAT. Slide decks and recorded lectures that bridge intuition and the message-passing formalism with unusual clarity.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[PyTorch Geometric tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html) <i class="fas fa-external-link-alt"></i>
<span class="annotation">The most-used GNN library. Colab notebooks step through GCN, GraphSAGE, GAT, GIN, and graph-level tasks with runnable code.</span>
</li>
<li data-tier="fundamentals" markdown="1">
[Stanford CS224W — Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/) <i class="fas fa-external-link-alt"></i>
<span class="annotation">Jure Leskovec's full course. Slides, recorded lectures, and Colab assignments. The most thorough syllabus on graph ML available for free.</span>
</li>
<li data-tier="indepth" markdown="1">
[Hamilton et al. (2017) — GraphSAGE paper](https://arxiv.org/abs/1706.02216) <i class="fas fa-external-link-alt"></i>
<span class="annotation">"Inductive Representation Learning on Large Graphs" — introduced neighbour sampling, the trick that made web-scale GNNs possible.</span>
</li>
<li data-tier="indepth" markdown="1">
[Veličković et al. (2018) — GAT paper](https://arxiv.org/abs/1710.10903) <i class="fas fa-external-link-alt"></i>
<span class="annotation">"Graph Attention Networks" — brought attention to GNNs. Required reading before reaching for any attention-based variant.</span>
</li>
<li data-tier="indepth" markdown="1">
[Xu et al. (2019) — GIN paper](https://arxiv.org/abs/1810.00826) <i class="fas fa-external-link-alt"></i>
<span class="annotation">"How Powerful are Graph Neural Networks?" — established the WL-1 ceiling and the GIN construction that achieves it. The reference for thinking about expressive power.</span>
</li>
</ul>

</div>
</content>
</invoke>
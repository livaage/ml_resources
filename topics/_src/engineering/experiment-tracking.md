---
title: Experiment Tracking — ML Resources Hub
eyebrow_text: ← Engineering · Development Loop
eyebrow_href: {{root}}engineering.html
heading: Experiment Tracking
lead: Log every run — params, metrics, artefacts — and make them searchable. The single highest-leverage engineering habit in ML.
active_nav: engineering
prev_href: env-deps.html
prev_title: Environments &amp; Dependencies
next_href: hyperparameter-search.html
next_title: Hyperparameter Search
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**If a run isn't logged, it didn't happen.** Treat every training run as an experiment with a unique ID, all its hyperparameters captured, all its metrics streamed, all its artefacts saved. You'll thank yourself in three weeks when you're trying to find "the one where I tried dropout 0.3 with cosine schedule."

</div>

<article class="tldr-body" markdown="1">

Without tracking: spreadsheets, dated folders, and the eternal regret of "which checkpoint was the good one?". With tracking: a searchable dashboard of every run, plottable side-by-side, with every config and artefact one click away.

**What to log.** Hyperparameters, the git commit, the dataset version, training loss and val metrics (per step), system metrics (GPU util), and final artefacts (checkpoints, predictions, plots). For deep learning, also gradients norms and learning rate.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Tools

- **W&B (wandb)**: hosted, free for individuals, nicest UI. Default in many shops.
- **MLflow**: open-source, self-hosted. Solid for production-adjacent workflows.
- **Comet**: similar to W&B, hosted
- **TensorBoard**: built into PyTorch, fine for one-off projects
- **Neptune, ClearML, Aim**: niche but loyal fanbases

</div>

<div class="no" markdown="1">

### What to log

- All hyperparameters, automatically (Hydra integration is nice)
- Train loss + every val metric, per step
- The git commit hash and any uncommitted diffs
- The dataset version / hash
- Final checkpoints and prediction artefacts
- Learning rate, gradient norms, system utilisation

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import wandb
import torch

run = wandb.init(
    project="my-project",
    config={
        "lr":      1e-3,
        "batch":   64,
        "model":   "resnet18",
        "dataset": "cifar10",
    },
    tags=["baseline"],
)

for step in range(num_steps):
    loss = train_step()
    val_acc = validate() if step % 500 == 0 else None
    wandb.log({"train/loss": loss.item(),
               "val/acc":   val_acc,
               "lr":        scheduler.get_last_lr()[0]}, step=step)

# Save the final model
torch.save(model.state_dict(), "final.pt")
wandb.save("final.pt")
run.finish()
```

</div>

<div class="level-next">
<span>Want the schema discipline, comparison patterns, & integration with Hydra?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">A run is a row, a metric is a column</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{Run}: (\text{config}, \text{git}, \text{data version}) \;\to\; (\text{metrics}_{t}, \text{artefacts}) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Every run uniquely identified

</li>
<li markdown="1">

Configs flatten to columns; metrics are time series; artefacts are pointers

</li>
<li markdown="1">

Compare across runs by querying the dashboard

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{Run} : (\text{config},\; \text{git sha},\; \text{data version}) \;\to\; (\text{metrics over time},\; \text{artefacts}) $$</span>

**In words.** Think of every training run as a function from inputs to outputs. The inputs are everything that *defines* the run: the config (hyperparameters), the git commit (which code), and the dataset version. The outputs are the time-series of metrics (loss, accuracy, etc., indexed by step `t`) and the final artefacts (checkpoints, predictions, plots). The arrow `→` here means "produces". In tracking systems this gets flattened into a database row per run — inputs become searchable columns, metrics become charts you can overlay, artefacts become files keyed by run ID.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`config`all hyperparameters and runtime settings

</li>
<li markdown="1">

`git sha`commit hash (ideally plus uncommitted diff) for the code

</li>
<li markdown="1">

`data version`dataset hash or pointer to a fixed dataset version

</li>
<li markdown="1">

`metrics over time`scalar series indexed by training step

</li>
<li markdown="1">

`artefacts`checkpoints, prediction dumps, evaluation reports

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Logging discipline.** Decide on a schema upfront and stick to it: `train/loss`, `val/loss`, `val/acc`, `val/precision`. Don't have `loss` in one run and `training_loss` in another — the dashboard can't compare them.

**Logging frequency.** Train loss per batch is too noisy and expensive; per N batches (50–500) is fine. Val metrics per epoch (or every K steps). Gradient norms periodically — they're often the first sign of trouble.

**Tagging and grouping.** Add tags ("baseline", "ablation-dropout", "phase-2") to make runs filterable. Group runs by sweep ID so you can compare "all hyperparameter search trials from yesterday".

**Versioning artefacts.** Models, prediction dumps, and evaluation reports. W&B's Artifacts and MLflow's Model Registry both handle this — promote to "staging" then "production" with explicit tracking of which run produced each.

**The git + diff trick.** Save the commit hash AND the uncommitted diff (`git diff HEAD`). Now you can reproduce any run exactly, even if it was launched from a dirty working tree.

**Hydra + W&B.** Hydra parses the YAML config; pass the resulting dict directly to `wandb.init(config=...)`. Now every Hydra override automatically shows up as a column in the dashboard.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import hydra, wandb, subprocess
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Capture the git state for reproducibility
    git_sha  = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    git_diff = subprocess.check_output(["git", "diff", "HEAD"]).decode()

    run = wandb.init(
        project=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.get("tags", []),
        notes=f"git: {git_sha[:8]}",
    )
    if git_diff:
        # Save the uncommitted diff as an artefact
        with open("uncommitted.diff", "w") as f: f.write(git_diff)
        wandb.save("uncommitted.diff")

    train(cfg)
    run.finish()
```

</div>

<div class="level-next">
<span>Want offline runs, distributed logging, and team workflows?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">A complete run record</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{record} = (\text{code}, \text{config}, \text{data}, \text{env}, \text{seeds}, \text{metrics}, \text{artefacts}) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Code: git commit + uncommitted diff

</li>
<li markdown="1">

Env: lockfile or container hash

</li>
<li markdown="1">

Data: dataset hash / version pointer

</li>
<li markdown="1">

Seeds: random, numpy, torch, cuda

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{run record} \;=\; (\text{code},\; \text{config},\; \text{data},\; \text{environment},\; \text{seeds},\; \text{metrics},\; \text{artefacts}) $$</span>

**In words.** A complete record of a single training run is a tuple of seven things — drop any one and you lose reproducibility. **Code** is the exact source (git sha + uncommitted diff). **Config** is all hyperparameters. **Data** is which dataset version was used. **Environment** captures library versions (lockfile) or the container hash. **Seeds** are the random seeds for Python, NumPy, PyTorch, and CUDA — without these, "same code, same data" still gives different results. **Metrics** are the time series. **Artefacts** are the binary outputs (checkpoints, plots).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`code`git commit + any uncommitted diff

</li>
<li markdown="1">

`config`all hyperparameters and runtime settings

</li>
<li markdown="1">

`data`dataset version hash or pointer

</li>
<li markdown="1">

`environment`lockfile (uv.lock, poetry.lock) or container digest

</li>
<li markdown="1">

`seeds`random seeds for Python, NumPy, PyTorch, CUDA

</li>
<li markdown="1">

`metrics`scalar time series logged during the run

</li>
<li markdown="1">

`artefacts`saved checkpoints, prediction dumps, plots

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Offline runs.** When training on a cluster without internet, log to local files first (`WANDB_MODE=offline` or MLflow's local backend) and sync after. Same for W&B Service mode on isolated networks.

**Distributed logging.** Only rank 0 should log scalar metrics — otherwise every rank writes them and you get an N× inflated step counter. Rank 0 broadcasts the run ID; other ranks can save per-rank artefacts (gradient histograms) under separate keys.

**Team workflows.** Shared projects with role-based access (W&B Teams, MLflow's auth). Naming conventions (`project_phase_owner`). Tagging discipline (`baseline`, `candidate`, `production`). Reports/Notebooks for sharing findings.

**Cost tracking.** Modern dashboards can ingest GPU-hours and dollar cost per run. Useful for blameless retros and for spotting runs that are 80% of the budget for 5% of the gain.

**Integration with experiment platforms.** Optuna's trials can stream into W&B sweeps. Lightning's Trainer auto-logs to whichever tracker you set. AzureML, Vertex, SageMaker all bridge to common trackers.

**The "experiment journal" pattern.** Per project, maintain a short markdown "what I tried, what happened, why" file alongside the code. Tracking dashboards are noisy; a few hand-written sentences per branch is what you'll actually re-read.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import os, torch.distributed as dist
import wandb

def setup_logging(cfg):
    is_rank0 = dist.get_rank() == 0 if dist.is_initialized() else True
    if is_rank0:
        run = wandb.init(project=cfg.project, config=cfg)
        # Share the run ID with other ranks so they know who they are
        os.environ["WANDB_RUN_ID"] = run.id
    return is_rank0

def log_metrics(metrics, step, is_rank0):
    if not is_rank0: return
    wandb.log(metrics, step=step)

# Offline-then-sync workflow for clusters without internet
# Run: WANDB_MODE=offline python train.py
# Later: wandb sync wandb/offline-run-...
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

[Weights & Biases Docs <i class="fas fa-external-link-alt"></i>](https://docs.wandb.ai/){: target="_blank" }
<span class="annotation">The reference for W&amp;B. Quickstarts for every framework; depth on sweeps, reports, and team workflows.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[MLflow Documentation <i class="fas fa-external-link-alt"></i>](https://mlflow.org/docs/latest/index.html){: target="_blank" }
<span class="annotation">Open-source alternative if you need self-hosted or want a model registry in the same tool.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Aim — Open-source Experiment Tracker <i class="fas fa-external-link-alt"></i>](https://github.com/aimhubio/aim){: target="_blank" }
<span class="annotation">Modern, fast, open-source. Worth a look if W&amp;B is too heavyweight and MLflow's UI bothers you.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Sacred <i class="fas fa-external-link-alt"></i>](https://github.com/IDSIA/sacred){: target="_blank" }
<span class="annotation">The original Python experiment manager. Still has a clean philosophy and a loyal user base.</span>

</li>
</ul>

</div>

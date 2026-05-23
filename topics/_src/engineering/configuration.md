---
title: Configuration Management — ML Resources Hub
eyebrow_text: ← Engineering · Project Structure
eyebrow_href: {{root}}engineering.html
heading: Configuration Management
lead: Hydra, YAML, dotenv — keep hyperparameters and paths out of your code.
active_nav: engineering
prev_href: repo-layout.html
prev_title: Repository Layout
next_href: reproducibility.html
next_title: Reproducibility
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Never hardcode hyperparameters.** Configs go in YAML files (or similar). Your code reads them at runtime. Two reasons: you can run the same code with different settings, and the config file is a self-documenting record of what produced each result.

</div>

<article class="tldr-body" markdown="1">

The anti-pattern: `learning_rate = 1e-3` at the top of `train.py`. To try a different value you either edit the file (and lose track of what you changed) or pass an argparse flag (which doesn't scale beyond ~5 parameters).

The pattern: put everything tweakable in a YAML file. Your code reads it. The file is checked into git or written into the run's output directory, so every experiment has a complete record of what produced it.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```yaml
# configs/exp01.yaml
seed: 42
model:
  name: resnet18
  num_classes: 10
data:
  name: cifar10
  batch_size: 128
  num_workers: 4
optimizer:
  name: adam
  lr: 1e-3
  weight_decay: 1e-4
training:
  epochs: 50
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Use configs when

- You'll run more than one experiment with the same code
- You want a permanent record of "what produced this result"
- You want to share an experiment definition with a collaborator
- You'll need to sweep over hyperparameters

</div>

<div class="no" markdown="1">

### Skip when

- Truly one-off script with three parameters — argparse is fine
- You're prototyping in a notebook
- The config is bigger than the code (you've over-configured)

</div>

</div>

<div class="level-next">
<span>Want to see Hydra and config composition?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

Use **Hydra** (or **OmegaConf**) to compose configs from reusable fragments and override them from the CLI. Secrets and environment-specific paths go in `.env` files, never in YAML.

</div>

<article class="tldr-body" markdown="1">

**Hydra** is the de-facto standard in research ML. It lets you split your config across multiple files — base + model + data + experiment — and compose them at run time. You can also override any value from the command line.

Suppose `configs/base.yaml` has defaults, `configs/model/resnet18.yaml` has model-specific settings, and `configs/experiment/exp01.yaml` overrides for a specific experiment. You launch as:

`python train.py +experiment=exp01 model=resnet50 optimizer.lr=3e-4`

Hydra merges everything, writes the final config to the run's output directory, and your script gets a single `cfg` object.

**Secrets and machine-specific paths** don't belong in YAML. Use a `.env` file (gitignored) and read environment variables. The YAML config can reference them: `data_dir: ${oc.env:DATA_DIR}`.

**Validate configs.** Define a [structured config](https://hydra.cc/docs/tutorials/structured_config/intro/){: target="_blank" } using dataclasses or Pydantic. Catch typos at load time, not after a 3-hour training run.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))         # for the run log

    model = build_model(cfg.model)
    data  = build_data(cfg.data)
    opt   = build_optimizer(model, cfg.optimizer)

    for epoch in range(cfg.training.epochs):
        ...

if __name__ == "__main__":
    train()
```

</div>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Reach for it when

- You'll run dozens or hundreds of experiments
- Multiple people share a codebase and want isolation
- You need to sweep hyperparameters (Hydra + Optuna sweeper)
- You want each run to write its config alongside outputs

</div>

<div class="no" markdown="1">

### Skip it when

- You don't want a config framework's learning curve
- Plain Python dataclasses + a typed factory function suffice
- Static analysis matters more than runtime composition
- You're integrating with a framework that already configures itself

</div>

</div>

<div class="level-next">
<span>Want structured configs, Pydantic, and the alternative libraries?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Treat configs as typed objects, not dicts.** Use dataclasses, Pydantic, or Hydra structured configs to validate the schema at load time. The config becomes part of your code's public interface — refactor it deliberately.

</div>

<article class="tldr-body" markdown="1">

**Hydra structured configs.** Define a dataclass schema for each part of your config, register it with Hydra's `ConfigStore`, and Hydra will validate that YAML files match the schema. Typos in field names fail at load time, not after a 6-hour run. IDEs autocomplete config access.

**Pydantic alternative.** Stronger validation (types, value constraints, custom validators), better error messages, JSON-schema export. Slightly more code but more powerful. Works particularly well when configs come from user input or APIs.

**OmegaConf interpolation.** Reference one config value from another, pull from environment variables, evaluate expressions. `experiment_name: ${model.name}-${data.name}`. Use sparingly — too much interpolation makes configs unreadable.

**Config inheritance and composition.** Hydra's "defaults list" composes fragments via a list of references. Override at any level by specifying a different fragment. Powerful but learnable in stages — start with a single config, add composition as needs grow.

**Anti-pattern: dynamic configs.** Computing config values in code based on other config values is fragile. If `batch_size` depends on number of GPUs, set both explicitly per experiment rather than calculating one from the other at runtime.

**Alternative libraries.** Hydra is the default but worth knowing about: [gin](https://github.com/google/gin-config){: target="_blank" } (decorator-based, less ceremony, popular at Google research), [Lightning's CLI](https://github.com/PyTorchLightning/pytorch-lightning){: target="_blank" } (jsonargparse-based, integrates with Lightning trainers), plain dataclasses + tyro / argparse (no extra dependency).

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from hydra import initialize, compose
from omegaconf import DictConfig

@dataclass
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = True

@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 50
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=TrainConfig)

# Load & validate against the dataclass schema
with initialize(version_base=None, config_path="../configs"):
    cfg: TrainConfig = compose(config_name="base", overrides=["optimizer.lr=3e-4"])
    # Typos in YAML now fail here, not 3 hours into training
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

[Hydra documentation <i class="fas fa-external-link-alt"></i>](https://hydra.cc/docs/intro/){: target="_blank" }
<span class="annotation">The official docs. Read the tutorials in order — they're short, well-paced, and cover composition, multirun, and sweepers.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[OmegaConf <i class="fas fa-external-link-alt"></i>](https://omegaconf.readthedocs.io/){: target="_blank" }
<span class="annotation">The library Hydra wraps. Useful when you want config features without Hydra's CLI framework.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Pydantic <i class="fas fa-external-link-alt"></i>](https://docs.pydantic.dev/){: target="_blank" }
<span class="annotation">The modern Python validation library. Use for any config that crosses a trust boundary — user input, API params, etc.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[gin-config <i class="fas fa-external-link-alt"></i>](https://github.com/google/gin-config){: target="_blank" }
<span class="annotation">Google's lighter alternative. Decorator-based, less ceremony than Hydra, good for smaller projects.</span>

</li>
</ul>

</div>

---
title: CLI Patterns — ML Resources Hub
eyebrow_text: ← Engineering · Time-Savers
eyebrow_href: {{root}}engineering.html
heading: CLI Patterns
lead: argparse, click, typer, fire — and how to wire a config file into the command line without losing your mind.
active_nav: engineering
prev_href: training-scaffolding.html
prev_title: Training Scaffolding
next_href: useful-libraries.html
next_title: Useful Libraries
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Every script you'll run more than twice deserves a CLI.** A clean command line + a config file gives you reproducibility, schedulability, and shareability for free. Modern Python has good tools (Typer, Hydra) that make this nearly trivial.

</div>

<article class="tldr-body" markdown="1">

**Three reasonable choices.** `typer`: type-hint-driven, modern, great defaults. `click`: classic, mature, used everywhere. `argparse`: stdlib; fine for tiny scripts. `fire`: zero-config; useful for prototypes. Skip `sys.argv` parsing; you'll regret it.

**The pattern.** One CLI per script. Each command takes a config (YAML / Hydra). Hyperparameters in the config; flags for things that vary per-run (output dir, debug mode). The CLI just dispatches; the work is in modules.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### What goes where

- **CLI flags**: run-specific (out dir, debug, dry-run)
- **Config file**: hyperparameters, paths, model architecture
- **Env vars**: secrets, API keys, runtime config
- **Code**: never put paths or hyperparameters here

</div>

<div class="no" markdown="1">

### Common mistakes

- 15 positional arguments — make them named
- Hard-coded paths in the CLI defaults
- One giant script with subcommands for unrelated things
- No `--help` text — users (you, in 3 weeks) will hate you

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import typer
from pathlib import Path
import yaml

app = typer.Typer(no_args_is_help=True)

@app.command()
def train(
    config: Path = typer.Argument(..., help="Path to YAML config"),
    out:    Path = typer.Option("runs/", help="Output directory"),
    debug:  bool = typer.Option(False, help="Quick smoke run"),
    seed:   int  = typer.Option(0,    help="Random seed"),
):
    """Train a model from a YAML config."""
    cfg = yaml.safe_load(config.read_text())
    if debug:
        cfg["max_steps"] = 10
    run_training(cfg, out=out, seed=seed)

@app.command()
def evaluate(
    checkpoint: Path,
    test_data:  Path,
    threshold:  float = 0.5,
):
    """Evaluate a checkpoint on a held-out test set."""
    ...

if __name__ == "__main__":
    app()
```

</div>

<div class="level-next">
<span>Want Hydra, multi-command apps, & clean help-text patterns?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Config + CLI interaction</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{values} \;=\; \text{defaults} \;\triangleleft\; \text{config file} \;\triangleleft\; \text{env vars} \;\triangleleft\; \text{CLI flags} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

*◁* = "overridden by"

</li>
<li markdown="1">

Each layer wins over the previous

</li>
<li markdown="1">

Standard precedence; matches what users expect

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{final values} \;=\; \text{defaults} \;\;\text{overridden by}\;\; \text{config} \;\;\text{overridden by}\;\; \text{env vars} \;\;\text{overridden by}\;\; \text{CLI flags} $$</span>

**In words.** Configuration values come from multiple places, and you need a clear rule for who wins. The standard chain — read left to right, with each later source *overriding* any earlier one. So baked-in defaults are the weakest; the config file overrides those; environment variables override the config; and CLI flags override everything. This matches the principle "the more local / explicit the source, the higher its priority". The `◁` symbol in the math version is just shorthand for "overridden by".
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`defaults`fallback values written into the code

</li>
<li markdown="1">

`config file`YAML / TOML loaded at startup

</li>
<li markdown="1">

`env vars`shell environment variables (good for secrets and CI)

</li>
<li markdown="1">

`CLI flags`command-line arguments — highest priority

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Hydra.** Facebook's config framework. YAML configs, override from CLI (`python train.py model.lr=1e-3`), composable configs (`defaults: [model: resnet, data: cifar10]`), multi-runs / sweeps. Most ML production projects converge on Hydra.

**Pydantic + CLI.** Define configs as Pydantic models — get validation, type coercion, defaults. `typer` integrates well. Useful for strict schemas; pairs nicely with Pydantic-everywhere codebases.

**Subcommands.** `my-tool train ...`, `my-tool evaluate ...`, `my-tool deploy ...`. Typer's decorator pattern. Better than one huge script with a `--mode` flag.

**Help is documentation.** Every flag gets a one-line description. `--help` output is what you'll read in 3 weeks; make it good. Examples in the docstring are nicer than the user manual.

**Dry-run flag.** `--dry-run` prints what would happen without doing it. Useful for destructive operations (training that overwrites, deployments, data writes).

**Config logging.** The script writes the fully-resolved config to the run's output directory. Every flag override, every default, every env var — recorded. Reproducibility starts here.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Hydra automatically creates an output dir and writes cfg.yaml there
    print(OmegaConf.to_yaml(cfg))                 # print the resolved config
    run_training(cfg)

if __name__ == "__main__":
    main()

# Usage:
# python train.py                                 # defaults
# python train.py model.lr=1e-2 data=cifar100      # override individual values
# python train.py --multirun model.lr=1e-2,1e-3,1e-4  # sweep
```

</div>

<div class="level-next">
<span>Want completion, plugins, & long-running daemon-style CLIs?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">The CLI contract</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{stdin} \to \text{exit code, stdout, stderr} $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Exit code 0 on success, non-zero on failure (matters for CI / shells)

</li>
<li markdown="1">

stdout for data / results; stderr for logs / progress

</li>
<li markdown="1">

Pipeable, scriptable, testable

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{input stream} \;\to\; (\text{exit code},\; \text{output stream},\; \text{error stream}) $$</span>

**In words.** The Unix contract every well-behaved CLI honours. You read from `stdin` (standard input) and produce three outputs: a numeric **exit code** (0 means success, anything else means failure — shells and CI key off this), `stdout` (standard output — where actual results go), and `stderr` (standard error — where logs, warnings, and progress bars go). Keeping logs out of stdout means your output can be piped into the next command without contamination. This separation is what makes CLI tools composable.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`exit code`integer status; 0 = success, non-zero = failure

</li>
<li markdown="1">

`stdout`the program's "real" output — pipe target

</li>
<li markdown="1">

`stderr`diagnostic messages, separate from stdout

</li>
<li markdown="1">

`stdin`input stream (file, pipe, keyboard)

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Shell completion.** Typer and Click both auto-generate completion scripts for bash / zsh / fish. `my-tool --install-completion`. Cheap UX win; pays dividends every day.

**Plugins.** Click and Typer both support plugin loading from entry-points. Useful for very large CLIs (a "platform" CLI with sub-tools). Most ML projects don't need this — but it's there when you do.

**Long-running CLIs.** A train command might run for days. Print structured logs to stderr, write metrics to a logger / file, write checkpoints. Support graceful shutdown on Ctrl+C — save state and exit cleanly.

**Daemon mode.** Some CLIs spawn long-lived services (serving, monitoring). systemd unit files, Docker containers, or supervisord. The CLI itself should fork-and-detach cleanly or run in the foreground for the supervisor.

**Testable CLIs.** Click and Typer both ship `CliRunner` for invoking commands programmatically. Asserts exit code + stdout. Same as testing any function.

**Environment variable conventions.** Pydantic Settings + dotenv for secrets. Prefix env vars (`MYAPP_LOG_LEVEL`) to avoid collision. Document them.

**Versioning.** Every CLI has `--version`. Helps debugging "which version is the CI runner using" mysteries.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import typer
from typer.testing import CliRunner

app = typer.Typer()

@app.command()
def add(a: int, b: int):
    """Add two numbers."""
    typer.echo(a + b)

# Test the CLI as a unit
def test_add():
    runner = CliRunner()
    result = runner.invoke(app, ["3", "4"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "7"
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

[Typer Documentation <i class="fas fa-external-link-alt"></i>](https://typer.tiangolo.com/){: target="_blank" }
<span class="annotation">Modern type-hint-driven CLI library. The minimum-friction way to add a CLI to a Python project today.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Click Documentation <i class="fas fa-external-link-alt"></i>](https://click.palletsprojects.com/){: target="_blank" }
<span class="annotation">Mature, classic, used everywhere. More boilerplate than Typer; more flexible for complex command hierarchies.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Hydra Documentation <i class="fas fa-external-link-alt"></i>](https://hydra.cc/){: target="_blank" }
<span class="annotation">Facebook's config framework. The right answer when you have many configurations to compose and override.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Command Line Interface Guidelines <i class="fas fa-external-link-alt"></i>](https://clig.dev/){: target="_blank" }
<span class="annotation">Opinionated guide to good CLI design. Not ML-specific; very applicable.</span>

</li>
</ul>

</div>

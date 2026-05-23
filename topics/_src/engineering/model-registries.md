---
title: Model Registries — ML Resources Hub
eyebrow_text: ← Engineering · CI / CD
eyebrow_href: {{root}}engineering.html
heading: Model Registries
lead: A central catalogue of every model version — with metadata, lineage, and clear stages.
active_nav: engineering
prev_href: automated-retraining.html
prev_title: Automated Retraining
next_href: training-debugging.html
next_title: Training Debugging
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**One source of truth for every model artefact.** Each trained model gets a name, a version, metadata (metrics, config, lineage), and a stage (None → Staging → Production → Archived). Deployment systems pull "the model tagged production" by name, never by file path.

</div>

<article class="tldr-body" markdown="1">

Why this matters: in production you want to upgrade or roll back a model without redeploying the serving infrastructure. The registry is the indirection — the serving code asks "give me the current production model"; the registry returns the right artefact. Versioning, audit trails, and rollback are all properties of the registry.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### What the registry stores

- The model artefact (weights / pickle / ONNX)
- Hyperparameters and the training config
- Training metrics and final eval scores
- Dataset version / hash
- Lineage: parent model, training run, code commit
- Stage: None / Staging / Production / Archived

</div>

<div class="no" markdown="1">

### Tools

- **MLflow Model Registry**: open-source, mature
- **W&B Artifacts**: hosted, same-tool as tracking
- **SageMaker Model Registry**: AWS-native
- **Vertex Model Registry**: GCP-native
- **BentoML Yatai**: model + bento packaging

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import mlflow

# At the end of training, log the model and metadata
with mlflow.start_run() as run:
    mlflow.log_params(cfg.flatten())
    mlflow.log_metrics({"val/auc": 0.91, "test/auc": 0.89})
    mlflow.pytorch.log_model(model, "model")

# Register as a versioned entity
result = mlflow.register_model(
    f"runs:/{run.info.run_id}/model",
    name="credit-risk-classifier",
)
print(f"Registered version {result.version}")

# Promote to production after promotion gate passes
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="credit-risk-classifier", version=result.version, stage="Production",
)

# At serve time
model = mlflow.pytorch.load_model("models:/credit-risk-classifier/Production")
```

</div>

<div class="level-next">
<span>Want lineage tracking, model cards, & multi-environment promotion?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Model versioning identity</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ \text{model version} \;=\; \text{hash}(\text{code commit}, \text{data hash}, \text{config}, \text{seeds}) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Reproducibility = re-deriving the same model from these inputs

</li>
<li markdown="1">

The registry stores the artefact and the inputs separately

</li>
<li markdown="1">

Audit trail: who created it, when, with what data, why promoted

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{model version id} \;=\; \text{fingerprint of (code, data, config, seeds)} $$</span>

**In words.** A model version is uniquely identified by everything that went into producing it: the `code commit` (a specific git SHA), the `data hash` (a fingerprint of the training data), the `config` (hyperparameters, paths), and the `seeds` (random number generators). `hash(...)` just means "smush all of these inputs into one short fingerprint string". Two runs with identical inputs should produce the same version; if anything differs, the version differs.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`code commit`git SHA at training time

</li>
<li markdown="1">

`data hash`fingerprint of the dataset used

</li>
<li markdown="1">

`config`hyperparameters & settings

</li>
<li markdown="1">

`seeds`random number generator seeds for reproducibility

</li>
<li markdown="1">

Audit trail: who created it, when, with what data, why promoted

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Stages.** "None" (just registered), "Staging" (under evaluation), "Production" (serving live), "Archived" (kept for audit). The transitions need explicit gates — usually CI green + eval pass for Staging, human review for Production.

**Aliases vs stages.** MLflow has both. Stages are bucket-based; aliases are pointer-based ("@latest", "@champion", "@candidate"). Modern MLflow favours aliases; flexible enough for non-linear workflows.

**Multi-environment promotion.** Dev → staging → production. Each environment's serving system points at its registry stage. Promotion is a registry operation, not a redeploy.

**Lineage.** Every model points to its training run, which points to its config, dataset version, and parent model (if fine-tuned). A few hops takes you from a production prediction back to the raw data that produced it.

**Model cards.** Mitchell et al. (2019). Standardised model documentation — intended use, training data, performance metrics across subgroups, ethical considerations. Increasingly a regulatory requirement; HuggingFace, OpenAI, and Anthropic all ship them.

**Artefact storage.** The registry tracks; the storage holds the weights. S3 / GCS / Azure Blob / on-prem. Use registries that decouple metadata storage from artefact storage so you can swap backends without losing history.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Modern MLflow style — use aliases for routing
client.set_registered_model_alias(
    name="credit-risk-classifier", alias="production", version=42,
)
client.set_registered_model_alias(
    name="credit-risk-classifier", alias="champion", version=42,
)
client.set_registered_model_alias(
    name="credit-risk-classifier", alias="challenger", version=43,
)

# Serving code is decoupled from versions
prod = mlflow.pytorch.load_model("models:/credit-risk-classifier@production")

# Roll back: just re-point the alias, no redeploy
client.set_registered_model_alias(
    name="credit-risk-classifier", alias="production", version=41,
)
```

</div>

<div class="level-next">
<span>Want model cards, governance, & supply-chain attestation?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Supply chain for models</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ (\text{base model}, \text{dataset}, \text{code}, \text{config}) \xrightarrow{\text{training}} (\text{artefact}, \text{provenance}, \text{signature}) $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

Each input is itself versioned and signed

</li>
<li markdown="1">

Output artefact carries a provenance record traceable to all inputs

</li>
<li markdown="1">

SLSA-like attestation for ML — not yet standard but converging

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ (\text{inputs: base model, dataset, code, config}) \;\to\; (\text{outputs: artefact, provenance, signature}) $$</span>

**In words.** A training run is a function that takes versioned inputs and produces both the trained model artefact and a record of where it came from. The arrow (`→`) is just "produces"; "training" sits above it to label what's happening. The `provenance` is a paper trail back to every input; the `signature` is a cryptographic seal so consumers can verify the artefact hasn't been tampered with. If you change *any* input, you should get a different, traceable output.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`inputs`each one separately versioned and signed

</li>
<li markdown="1">

`artefact`the trained weights / model file itself

</li>
<li markdown="1">

`provenance`traceability record back to every input

</li>
<li markdown="1">

`signature`cryptographic seal proving the artefact is authentic

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Model cards in the registry.** Auto-generate from the training run: dataset, training metrics, subgroup metrics, intended use, known limitations. Mitchell et al.'s template is the reference; HuggingFace has built-in support.

**Governance & audit.** Who can promote? Who reviewed? When? Most enterprise registries (MLflow Enterprise, AzureML, Vertex) have role-based access + audit logs. For high-stakes domains (medical, financial), this is a regulatory requirement.

**Model signing & attestation.** Cryptographically sign artefacts so consumers can verify provenance. Sigstore for ML. SBOM (software bill of materials) extending to model artefacts. Active area in 2024+ as supply-chain attacks become a concern.

**Lineage propagation.** Fine-tuned model → base model → pre-training data. Each step links to the previous; a prediction can be traced back through fine-tuning to pre-training data. Useful for debugging and compliance.

**Registry-driven CI.** CI workflows triggered by registry events: new "staging" model → run extended evals; new "production" model → deploy + announce. Registries with webhooks (MLflow Enterprise, W&B) make this natural.

**Distillation lineage.** A distilled student model points to its teacher. Compound chains (teacher → student → re-distilled student) need careful tracking; otherwise debugging "where did this behaviour come from" is impossible.

**Multi-model systems.** When a production prediction depends on several models (retrieval encoder + ranker + reranker + LLM), the registry needs to support "deployments" that bundle versions across multiple model names.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import mlflow
from huggingface_hub import ModelCard, ModelCardData

# Auto-generate a model card from the training run
def make_card(run_id, model_name, version):
    run = mlflow.get_run(run_id)
    card_data = ModelCardData(
        language="en",
        license="apache-2.0",
        model_name=model_name,
        tags=["classification", "credit-risk"],
        metrics={
            "val_auc": run.data.metrics["val/auc"],
            "test_auc": run.data.metrics["test/auc"],
        },
    )
    card = ModelCard.from_template(card_data, template_path="MODEL_CARD_TEMPLATE.md")
    card.save(f"cards/{model_name}-{version}.md")
    return card
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

[MLflow Model Registry Docs <i class="fas fa-external-link-alt"></i>](https://mlflow.org/docs/latest/model-registry.html){: target="_blank" }
<span class="annotation">The most-used open-source registry. Stages, aliases, webhooks, governance.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[W&B Artifacts <i class="fas fa-external-link-alt"></i>](https://docs.wandb.ai/guides/artifacts){: target="_blank" }
<span class="annotation">W&amp;B's take on artefact + model versioning. Tighter integration with W&amp;B tracking; less full-featured as a standalone registry.</span>

</li>
<li data-tier="indepth" markdown="1">

[Mitchell et al. (2019) — Model Cards <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/1810.03993){: target="_blank" }
<span class="annotation">The model card paper. Standardised documentation now widely adopted across the industry.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[HuggingFace — Model Cards <i class="fas fa-external-link-alt"></i>](https://huggingface.co/docs/hub/model-cards){: target="_blank" }
<span class="annotation">Practical template + library for producing model cards. The largest registry of public model cards.</span>

</li>
</ul>

</div>

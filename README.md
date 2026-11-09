# SELAUR-G7

Research fork of **SELAUR: Self-Evolving LLM Agent via Uncertainty-aware
Rewards** for exploring uncertainty-aware reward design in LLM agents.

This repository is maintained by Group G7. It preserves the original SELAUR
implementation as the experimental foundation while allowing multiple research
directions to be developed and evaluated independently.

> **Current implementation:** Action-SELAUR token scoping is implemented on
> this branch. Full-response SELAUR remains the default for backward
> compatibility.

## Foundation: SELAUR

SELAUR augments a critic-free, two-level group advantage with uncertainty-aware
reward shaping. For every agent step, it estimates uncertainty from generated
token probabilities using entropy contribution, least confidence, and the
top-1/top-2 probability margin.

The signals are combined, smoothed across the trajectory, and injected into
step- and episode-level group scores before advantage normalization. The code
supports WebShop, ALFWorld, and the other environments inherited from
`verl-agent`.

- Upstream: [SELAUR](https://github.com/adoptedirelia/SELAUR-Self-Evolving-LLM-Agent-via-Uncertainty-aware-Rewards)
- Paper: [SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards](https://arxiv.org/abs/2602.21158)

## Current Direction: Action-SELAUR

### Motivation

In WebShop and ALFWorld, the policy produces separate reasoning and action
sections:

```text
<think>
Reason about the observation and available actions.
</think>
<action>
click[item_123]
</action>
```

Only the content inside `<action>...</action>` is executed by the
environment. Original SELAUR, however, aggregates token uncertainty over the
complete generated response, including:

- reasoning tokens inside `<think>...</think>`;
- action tokens inside `<action>...</action>`;
- structural tags and formatting tokens.

Reasoning uncertainty and decision uncertainty need not agree. A policy may
express uncertain reasoning but still choose the correct action, or generate a
fluent and confident explanation followed by the wrong action. Full-response
averaging may therefore introduce response-length and reasoning-style
confounds into reward shaping.

### Implemented method

Original full-response uncertainty:

$$
U_{\text{full}} = \frac{1}{N}\sum_{i=1}^{N} u_i
$$

Action-only uncertainty:

$$
U_{\text{action}} = \frac{1}{|A|}\sum_{i \in A} u_i
$$

Here, \(A\) is the token span corresponding to the action actually passed to
the environment.

Action-SELAUR changes only the token mask used by the uncertainty estimator.
The following components remain fixed:

- entropy, least-confidence, and top-2 margin signals;
- signal-combination weights;
- trajectory smoothing;
- failure-aware reward transformation;
- success-rate gate;
- model, rollout, optimization, and evaluation settings.

This produces a controlled comparison in which the only intended independent
variable is **where uncertainty is measured**.

### Action-span definition

The action span must match the environment projection logic:

1. find the first valid `<action>` opening tag;
2. find its corresponding `</action>` closing tag;
3. select only generated tokens inside the tags;
4. exclude the tags themselves from action uncertainty;
5. record whether a valid action mask was found.

Malformed responses require a fallback consistent with the action executed by
the environment. Its rate must be logged so invalid formats cannot silently
bias action uncertainty.

### Configuration

Set the uncertainty token scope through Hydra:

```bash
selaur.token_scope=full        # original SELAUR (default)
selaur.token_scope=action      # tokens overlapping action content
selaur.token_scope=non_action  # tokens outside the tagged action block
```

Malformed or empty action blocks fall back to the full response.

## Research Hypotheses

- **H1 — Length confounding:** full-response uncertainty is more strongly
  correlated with response length and reasoning style than action uncertainty.
- **H2 — Diagnostic quality:** action uncertainty predicts invalid actions and
  episode failure better than full-response uncertainty.
- **H3 — Learning efficiency:** when uncertainty shaping is active,
  Action-SELAUR improves early-training sample efficiency relative to SELAUR.

Separating an action into operator and argument tokens, such as `click`
versus `[item_123]`, is reserved for later work.

## Minimal Experimental Plan

### Stage 1: diagnostic study

Compute three measurements from identical collected rollouts:

| Condition | Token mask |
| --- | --- |
| SELAUR | Complete generated response |
| Action-SELAUR | Content inside `<action>...</action>` |
| Non-action control | Response tokens outside the action span |

Primary diagnostic metrics:

- Spearman correlation between uncertainty and response length;
- AUROC/AUPRC for invalid-action prediction;
- AUROC/AUPRC for episode-failure prediction;
- uncertainty distributions for successful and failed trajectories;
- action-token fraction and action-mask fallback rate;
- uncertainty differences between correct and incorrect actions from the same
  anchor state.

Training should proceed only if action uncertainty provides a meaningful
diagnostic improvement over both full-response uncertainty and the non-action
control.

### Stage 2: training comparison

The minimum training comparison contains:

1. SELAUR with full-response uncertainty;
2. Action-SELAUR with action-only uncertainty.

The first pass uses one seed and a shortened horizon. Additional seeds and
full-length runs are justified only after a clear signal appears.

Primary training metrics:

- learning-curve area under the curve;
- success rate and environment task score;
- invalid-action ratio;
- uncertainty-shaping activation rate;
- policy KL, gradient norm, and training stability;
- final performance under the same training budget.

The success-rate gate must be logged explicitly. If it never activates, the
training comparison cannot test the effect of changing the uncertainty mask.

## Implementation Surface

| Module | Responsibility |
| --- | --- |
| `agent_system/multi_turn_rollout/rollout_loop.py` | Build and apply the configured mask while response IDs and rollout log-probabilities remain aligned. |
| `selaur/uncertainty/masking.py` | Map generated token IDs to full, action, or non-action masks without re-tokenizing. |
| `selaur/config.py` | Validate the `token_scope` setting. |
| `tests/selaur/` | Test span extraction, fallback behavior, masking, and backward compatibility. |

The rollout collector constructs the mask before storing top-1/top-2 log-probabilities.

## Repository Layout

```text
selaur/             uncertainty signals, transforms, and advantages
agent_system/       environments and multi-turn rollout collection
verl/               RL training engine
examples/           SELAUR and baseline launch scripts
tests/selaur/       algorithm unit tests
docs/               inherited documentation
```

## Installation

Base environment:

```bash
conda create -n selaur python=3.12 -y
conda activate selaur
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
pip install vllm==0.8.5
python -c "import verl, selaur, vllm; print('ok')"
```

WebShop requires a separate Python 3.10 environment because of dependency
constraints. ALFWorld can use the base SELAUR environment after installing its
environment package and assets. Keep datasets, model caches, checkpoints, and
experiment logs outside the Git repository.

## Running the Existing Baseline

WebShop:

```bash
conda activate selaur-webshop
bash examples/selaur_trainer/run_webshop.sh
```

ALFWorld:

```bash
conda activate selaur
bash examples/selaur_trainer/run_alfworld.sh
```

## Running Action-SELAUR

In `verl/trainer/config/ppo_trainer.yaml`, set the SELAUR token scope to
`action`:

```yaml
selaur:
  token_scope: action
```

Keep the remaining SELAUR and training settings unchanged, then run the same
environment-specific launcher.

WebShop:

```bash
conda activate selaur-webshop
bash examples/selaur_trainer/run_webshop.sh
```

ALFWorld:

```bash
conda activate selaur
bash examples/selaur_trainer/run_alfworld.sh
```

Set `token_scope: full` to reproduce the original SELAUR baseline. Use
`token_scope: non_action` for the reasoning-token negative control.

Unit tests:

```bash
pytest tests/selaur -q
```

## Git Workflow

- `origin` points to the private G7 repository.
- `upstream` points to the original public SELAUR repository.
- `main` preserves the shared integration baseline.
- Each researcher works on an independent branch or Git worktree.
- Changes are merged through reviewed pull requests.
- Data, checkpoints, local logs, credentials, and agent coordination files are
  never committed.

## Attribution and License

This project is derived from the official SELAUR implementation and retains its
Apache License 2.0 terms, copyright notices, and attribution. See
[`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt).

If this repository or its upstream implementation is used in research, cite:

```bibtex
@article{zhang2026selaur,
  title   = {SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards},
  author  = {Zhang, Dengjia and Liu, Xiaoou and Cheng, Lu and Wang, Yaqing and
             Murray, Kenton and Wei, Hua},
  journal = {arXiv preprint arXiv:2602.21158},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.21158}
}
```

# Treasure continual Fleet, 5-shot with AIGIBench replay

This independent entry point leaves the existing pretraining, few-shot baseline,
`Run.sh`, and Table 1 scripts unchanged. It carries one adapted Fleet model through
64 generators and evaluates every domain at states 0 through 64. The complete run
is intentionally **not** launched during implementation.

## Protocol and resolved Fleet conventions

`RunTable1.sh` supplies `--n_fake_support "$shot" --n_real_support "$shot"` to
`fewshot_experiment.py`: **5-shot means five fake plus five real images**, ten
support images per generator. Real support comes from cc12m, with Fleet's
filename filter when available. Sampling uses sorted canonical paths and an
independent seed derived from the experiment seed and exact generator name.
All 64 support sets are fixed before initial evaluation, even in a smoke run.
Their union is excluded from every query and retention set, including real
support for future steps. Symlinks and hardlinks are recognized. These checks
identify files, not re-encoded or separately copied near-duplicate images.

Support sets differ from legacy runs that sampled unsorted filesystem listings;
use the saved manifests for comparisons between continual methods. No Treasure
support or query images enter AIGIBench replay. Real query images are a shared,
fixed Treasure real set, evaluated once per state and reused in each binary
real-versus-generator task. Each generator has its own fixed fake query set.
Generators' names never enter model inference.

The local Fleet evaluator does **not** output probabilities or use a 0.5
threshold: it computes `confidence = cos(real prototype) - cos(fake prototype)`
and labels `confidence > 0` real, otherwise fake. The new experiment reuses that
exact evaluator and adds the explicit uncalibrated adapter
`p(fake) = sigmoid(-confidence)`. Thus **p(fake) >= 0.5 is fake**, including ties,
and decisions and rankings agree with Fleet. This is a documented adapter,
not an existing Fleet probability convention. There is no fitted temperature,
calibration, or threshold search. `decision_threshold: 0.5` is required and checked.
All rates, AUROC and AP use fractions, not percentages; improved/declined
percentage summaries are explicitly percentages. ECE (15 equal-width bins) and
binary log loss describe these uncalibrated adapter probabilities, not Fleet's
contrastive training loss. TPR-at-FPR reports the best empirical ROC point at or
below the FPR cap, without interpolation.

The default replay pool has 1,000 AIGIBench training images: 500 fake, 500 real,
sampled uniformly without replacement within each class. `fixed` reuses the
sample set; `resample_per_step` uses `seed + step`. Each training update combines
all ten support images with up to 32 replay images (ratio 3.2). A final short
replay batch has a smaller ratio. Replay reports include unique examples used
per step, cumulative per-step counts (including repeat use across steps), and
training presentations across epochs. Replay candidates exclude evaluation files
and all Treasure supports. Treasure replay is disabled; enabling the reserved
option explicitly fails rather than silently changing the experiment.

The original AIGIBench-pretrained model supplies distillation targets and routing
anchors. Prototypes remain fixed. Fleet's `train_epoch`, `compute_anchors`,
contrastive, mutual-avoidance and distillation losses are reused. The defaults
retain the legacy 20 epochs, AdamW lr=3e-5, weight decay=.01, avoidance weight=20,
distillation weight=10, contrastive weight=0 for epochs 1–5 and 5 thereafter.
Replay center crops are deterministic to align cached teacher features; support
uses Fleet's random training crops. AdamW resets at each generator. There is no
scheduler. Disabling replay is an explicitly different contrastive-only support
ablation (no distillation/avoidance). All generators are adapted, including ones
with high initial accuracy; the baseline's optional accuracy-based skip is not
used in this continual protocol.

## Chronology and limits

`treasure_generator_order.csv` contains 64 exact directory names, display names,
release dates, available family labels, sources and uncertainty notes.
The local `data/Treasure/dataset_index.json`, metadata and Fleet paper
[Table 8 / Appendix A.4](https://arxiv.org/html/2606.31082v1#A1) were inspected;
they provide taxonomy and sequence experiments but no 64-generator chronological
order or release-date table. This file is **manually curated**, using official
publisher announcements/repositories and primary papers. For historical research
models, first public paper dates are explicitly marked proxies for a release;
these are not claimed to be weight-publication dates. Repository news dates,
regional API launches, and version-specific ambiguities are annotated per row.

Dated entries sort ascending by ISO date. Month-only dates precede day-specific
dates in that month; this is an explicit convention, not a claim about which
came first within the month. Equal dates use exact directory name in Unicode
lexicographic order. **MAE, Wukong and ideogram have unknown dates**, appear last
in that documented order, and must not be interpreted as the newest models.
MAE's identity is ambiguous (MaskGit/MAE), Ideogram lacks a version, and a reliable
Wukong release date was not established. This is a reproducible partial chronology,
not a fully verified oldest-to-newest ordering of all 64 models. Replace uncertain
entries with authoritative evidence before making strict temporal claims. The
entire resolved order is printed, embedded in config, and copied as CSV and JSON.

## Evaluation and artifacts

State 0 is the pretrained model. State i follows adaptation i. At step i, its
pre-adaptation measurement is the existing state i-1, so there is no redundant
full evaluation. A normal completed run has exactly 65 x 64 entries per detection
metric. A two-generator smoke run has 3 x 2. Every state also evaluates a fixed
AIGIBench validation subset, by default up to 1,000 images per class.

Outputs live only under `All_outputs/<experiment_name>/`:

- `config_resolved.yaml`, `generator_order.csv`, `generator_order.json`.
- `support_manifests/<generator>.json` and `experiment_splits.json` (all supports,
  fixed queries/retention, and AIGIBench replay candidate paths).
- `replay_manifests/step_NNN.json` (paths, labels, class counts, policy and seed).
- `checkpoints/state_NNN.pth`, `latest.pth`, `best.pth`.
- `metrics/evaluation_<metric>.csv` for every detection metric, including sample
  counts, confusion counts, ECE and log loss; state/checkpoint/seed metadata.
- `metrics/metrics_long.csv`: one row per state/domain. `current_generator_phase`
  marks pre at state j-1 and post at state j; `associated_adaptation_step` identifies
  j while `adaptation_step` is the state's completed step. Domain relation is
  past/current/future relative to the adapted domain.
- `metrics/continual_metrics_by_step.csv`, `retention.csv`, `training.csv`,
  `runtime_and_replay.csv`, `completed_state.json`, `final_summary.json`.
- AUROC/balanced-accuracy heatmaps and AUROC per-generator forgetting/gain plots.
- `logs/`, `wandb/`, launcher PID/lock files.

Summaries for AUROC (primary), balanced accuracy, AP and accuracy implement the
requested formulas. Undefined step quantities are null/empty. Running forgetting
can be negative; final forgetting includes the final state in the maximum and
cannot be negative. Macro averages weight each generator equally. Worst/best
performance and largest gain/forgetting statistics are also included. Final
retention and its change from state 0 are in the summary.

## Checkpoints and resume

State 0, latest, best macro `best_metric` state, and requested periodic states
are retained. Latest/best are atomic hardlinks, avoiding redundant disk copies.
Best selection is for archival only; it never resets the continual model.
Checkpoints contain model, optimizer, prototypes, completed step, accumulated
metrics, replay-manifest hashes, split-manifest hash, input identities, and Python,
NumPy, CPU/CUDA RNG states. A completed checkpoint is the commit point; derivative
CSV/JSON files are rebuilt from it on resume. A partially completed adaptation
restarts from the preceding checkpoint. Resume does not restore the previous
step's optimizer because the documented protocol resets AdamW at each new domain.

Resume validates configuration, checkpoint/prototype/order SHA-256 identities,
and manifests. Dataset paths are checked again. Image bytes are not fingerprinted;
keep datasets immutable. Use a new experiment name for changed seeds or protocol.
An incomplete state-0 initialization is not overwritten silently. Direct invocations
and launchers both guard against concurrent writes to the same experiment.

## Running and monitoring

Use the existing AIGB environment. Additional local dependencies are PyYAML,
scikit-learn and matplotlib; W&B and pandas are optional for synchronization.
No external model downloads are needed with the provided local DINO directory.
Relative paths always resolve from the Fleet repository root.

```bash
cd /egr/research-sprintai/baliahsa/projects/Fleet
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=/egr/research-sprintai/baliahsa/miniconda3/envs/AIGB/bin/python
"$PYTHON" -m fleet.train.treasure_continual_5shot --validate-config
"$PYTHON" -m fleet.train.treasure_continual_5shot --validate-paths
"$PYTHON" -m fleet.train.treasure_continual_5shot --self-test
CUDA_VISIBLE_DEVICES=0 bash run_treasure_continual_5shot_background.sh \
  Configs/treasure_continual_fleet_5shot_aigibench_replay.yaml
ps -p "$(cat All_outputs/treasure_continual_fleet_5shot_aigibench_replay/experiment.pid)" -o pid,etime,cmd
tail -f All_outputs/treasure_continual_fleet_5shot_aigibench_replay/logs/continual_*.log
```

The launcher accepts `PYTHON`, `CUDA_VISIBLE_DEVICES`, and optional config path.
It uses nohup, timestamps logs, publishes PID, and refuses a live duplicate PID.
`FORCE=1` bypasses a stale PID check; it does not defeat the training output lock.
To run concurrent independent experiments, choose distinct experiment names.

W&B defaults to disabled. Set `wandb.mode` to online or offline as desired; keys
stay in the environment/existing W&B login. Config includes git dirty status,
input hashes, exact order, support/replay/threshold policies. `continual/step`
drives continual, retention, replay, training and runtime plots. Epoch losses,
current pre/post/gain, macro averages, BWT/FWT/forgetting, forward performance,
incremental averages, retention deltas, timings and GPU memory are logged. Final
summaries, matrix Tables, plots and CSV/JSON/manifests are uploaded as artifacts.
Remote initialization/logging/upload errors warn and preserve local outputs.
For offline resumes, W&B may create separate offline segments; local checkpoints
and metrics remain the authoritative complete trajectory.

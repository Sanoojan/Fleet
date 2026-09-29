#!/usr/bin/env python3
"""Continual Fleet adaptation; independent of the existing training entry points."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import time
import warnings

import numpy as np

from fleet.train.continual_metrics import (
    SUMMARY_METRICS, detection_metrics, fake_probability, final_summary, step_summary,
)
from fleet.train.treasure_continual_data import (
    ROOT, atomic_text, digest, load_config, load_order, prepare_data, replay_manifest,
    save_csv, save_json,
)


def git_info():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                status=git('status', '--porcelain'))


class Tracking:
    """Every remote operation is optional; local files remain authoritative."""
    def __init__(self, c):
        self.run = self.wandb = None
        if c['wandb']['mode'] == 'disabled':
            return
        try:
            # Keep artifacts/cache off quota-limited home directories.
            for variable, directory in (('WANDB_CACHE_DIR', 'cache'), ('WANDB_DATA_DIR', 'staging'),
                                        ('WANDB_ARTIFACT_DIR', 'artifacts')):
                path = Path(c['output_dir']) / 'wandb' / directory
                path.mkdir(parents=True, exist_ok=True)
                os.environ[variable] = str(path)
            import wandb
            self.wandb = wandb
            w = c['wandb']
            self.run = wandb.init(project=w['project'], entity=w['entity'], group=w['group'],
                                  tags=w['tags'], mode=w['mode'], name=w['run_name'] or c['experiment_name'],
                                  dir=c['output_dir'], config=c,
                                  id=c['run_id'], resume='allow')
            wandb.define_metric('continual/step')
            for prefix in ('continual/*', 'retention/*', 'train/*', 'runtime/*', 'replay/*'):
                wandb.define_metric(prefix, step_metric='continual/step')
        except Exception as exc:
            warnings.warn(f'W&B unavailable; keeping local outputs: {exc}')
            self.run = None

    def log(self, payload):
        if self.run:
            import torch
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            with torch.random.fork_rng():
                try:
                    self.run.log(payload)
                except Exception as exc:
                    warnings.warn(f'W&B logging failed: {exc}')
                finally:
                    random.setstate(python_rng)
                    np.random.set_state(numpy_rng)

    def finish(self, out, summaries, step):
        if not self.run:
            return
        try:
            import pandas as pd
            payload = {'continual/step': step}
            for metric in SUMMARY_METRICS:
                payload[f'matrices/{metric}'] = self.wandb.Table(dataframe=pd.read_csv(out / 'metrics' / f'evaluation_{metric}.csv'))
                for key, value in summaries[metric].items():
                    if isinstance(value, (int, float)):
                        self.run.summary[f'final/{key}_{metric}'] = value
            for path in (out / 'metrics').glob('*.png'):
                payload[f'plots/{path.stem}'] = self.wandb.Image(str(path))
            self.log(payload)
            artifact = self.wandb.Artifact(f'{out.name}-results', type='continual-results')
            artifact.add_dir(str(out / 'metrics'))
            for filename in ('config_resolved.yaml', 'generator_order.csv', 'generator_order.json'):
                artifact.add_file(str(out / filename))
            for directory in ('support_manifests', 'replay_manifests'):
                artifact.add_dir(str(out / directory))
            self.run.log_artifact(artifact)
        except Exception as exc:
            warnings.warn(f'W&B artifact upload failed; local outputs saved: {exc}')
        finally:
            try:
                self.run.finish()
            except Exception as exc:
                warnings.warn(f'W&B finish failed; local outputs saved: {exc}')


def create_model(c, checkpoint):
    from fleet.models.dual_branch_contrastive import DualBranchContrastiveModel
    from fleet.utils import infer_q_dim_and_spec_from_paths
    q, _ = infer_q_dim_and_spec_from_paths([c['pretrained_checkpoint'], c['prototype_path']])
    q = checkpoint.get('q_dim', q)
    if not q:
        raise ValueError('Cannot infer q_dim from checkpoint metadata or path')
    kwargs = {k: checkpoint.get(k, c['model'][k]) for k in c['model']}
    kwargs.update(q_dim=q, num_heads=1024 // q, dinov3_model_path=c['model_path'])
    model = DualBranchContrastiveModel(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model


def loader(c, paths, labels, processor, training=False, batch_size=None):
    from torch.utils.data import DataLoader
    from fleet.datasets.dual_branch_fewshot import DualBranchImageDataset
    dataset = DualBranchImageDataset(paths, labels, processor, c['xception_crop_size'], is_training=training)
    return DataLoader(dataset, batch_size=batch_size or c['evaluation_batch_size'], shuffle=False,
                      num_workers=c['num_workers'], pin_memory=True)


def adapt(c, model, teacher, processor, device, support, replay, tracking, state):
    import torch
    from fleet.train.fewshot_experiment import compute_anchors, train_epoch
    from fleet.utils import fewshot_contrastive_loss
    params = []
    for p in model.parameters():
        p.requires_grad_(False)
    for component in c['trainable_components']:
        if component == 'lora':
            for _, layer in model.lora_modules:
                params.extend([layer.lora_A, layer.lora_B])
        else:
            params.extend(getattr(model, component).parameters())
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=c['learning_rate'], weight_decay=c['weight_decay'])
    support_loader = loader(c, support['fake'] + support['real'], [0]*5 + [1]*5, processor,
                            training=True, batch_size=c['support_batch_size'])
    # Deterministic replay crops keep batch-index distillation targets aligned.
    memory_loader = loader(c, replay['paths'], replay['labels'], processor,
                           batch_size=c['replay_batch_size']) if c['replay_enabled'] else None
    features = {}
    if memory_loader is not None:
        teacher.to(device).eval()
        with torch.no_grad():
            for idx, (dino, fft, _) in enumerate(memory_loader):
                features[idx] = teacher(dino.to(device), fft.to(device))[0].cpu()
        fake_anchor, real_anchor = compute_anchors(teacher, memory_loader, device)
        teacher.cpu()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    history = []
    for epoch in range(c['adaptation_epochs']):
        weight = 0.0 if epoch < c['contrastive_warmup_epochs'] else c['contrastive_weight']
        if memory_loader is not None:
            losses = train_epoch(model, support_loader, memory_loader, optimizer, device,
                                 features, fake_anchor, real_anchor, c['avoidance_weight'],
                                 c['distillation_weight'], weight, c['temperature'])
        else:
            # Explicit no-replay ablation: contrastive-only support adaptation.
            model.train()
            batches = list(support_loader)
            dino = torch.cat([b[0] for b in batches]).to(device)
            fft = torch.cat([b[1] for b in batches]).to(device)
            labels = torch.cat([b[2] for b in batches]).to(device)
            optimizer.zero_grad()
            loss = fewshot_contrastive_loss(model(dino, fft)[0], labels, c['temperature']) * c['contrastive_weight']
            loss.backward()
            optimizer.step()
            losses = (loss.item(), loss.item(), 0., 0.)
        row = dict(state=state, epoch=epoch+1, learning_rate=c['learning_rate'],
                   **dict(zip(('loss', 'contrastive_loss', 'distillation_loss', 'avoidance_loss'), map(float, losses))))
        history.append(row)
        print(f'State {state}, epoch {epoch+1}: {row}', flush=True)
        tracking.log({'continual/step': state, **{f'train/{k}': v for k, v in row.items()}})
    return optimizer.state_dict(), history


def evaluate(c, model, processor, device, prototypes, data, names):
    from fleet.train.fewshot_experiment import validate_with_prototypes
    sets = {'real': {'paths': data['real_query'], 'labels': [1]*len(data['real_query'])}}
    for name in names:
        sets[name] = {'paths': data['queries'][name], 'labels': [0]*len(data['queries'][name])}
    retention = data['retention']
    sets['retention'] = {'paths': retention[0] + retention[1], 'labels': [0]*len(retention[0]) + [1]*len(retention[1])}
    result = validate_with_prototypes(model, sets, *prototypes, processor, device,
                                     c['xception_crop_size'], c['evaluation_batch_size'], c['num_workers'])
    real = result['real']['confidences']
    metrics = []
    for name in names:
        fake = result[name]['confidences']
        metrics.append(detection_metrics([1]*len(fake) + [0]*len(real), fake_probability(fake + real)))
    ret = detection_metrics([1]*len(retention[0]) + [0]*len(retention[1]), fake_probability(result['retention']['confidences']))
    return metrics, ret


def export_metrics(c, records, names, out):
    matrices = {metric: np.array([[domain[metric] for domain in row['domains']] for row in records])
                for metric in records[0]['domains'][0]}
    metadata = lambda row: dict(state_index=row['state'], adapted_generator=row['adapted_generator'],
                                checkpoint_identity=row['checkpoint_identity'], seed=c['seed'])
    for metric, matrix in matrices.items():
        save_csv(out / 'metrics' / f'evaluation_{metric}.csv',
                 [dict(**metadata(row), **dict(zip(names, matrix[i].tolist()))) for i, row in enumerate(records)])
    long_rows, step_rows, retention_rows = [], [], []
    for row in records:
        i = row['state']
        for j, name in enumerate(names, 1):
            long_rows.append(dict(**metadata(row), adaptation_step=i, evaluated_generator=name,
                                  relation='past' if j < i else 'current' if j == i else 'future',
                                  current_generator_phase='post' if j == i else 'pre' if j == i+1 else '',
                                  associated_adaptation_step=j if j in (i, i+1) else '',
                                  decision_threshold=.5, elapsed_seconds=row['elapsed_seconds'],
                                  **row['domains'][j-1]))
        sr = {'state_index': i}
        for metric in SUMMARY_METRICS:
            sr.update({f'{key}_{metric}': value for key, value in step_summary(matrices[metric][:i+1]).items() if key != 'state'})
        step_rows.append(sr)
        retention_rows.append(dict(state_index=i, decision_threshold=.5, **row['retention'],
                                   **{f'delta_{key}': value-records[0]['retention'][key] for key, value in row['retention'].items()}))
    save_csv(out / 'metrics' / 'metrics_long.csv', long_rows)
    save_csv(out / 'metrics' / 'continual_metrics_by_step.csv', step_rows)
    save_csv(out / 'metrics' / 'retention.csv', retention_rows)
    save_csv(out / 'metrics' / 'runtime_and_replay.csv', [dict(state=r['state'], **r['runtime'], **r['replay']) for r in records])
    save_csv(out / 'metrics' / 'training.csv', [h for r in records for h in r['training']])
    return matrices


def plots(out, matrices, names, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('auroc', 'balanced_accuracy'):
        fig, ax = plt.subplots(figsize=(max(9, len(names)*.22), 7))
        heat = ax.imshow(matrices[metric], vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax.set_ylabel('Completed adaptations (state 0 is pretrained)')
        ax.set_title(metric)
        fig.colorbar(heat, ax=ax)
        fig.tight_layout()
        fig.savefig(out / 'metrics' / f'{metric}_heatmap.png', dpi=160)
        plt.close(fig)
    for kind in ('forgetting', 'adaptation_gain'):
        fig, ax = plt.subplots(figsize=(max(9, len(names)*.22), 5))
        values = summaries['auroc'][f'per_generator_{kind}']
        ax.bar(range(len(names)), [values[n] for n in names])
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax.set_ylabel(f'AUROC {kind}')
        fig.tight_layout()
        fig.savefig(out / 'metrics' / f'per_generator_{kind}.png', dpi=160)
        plt.close(fig)


def checkpoint_save(path, payload):
    import torch
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def run(c, order, data):
    import torch
    from transformers import AutoImageProcessor
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    # Advisory lock also protects direct invocations and survives launcher exit.
    import fcntl
    lock = (out / 'training.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Experiment already running: {out}') from exc
    for d in ('checkpoints', 'metrics', 'logs', 'wandb', 'support_manifests', 'replay_manifests'):
        (out / d).mkdir(exist_ok=True)
    latest = out / 'checkpoints' / 'latest.pth'
    if latest.exists() and not c['resume']:
        raise ValueError('Existing experiment checkpoint: enable resume or choose a new experiment_name')
    if (out / 'config_resolved.yaml').exists() and not latest.exists():
        raise ValueError('Incomplete initial state: use a new experiment_name or inspect/remove the incomplete output explicitly')
    names = [r['directory'] for r in order][:c['max_generators']]
    c['generator_order'] = order
    c['active_generators'] = names
    c['git'] = git_info()
    c['checkpoint_sha256'] = digest(c['pretrained_checkpoint'])
    c['prototype_sha256'] = {n: digest(Path(c['prototype_path']) / n) for n in ('fake_prototype.npy', 'real_prototype.npy')}
    c['order_sha256'] = digest(c['generator_order_file'])
    c['run_id'] = __import__('hashlib').sha256(str(out).encode()).hexdigest()[:16]
    signature = {k: v for k, v in c.items() if k not in ('resume', 'wandb', 'git', 'evaluation_loss_definition', 'distillation_teacher', 'replay_transform', 'optimizer_policy')}
    resumed = torch.load(latest, map_location='cpu', weights_only=False) if latest.exists() else None
    if resumed and resumed['signature'] != signature:
        raise ValueError('Resume configuration/input identity differs from completed checkpoint')
    # Freeze all support, query, retention and candidate replay paths before state 0.
    manifest = json.loads(json.dumps(data))
    if resumed:
        stored = json.load(open(out / 'support_manifests' / 'experiment_splits.json'))
        if stored != manifest or digest(out / 'support_manifests' / 'experiment_splits.json') != resumed['manifest_sha256']:
            raise ValueError('Dataset or support/query manifests changed since checkpoint')
        for filename, expected in resumed['replay_manifest_hashes'].items():
            if digest(out / 'replay_manifests' / filename) != expected:
                raise ValueError(f'Replay manifest modified: {filename}')
    else:
        save_json(out / 'support_manifests' / 'experiment_splits.json', manifest)
        for name, support in data['supports'].items():
            save_json(out / 'support_manifests' / f'{name}.json', support)
    import yaml
    atomic_text(out / 'config_resolved.yaml', yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    save_csv(out / 'generator_order.csv', order)
    save_json(out / 'generator_order.json', order)
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Loading pretrained checkpoint: {c["pretrained_checkpoint"]} ({c["checkpoint_sha256"]})', flush=True)
    pretrained = torch.load(c['pretrained_checkpoint'], map_location='cpu', weights_only=False)
    processor = AutoImageProcessor.from_pretrained(c['model_path'], trust_remote_code=True, local_files_only=True)
    model = create_model(c, pretrained)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    del pretrained
    records, best_state, best_value, replay_hashes = [], 0, -1.0, {}
    if resumed:
        model.load_state_dict(resumed['model_state_dict'])
        records = resumed['records']
        best_state, best_value = resumed['best_state'], resumed['best_value']
        replay_hashes = resumed['replay_manifest_hashes']
        random.setstate(resumed['rng']['python'])
        np.random.set_state(resumed['rng']['numpy'])
        torch.set_rng_state(resumed['rng']['torch'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resumed['rng']['cuda'])
    model.to(device)
    prototypes = [torch.from_numpy(np.load(Path(c['prototype_path']) / n)).to(device)
                  for n in ('fake_prototype.npy', 'real_prototype.npy')]
    # Tracking initialization may draw random numbers; isolate it from training.
    rng_before_tracking = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    tracking = Tracking(c)
    random.setstate(rng_before_tracking[0])
    np.random.set_state(rng_before_tracking[1])
    torch.set_rng_state(rng_before_tracking[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_before_tracking[3])
    started = time.monotonic()
    previous_elapsed = records[-1]['elapsed_seconds'] if records else 0.
    if records:
        matrices = export_metrics(c, records, names, out)
    for state in range(len(records), len(names)+1):
        print(f'Continual state {state}/{len(names)}', flush=True)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        adaptation_start = time.monotonic()
        optimizer_state, history = None, []
        replay_report = dict(examples=0, cumulative_examples=0, fake_count=0, real_count=0,
                             sampling_seed=c['seed'], sampling_policy=c['replay_sampling_strategy'], presentations=0)
        if state:
            replay = replay_manifest(c, data, state)
            if not c['replay_enabled']:
                replay.update(paths=[], labels=[], fake_count=0, real_count=0)
            replay_path = out / 'replay_manifests' / f'step_{state:03d}.json'
            save_json(replay_path, replay)
            replay_hashes[replay_path.name] = digest(replay_path)
            optimizer_state, history = adapt(c, model, teacher, processor, device,
                                             data['supports'][names[state-1]], replay, tracking, state)
            replay_report.update(examples=len(replay['paths']), cumulative_examples=sum(r['replay']['examples'] for r in records)+len(replay['paths']),
                                 fake_count=replay['fake_count'], real_count=replay['real_count'], sampling_seed=replay['seed'],
                                 presentations=len(replay['paths'])*c['adaptation_epochs'])
        adaptation_time = time.monotonic() - adaptation_start
        eval_start = time.monotonic()
        domains, retention = evaluate(c, model, processor, device, prototypes, data, names)
        runtime = dict(adaptation_seconds=adaptation_time, evaluation_seconds=time.monotonic()-eval_start,
                       gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
        identity = f'{c["run_id"]}/state_{state:03d}'
        records.append(dict(state=state, adapted_generator=names[state-1] if state else '',
                            checkpoint_identity=identity, domains=domains, retention=retention,
                            elapsed_seconds=previous_elapsed+time.monotonic()-started,
                            runtime=runtime, replay=replay_report, training=history))
        score = float(np.mean([m[c['best_metric']] for m in domains]))
        if score > best_value:
            best_value, best_state = score, state
        payload = dict(completed_step=state, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer_state,
                       scheduler_state_dict=None, optimizer_policy='AdamW reset at each generator; only completed steps resume',
                       records=records, signature=signature, best_value=best_value, best_state=best_state,
                       fake_prototype=prototypes[0].cpu(), real_prototype=prototypes[1].cpu(),
                       manifest_sha256=digest(out / 'support_manifests' / 'experiment_splits.json'),
                       replay_manifest_hashes=replay_hashes,
                       rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        path = out / 'checkpoints' / f'state_{state:03d}.pth'
        checkpoint_save(path, payload)
        for alias, target_state in (('latest', state), ('best', best_state)):
            temp = out / 'checkpoints' / f'{alias}.tmp'
            temp.unlink(missing_ok=True)
            os.link(out / 'checkpoints' / f'state_{target_state:03d}.pth', temp)
            os.replace(temp, out / 'checkpoints' / f'{alias}.pth')
        # latest is the commit point; derivative summaries are rebuilt from it on resume.
        matrices = export_metrics(c, records, names, out)
        save_json(out / 'metrics' / 'completed_state.json', dict(completed_step=state, checkpoint_identity=identity,
                                                               best_state=best_state, decision_threshold=.5))
        log = {'continual/step': state, 'continual/current_generator': names[state-1] if state else '',
               'config/decision_threshold': .5, 'runtime/elapsed_seconds': records[-1]['elapsed_seconds'],
               **{f'runtime/{k}': v for k, v in runtime.items()},
               **{f'replay/{k}': v for k, v in replay_report.items()},
               **{f'retention/aigibench_{k}': v for k, v in retention.items()},
               **{f'retention/delta_{k}': v-records[0]['retention'][k] for k, v in retention.items()}}
        for metric in SUMMARY_METRICS:
            log.update({f'continual/{k}_{metric}': v for k, v in step_summary(matrices[metric]).items() if v is not None and k != 'state'})
            if state:
                log[f'continual/current_pre_{metric}'] = matrices[metric][state-1, state-1]
                log[f'continual/current_post_{metric}'] = matrices[metric][state, state-1]
        tracking.log(log)
        for old in (out / 'checkpoints').glob('state_*.pth'):
            n = int(old.stem.split('_')[1])
            periodic = c['checkpoint_frequency'] and n % c['checkpoint_frequency'] == 0
            if n not in (0, state, best_state) and not periodic:
                old.unlink()
    summaries = {m: final_summary(matrices[m], names) for m in SUMMARY_METRICS}
    save_json(out / 'metrics' / 'final_summary.json', dict(decision_threshold=.5, support_composition=c['support_composition'],
              primary_metric='auroc', completed_steps=len(names), smoke_test=len(names)!=64,
              evaluation_subset=c['max_evaluation_samples_per_generator'] is not None or c.get('max_real_evaluation_samples') is not None,
              continual=summaries, final_aigibench_retention=records[-1]['retention'],
              retention_change={k: v-records[0]['retention'][k] for k, v in records[-1]['retention'].items()},
              probability_transform=c['probability_transform'], chronology_policy=c['chronology_policy']))
    plots(out, matrices, names, summaries)
    tracking.finish(out, summaries, len(names))
    print(f'Completed {len(records)} evaluation states: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='Configs/treasure_continual_fleet_5shot_aigibench_replay.yaml')
    parser.add_argument('--validate-config', action='store_true')
    parser.add_argument('--validate-paths', action='store_true')
    parser.add_argument('--print-output-dir', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_treasure_continual.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    c = load_config(args.config)
    if args.print_output_dir:
        print(c['output_dir'])
        return
    order = load_order(c)
    print(f'Checkpoint: {c["pretrained_checkpoint"]}\nSupport: 5 fake + 5 real\nDecision threshold: 0.5')
    print('Chronology policy:', c['chronology_policy'])
    for row in order:
        print(f'{int(row["step"]):02d} {row["directory"]}: {row["release_date"] or "UNKNOWN"} | {row["source"]} | {row["uncertainty"]}')
    if args.validate_config:
        print('Configuration valid; no data/model loaded.')
        return
    data = prepare_data(c, order)
    if args.validate_paths:
        print('All 64 domains, support/query splits, replay pools, retention and configured paths validated.')
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    run(c, order, data)


if __name__ == '__main__':
    main()

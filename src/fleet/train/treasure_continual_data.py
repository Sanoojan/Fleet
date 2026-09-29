"""Configuration, deterministic manifests, and atomic output helpers."""
from __future__ import annotations

import csv
from datetime import datetime
import math
import hashlib
import io
import json
import os
from pathlib import Path
import random

from fleet.train.continual_metrics import validate_threshold

ROOT = Path(__file__).resolve().parents[3]
PATH_KEYS = ('pretrained_checkpoint', 'prototype_path', 'model_path', 'aigibench_train_path',
             'aigibench_validation_path', 'treasure_fake_root', 'treasure_real_root',
             'real_support_root', 'generator_order_file', 'output_root')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def save_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and fields is None:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    import yaml
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    c = yaml.safe_load(path.read_text())
    validate_threshold(c['decision_threshold'])
    if type(c['seed']) is not int or not 0 <= c['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    for key in ('resume', 'replay_enabled', 'treasure_replay_enabled'):
        if type(c[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    for key in ('temperature', 'learning_rate', 'weight_decay', 'distillation_weight',
                'avoidance_weight', 'contrastive_weight'):
        if not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if c['temperature'] == 0 or c['contrastive_warmup_epochs'] < 0:
        raise ValueError('temperature must be positive and warmup epochs nonnegative')
    for key in PATH_KEYS:
        p = Path(os.path.expandvars(c[key])).expanduser()
        c[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    if Path(c['output_root']) != ROOT / 'All_outputs':
        raise ValueError('output_root must resolve to <repository>/All_outputs')
    name = c['experiment_name']
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('experiment_name must be one directory name')
    c['output_dir'] = str(Path(c['output_root']) / name)
    if c['shot_count'] != 5 or c['support_composition'] != {'fake': 5, 'real': 5}:
        raise ValueError('This experiment requires 5 fake + 5 real support images')
    if c['treasure_replay_enabled']:
        raise ValueError('Treasure replay is reserved for future ablations; leave disabled')
    if c['replay_sampling_strategy'] not in ('fixed', 'resample_per_step'):
        raise ValueError('replay_sampling_strategy must be fixed or resample_per_step')
    if c['wandb']['mode'] not in ('online', 'offline', 'disabled'):
        raise ValueError('W&B mode must be online/offline/disabled')
    for key in ('adaptation_epochs', 'replay_size', 'replay_batch_size', 'support_batch_size',
                'evaluation_batch_size', 'retention_samples_per_class'):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['num_workers'] < 0 or c['checkpoint_frequency'] < 0 or c['learning_rate'] <= 0:
        raise ValueError('Invalid workers, checkpoint frequency, or learning rate')
    if not 0 < c['replay_fake_fraction'] < 1 or c['replay_batch_ratio'] <= 0:
        raise ValueError('Replay must contain both classes and a positive batch ratio')
    if c['replay_batch_size'] != round(10 * c['replay_batch_ratio']):
        raise ValueError('replay_batch_size must equal round(10 * replay_batch_ratio)')
    if not 1 <= round(c['replay_size'] * c['replay_fake_fraction']) < c['replay_size']:
        raise ValueError('Replay size/fraction must yield both classes')
    for key in ('max_generators', 'max_evaluation_samples_per_generator'):
        if c[key] is not None and (not isinstance(c[key], int) or c[key] < 1):
            raise ValueError(f'{key} must be null or a positive integer')
    real_limit = c.get('max_real_evaluation_samples')
    if real_limit is not None and (type(real_limit) is not int or real_limit < 1):
        raise ValueError('max_real_evaluation_samples must be null or a positive integer')
    if c['max_generators'] is not None and c['max_generators'] > 64:
        raise ValueError('max_generators cannot exceed 64')
    if not set(c['trainable_components']) <= {'lora', 'xception', 'xception_fc', 'routing', 'projection'} or not c['trainable_components']:
        raise ValueError('Invalid trainable_components')
    if c['best_metric'] not in ('auroc', 'balanced_accuracy', 'average_precision', 'accuracy'):
        raise ValueError('best_metric must be a supported macro detection metric')
    c['probability_transform'] = 'sigmoid(cos_fake - cos_real); explicit uncalibrated adapter, Fleet uses confidence <= 0'
    c['support_policy'] = 'independent name-derived seed; sorted canonical paths; 5 fake + 5 cc12m real; union of all supports excluded globally'
    c['chronology_policy'] = 'known dates ascending, exact-directory tie-break; unknown dates last with explicit uncertainty'
    c['evaluation_loss_definition'] = 'binary log loss of explicit uncalibrated sigmoid probability adapter'
    c['distillation_teacher'] = 'original AIGIBench-pretrained model; fixed prototypes and replay anchors'
    c['replay_transform'] = 'Fleet center crop, deterministic for distillation target alignment'
    c['optimizer_policy'] = 'AdamW reset per generator, no scheduler, completed-step resume'
    return c


def load_order(c):
    rows = list(csv.DictReader(open(c['generator_order_file'], newline='')))
    required = {'step', 'directory', 'display_name', 'release_date', 'family', 'source', 'uncertainty'}
    if len(rows) != 64 or len({r['directory'] for r in rows}) != 64:
        raise ValueError('Generator order must contain exactly 64 unique directories')
    if any(not required <= r.keys() or int(r['step']) != i for i, r in enumerate(rows, 1)):
        raise ValueError('Generator order fields/steps invalid')
    for row in rows:
        if row['release_date']:
            formats = {4: '%Y', 7: '%Y-%m', 10: '%Y-%m-%d'}
            try:
                datetime.strptime(row['release_date'], formats[len(row['release_date'])])
            except (ValueError, KeyError) as exc:
                raise ValueError(f'Invalid release date for {row["directory"]}') from exc
    if any(Path(r['directory']).name != r['directory'] or r['directory'] in ('.', '..') for r in rows):
        raise ValueError('Generator directory names must be single path components')
    if rows != sorted(rows, key=lambda r: (r['release_date'] or '9999', r['directory'])):
        raise ValueError('Order must follow documented date and directory tie-break policy')
    if any(not r['source'] or (not r['release_date'] and not r['uncertainty']) for r in rows):
        raise ValueError('Every row needs provenance and unknown dates need uncertainty notes')
    return rows


def images(root):
    return sorted({str(p.resolve()) for p in Path(root).rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file()})


def stream_seed(seed, name):
    return (seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)) % (2**32)


def sample(paths, count, seed):
    if len(paths) < count:
        raise ValueError(f'Requested {count} samples from only {len(paths)} images')
    return random.Random(seed).sample(sorted(paths), count)


def exclude_support(paths, support):
    excluded = {str(Path(p).resolve()) for p in support}
    # Canonical names catch symlink aliases; inode identities catch hard links.
    inodes = {(Path(p).stat().st_dev, Path(p).stat().st_ino) for p in excluded}
    return [p for p in paths if str(Path(p).resolve()) not in excluded
            and (Path(p).stat().st_dev, Path(p).stat().st_ino) not in inodes]


def prepare_data(c, order):
    """Read-only preflight: aggregate missing paths before importing the model."""
    errors = []
    for key in PATH_KEYS:
        if key == 'output_root':
            continue
        if not Path(c[key]).exists():
            errors.append(f'{key}: {c[key]}')
    for filename in ('fake_prototype.npy', 'real_prototype.npy'):
        if not (Path(c['prototype_path']) / filename).is_file():
            errors.append(f'prototype: {Path(c["prototype_path"]) / filename}')
    print('Preflight: scanning all 64 Treasure image domains...', flush=True)
    pools = {}
    for row in order:
        root = Path(c['treasure_fake_root']) / row['directory']
        pools[row['directory']] = images(root) if root.is_dir() else []
        if len(pools[row['directory']]) <= c['shot_count']:
            errors.append(f'{root}: need >5 images, found {len(pools[row["directory"]])}')
    if errors:
        raise ValueError('Missing/insufficient datasets or files:\n  ' + '\n  '.join(errors))
    from fleet.datasets import collect_aigibench_image_paths
    print('Preflight: collecting AIGIBench training/validation images...', flush=True)
    train_paths, train_labels = collect_aigibench_image_paths(c['aigibench_train_path'])
    val_paths, val_labels = collect_aigibench_image_paths(c['aigibench_validation_path'])
    train = {y: sorted({str(Path(p).resolve()) for p, label in zip(train_paths, train_labels) if y == label}) for y in (0, 1)}
    val = {y: sorted({str(Path(p).resolve()) for p, label in zip(val_paths, val_labels) if y == label}) for y in (0, 1)}
    real_pool = images(c['real_support_root'])
    cc = [p for p in real_pool if 'cc12m' in Path(p).name.lower()]
    real_pool = cc or real_pool
    supports = {}
    for row in order:
        name = row['directory']
        seed = stream_seed(c['seed'], name)
        supports[name] = dict(fake=sample(pools[name], 5, seed), real=sample(real_pool, 5, seed + 1), seed=seed)
    print('Preflight: excluding every support image from fixed query and retention sets...', flush=True)
    all_support = [p for s in supports.values() for cls in ('fake', 'real') for p in s[cls]]
    real_query = exclude_support(images(c['treasure_real_root']), all_support)
    queries = {name: exclude_support(paths, all_support) for name, paths in pools.items()}
    limit = c['max_evaluation_samples_per_generator']
    def cap(paths, count, salt):
        return sample(paths, min(len(paths), count), stream_seed(c['seed'], salt)) if count else paths
    real_query = cap(real_query, c.get('max_real_evaluation_samples', limit), 'real_query')
    queries = {name: cap(paths, limit, name + '/query') for name, paths in queries.items()}
    retention = {y: cap(exclude_support(val[y], all_support), c['retention_samples_per_class'], f'retention/{y}') for y in (0, 1)}
    print('Preflight: checking replay/evaluation separation...', flush=True)
    excluded_eval = real_query + [p for paths in queries.values() for p in paths] + retention[0] + retention[1]
    # Exclude evaluation aliases and all Treasure support from AIGIBench replay.
    train = {y: exclude_support(train[y], excluded_eval + all_support) for y in (0, 1)}
    counts = {0: round(c['replay_size'] * c['replay_fake_fraction'])}
    counts[1] = c['replay_size'] - counts[0]
    for y in (0, 1):
        if len(train[y]) < counts[y]:
            errors.append(f'AIGIBench training class {y}: need {counts[y]}, have {len(train[y])}')
        if not retention[y]:
            errors.append(f'AIGIBench retention class {y} is empty')
    if not real_query or any(not v for v in queries.values()):
        errors.append('Empty Treasure query split after support exclusion')
    if errors:
        raise ValueError('\n'.join(errors))
    return dict(supports=supports, queries=queries, real_query=real_query, train=train,
                retention=retention, replay_counts=counts)


def replay_manifest(c, data, step):
    seed = c['seed'] if c['replay_sampling_strategy'] == 'fixed' else c['seed'] + step
    paths, labels = [], []
    for y in (0, 1):
        selected = sample(data['train'][y], data['replay_counts'][y], seed + y)
        paths.extend(selected)
        labels.extend([y] * len(selected))
    return dict(step=step, seed=seed, policy=c['replay_sampling_strategy'], paths=paths,
                labels=labels, fake_count=labels.count(0), real_count=labels.count(1))

"""Detection and continual metrics; fake is positive, values are fractions."""
from __future__ import annotations

import numpy as np

THRESHOLD = 0.5
SUMMARY_METRICS = ('auroc', 'balanced_accuracy', 'average_precision', 'accuracy')


def validate_threshold(value):
    if isinstance(value, bool) or value != THRESHOLD:
        raise ValueError('decision_threshold must be exactly 0.5; calibration is unsupported')


def fake_probability(confidence):
    """Explicit adapter: Fleet confidence = cos(real) - cos(fake).

    Fleet itself has no probability output. sigmoid(-confidence) preserves its
    zero decision boundary including ties (fake). No fitted temperature.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(confidence))
    # Preserve Fleet's sign decision even when a tiny positive score rounds to .5.
    return np.where((confidence > 0) & (probability == .5),
                    np.nextafter(.5, 0.), probability)


def detection_metrics(labels, probabilities, decision_threshold=THRESHOLD):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    validate_threshold(decision_threshold)
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError('Expected matching nonempty 1D labels/probabilities')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected binary labels and finite probabilities in [0,1]')
    positive = y == 1
    pred = p >= THRESHOLD
    tp, tn = int((pred & positive).sum()), int((~pred & ~positive).sum())
    fp, fn = int((pred & ~positive).sum()), int((~pred & positive).sum())
    n_fake, n_real = tp + fn, tn + fp
    if not n_fake or not n_real:
        raise ValueError('Each evaluation domain must contain both real and fake images')
    tpr, tnr = tp / n_fake, tn / n_real
    fpr, curve_tpr, _ = roc_curve(y, p, drop_intermediate=False)
    # Empirical operating point, no interpolation or threshold calibration.
    ece = 0.0
    bins = np.minimum((p * 15).astype(int), 14)
    for b in range(15):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return dict(n_real=n_real, n_fake=n_fake, auroc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                accuracy=(tp + tn) / len(y), balanced_accuracy=(tpr + tnr) / 2,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tpr, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                fake_accuracy=tpr, real_accuracy=tnr, tp=tp, tn=tn, fp=fp, fn=fn,
                tpr_at_1pct_fpr=float(curve_tpr[fpr <= .01].max()),
                tpr_at_0_1pct_fpr=float(curve_tpr[fpr <= .001].max()),
                ece=float(ece), mean_eval_loss=float(-(y * np.log(clipped) + (1-y) * np.log(1-clipped)).mean()))


def _matrix(matrix):
    r = np.asarray(matrix, dtype=float)
    if r.ndim != 2 or not r.shape[1] or not 1 <= r.shape[0] <= r.shape[1] + 1 or not np.isfinite(r).all():
        raise ValueError('Expected finite (completed_states, generators) matrix including state 0')
    return r


def step_summary(matrix):
    r = _matrix(matrix)
    i, t = r.shape[0] - 1, r.shape[1]
    out = dict(state=i, all_macro=float(r[i].mean()), seen_macro=None,
               unseen_macro=float(r[i, i:].mean()) if i < t else None,
               current_adaptation_gain=None, mean_adaptation_gain=None,
               running_bwt=None, running_forgetting=None, forward_performance=None,
               running_fwt=None, average_incremental=None)
    if i:
        j = np.arange(i)
        gains = r[j + 1, j] - r[j, j]
        out.update(seen_macro=float(r[i, :i].mean()), current_adaptation_gain=float(gains[-1]),
                   mean_adaptation_gain=float(gains.mean()), forward_performance=float(r[j, j].mean()),
                   average_incremental=float(np.mean([r[k, :k].mean() for k in range(1, i + 1)])))
    if i >= 2:
        j = np.arange(i - 1)
        out.update(running_bwt=float((r[i, j] - r[j+1, j]).mean()),
                   running_forgetting=float(np.mean([r[k+1:i, k].max() - r[i, k] for k in j])),
                   running_fwt=float(np.mean([r[k, k] - r[0, k] for k in range(1, i)])))
    return out


def final_summary(matrix, generators):
    r = _matrix(matrix)
    t = len(generators)
    if r.shape != (t + 1, t):
        raise ValueError('Final summary requires every state and generator')
    j = np.arange(t)
    gains = r[j + 1, j] - r[j, j]
    forgetting = np.array([r[k+1:, k].max() - r[t, k] for k in j])
    previous_declined = int((r[t, :-1] < r[j[:-1]+1, j[:-1]]).sum())
    out = dict(average=float(r[-1].mean()), bwt=None, forgetting=None, fwt=None,
               forward_performance=float(r[j, j].mean()),
               average_incremental=step_summary(r)['average_incremental'],
               mean_adaptation_gain=float(gains.mean()), median_adaptation_gain=float(np.median(gains)),
               worst_generator=float(r[-1].min()), best_generator=float(r[-1].max()),
               std_across_generators=float(r[-1].std()), improved_count=int((gains > 0).sum()),
               improved_percentage=float(100 * (gains > 0).mean()),
               previous_declined_count=previous_declined,
               previous_declined_percentage=100 * previous_declined / (t-1) if t > 1 else None,
               maximum_forgetting=float(forgetting.max()),
               generator_maximum_forgetting=generators[int(forgetting.argmax())],
               largest_positive_gain_generator=generators[int(gains.argmax())] if gains.max() > 0 else None,
               largest_negative_gain_generator=generators[int(gains.argmin())] if gains.min() < 0 else None,
               per_generator_forgetting=dict(zip(generators, forgetting.tolist())),
               per_generator_adaptation_gain=dict(zip(generators, gains.tolist())))
    if t > 1:
        out.update(bwt=float((r[t, :-1] - r[j[:-1]+1, j[:-1]]).mean()),
                   forgetting=float(forgetting[:-1].mean()),
                   fwt=float((r[j[1:], j[1:]] - r[0, 1:]).mean()))
    return out

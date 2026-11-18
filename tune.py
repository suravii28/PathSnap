"""Bayesian/TPE tuning for the PathSnap HistGradientBoosting baseline.
Run from the folder containing the original train.py.
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from train import FEATURES, build_data, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--out', default='tuning_results')
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max-points', type=int, default=90)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    import optuna
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print('Loading TRAIN partition only. Student-test data is not used for tuning.', flush=True)
    train, stats = build_data(args.zip, 'train', args.k, args.stride, args.max_points)
    if train.empty:
        raise RuntimeError('No training rows generated')
    stats.to_csv(out / 'train_trip_stats.csv', index=False)
    splits = list(GroupKFold(n_splits=min(args.folds, train.trip.nunique())).split(train[FEATURES], train.label, groups=train.trip))
    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float('learning_rate', 0.015, 0.25, log=True),
            max_iter=trial.suggest_int('max_iter', 70, 280, step=10),
            max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 7, 63),
            max_depth=trial.suggest_int('max_depth', 3, 16),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 10, 80),
            l2_regularization=trial.suggest_float('l2_regularization', 1e-4, 20, log=True),
            random_state=args.seed,
            early_stopping=False,
        )
        fold_scores = []
        for tr_idx, va_idx in splits:
            model = HistGradientBoostingClassifier(**params)
            model.fit(train.iloc[tr_idx][FEATURES], train.iloc[tr_idx].label)
            fold_scores.append(score(model, train.iloc[va_idx])['observation_top1_proxy_accuracy'])
        trial.set_user_attr('fold_scores', [float(v) for v in fold_scores])
        result = float(np.mean(fold_scores))
        print(f'Trial {trial.number+1}/{args.trials}: CV Top-1 proxy={result:.5f}', flush=True)
        return result
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=args.seed), study_name='pathsnap_histgb_tpe')
    start = time.perf_counter()
    study.optimize(objective, n_trials=args.trials)
    study.trials_dataframe().to_csv(out / 'optuna_trials.csv', index=False)
    params = {**study.best_params, 'random_state': args.seed, 'early_stopping': False}
    best = HistGradientBoostingClassifier(**params)
    best.fit(train[FEATURES], train.label)
    joblib.dump(best, out / 'tuned_hist_gradient_boosting.joblib')
    report = {
        'model': 'hist_gradient_boosting', 'tuning_method': 'Optuna TPE',
        'objective': 'mean GroupKFold observation Top-1 proxy accuracy',
        'best_cv_top1_proxy_accuracy': study.best_value,
        'baseline_cv_top1_proxy_accuracy': 0.577257975761859,
        'best_params': study.best_params, 'n_trials': len(study.trials),
        'duration_seconds': time.perf_counter() - start,
        'training_trips': int(train.trip.nunique()), 'training_candidate_rows': int(len(train)),
        'features': FEATURES, 'args': vars(args),
        'caveats': [
            'Reference route proximity produces proxy labels, not verified GPS observation labels.',
            'Training excludes observations without a positive candidate.',
            'Midpoint KDTree candidates can miss geometrically nearest road segments.',
            'Student-test trips are not used for tuning or model selection.'
        ]
    }
    (out / 'tuning_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'best_cv_top1_proxy_accuracy': study.best_value, 'best_params': study.best_params, 'output': str(out.resolve())}, indent=2))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
score.py — faithful local copy of the OFFICIAL EchoRisk Task1 scorer.

Identical to /root/EchoPrime-Mamba/evaluation/score.py: merge on
(patient_id, timepoint), MAE (primary) / RMSE / Pearson r, with the same input
validation and JSON output.  Shipped here so the deliverable is self-contained;
training_run.py also cross-checks against the canonical file when present.

Usage: score.py <predictions.csv> <ground_truth.csv>
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

PRED, GT = sys.argv[1], sys.argv[2]
pred = pd.read_csv(PRED)
gt = pd.read_csv(GT)

assert {'patient_id', 'timepoint', 'lvef_pred'}.issubset(pred.columns), \
    f'predictions missing required columns; got {list(pred.columns)}'

gt_keys = set(zip(gt.patient_id.astype(str), gt.timepoint.astype(str)))
pr_keys = set(zip(pred.patient_id.astype(str), pred.timepoint.astype(str)))
missing = gt_keys - pr_keys
if missing:
    sys.exit(f'ERROR: predictions missing {len(missing)} (patient_id, timepoint) pairs.')

bad = pred[(pred.lvef_pred < 0) | (pred.lvef_pred > 100)]
if len(bad) > 0:
    sys.exit(f'ERROR: {len(bad)} lvef_pred values outside [0, 100]')
if pred.lvef_pred.isna().any():
    sys.exit('ERROR: NaN in lvef_pred')
dup = pred.duplicated(subset=['patient_id', 'timepoint']).sum()
if dup > 0:
    sys.exit(f'ERROR: {dup} duplicate (patient_id, timepoint) keys')

m = gt.merge(pred, on=['patient_id', 'timepoint'], how='left', validate='one_to_one')
y_true = m['lvef'].to_numpy(dtype=np.float64)
y_pred = m['lvef_pred'].to_numpy(dtype=np.float64)

mae = float(mean_absolute_error(y_true, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
n_unique = len(np.unique(y_pred))
pearson = None if n_unique < 2 else float(pearsonr(y_true, y_pred)[0])

print(json.dumps({
    'mae': round(mae, 4),
    'rmse': round(rmse, 4),
    'pearson_r': None if pearson is None else round(pearson, 4),
    'n': int(len(y_true)),
    'n_unique_preds': int(n_unique),
}, indent=2))

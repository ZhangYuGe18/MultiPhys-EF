"""v7_turbo evaluate.py — official-style metrics + per-bin MAE on a submission CSV.

  python evaluate.py                         # evaluate submissions/submission_val.csv
  python evaluate.py --csv path/to/sub.csv   # evaluate a specific submission
  python evaluate.py --gt  path/to/gt.csv    # against a specific ground truth

Reports MAE / RMSE / r / bias / slope, biplane vs single-view split, per-EF-bin
MAE, and the official score.py JSON.
"""
import os, sys, json, argparse, subprocess
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config as cfg
from echolvef.fusion import evaluate as ef_eval


def per_bin_mae(p, y, m, bins=[(0, 55, '[0,55)'), (55, 65, '[55,65)'),
                                (65, 70, '[65,70)'), (70, 100, '[70,inf)')]):
    out = {}
    for lo, hi, name in bins:
        b = m & (y >= lo) & (y < hi)
        if b.sum() > 0:
            out[name] = (float(np.mean(np.abs(p[b] - y[b]))), int(b.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(cfg.SUBMISSIONS, 'submission_val.csv'))
    ap.add_argument('--gt', default=cfg.GT_VAL)
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        print(f'ERROR: {a.csv} not found. Run infer.py / train.py first.'); return

    print(f'=== v7_turbo evaluate: {a.csv} ===')
    va = pd.read_csv(a.gt).dropna(subset=['lvef'])
    sub = pd.read_csv(a.csv)
    merged = va.merge(sub, on=['patient_id', 'timepoint'], how='left')
    p = merged.lvef_pred.values.astype(np.float32)
    y = merged.lvef.values.astype(np.float32)
    bi = (merged.video_a4c.notna() & merged.video_a2c.notna()).values

    m = ef_eval(p, y, tag='submission')
    print(f'  n={len(y)}  MAE={m["MAE"]:.4f}  RMSE={m["RMSE"]:.4f}  r={m["r"]:.4f}  '
          f'bias={m["bias"]:+.3f}  slope={m["slope"]:.3f}')
    print(f'  biplane(n={int(bi.sum())}) MAE={float(np.mean(np.abs(p[bi]-y[bi]))):.4f}  '
          f'single-view(n={int((~bi).sum())}) MAE={float(np.mean(np.abs(p[~bi]-y[~bi]))):.4f}')

    print('\nPer-bin MAE (biplane):')
    for k, (mm, n) in per_bin_mae(p, y, bi).items():
        print(f'  {k:10s} MAE={mm:.4f} (n={n})')

    if os.path.exists(cfg.SCORE):
        print('\n=== OFFICIAL score.py ===')
        r = subprocess.run([sys.executable, cfg.SCORE, a.csv, a.gt], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())


if __name__ == '__main__':
    main()

"""v7_turbo infer.py — cold-start LVEF inference (NO pre-cache), GPU-maximised.

Decodes DICOM live, computes appearance + geometry + motion on the GPU (with the
per-frame CPU glue moved into parallel decode workers, and the GPU models batched /
pipelined), applies the pre-fitted bundle, writes the official submission CSV.

  python infer.py --split val                      # verify MAE 4.0342 + timing
  python infer.py --split val --mode batched        # 1.5x faster (fp16-equivalent)
  python infer.py --split test --dicom-root <dir> --labels <test.csv> --out sub.csv

Modes:
  exact   (default) per-video GPU batch sizes match the reference -> MAE 4.0342
          to within 0.003 EF (precision held).
  batched cross-exam GPU batching -> fp16-equivalent (~4.034, max |Δ|=0.08 EF) for
          maximum GPU utilisation on weak GPUs.
"""
import os, sys, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config as cfg
from echolvef import turbo_core as core
from echolvef.heads_bundle import load_bundle


def list_exams(labels_csv, dicom_root):
    df = pd.read_csv(labels_csv)
    has_lvef = 'lvef' in df.columns
    if has_lvef:
        df = df.dropna(subset=['lvef']).reset_index(drop=True)
    exams = []
    for _, r in df.iterrows():
        def pth(v):
            return f"{dicom_root}/{r.patient_id}/{r.timepoint}/{v}" if isinstance(v, str) else None
        y = float(r['lvef']) if has_lvef and not pd.isna(r.get('lvef')) else np.nan
        exams.append(dict(pid=str(r.patient_id), tp=str(r.timepoint), y=y,
                          a4c=pth(r.video_a4c), a2c=pth(r.video_a2c)))
    return exams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--labels', default=None)
    ap.add_argument('--dicom-root', default=None)
    ap.add_argument('--bundle', default=cfg.BUNDLE)
    ap.add_argument('--mode', default='exact', choices=['exact', 'batched'])
    ap.add_argument('--workers', type=int, default=0, help='0 = adaptive (cpu_count-2, cap 32)')
    ap.add_argument('--batch', type=int, default=16, help='exams per GPU batch / log interval')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()

    labels = a.labels or (cfg.GT_VAL if a.split == 'val' else cfg.GT_VAL)
    dicom_root = a.dicom_root or os.path.join(cfg.DICOM_ROOT, a.split)
    out = a.out or os.path.join(cfg.SUBMISSIONS, f'submission_{a.split}.csv')
    workers = a.workers or max(1, min((os.cpu_count() or 4) - 2, 32))

    core.log('=' * 64)
    core.log(f'v7_turbo cold-start inference — split={a.split}  mode={a.mode}')
    core.log('=' * 64)
    core.log(f'  labels={labels}')
    core.log(f'  dicom_root={dicom_root}')
    core.log(f'  bundle={a.bundle}  workers={workers}  batch={a.batch}')

    exams = list_exams(labels, dicom_root)
    if a.n:
        exams = exams[:a.n]
    bundle = load_bundle(a.bundle)

    preds, t = core.run(exams, bundle, workers, a.batch, label=a.split, mode=a.mode)

    sub = pd.DataFrame({'patient_id': [e['pid'] for e in exams],
                        'timepoint': [e['tp'] for e in exams],
                        'lvef_pred': np.clip(preds, 0, 100)})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sub.to_csv(out, index=False)
    core.log(f'  wrote {out}')

    y = np.array([e['y'] for e in exams])
    if np.isfinite(y).all() and len(y):
        err = preds - y; ae = np.abs(err)
        r = float(np.corrcoef(preds, y)[0, 1])
        core.log(f'  ==== n={len(y)}  MAE={ae.mean():.4f}  RMSE={np.sqrt((err**2).mean()):.4f}  '
                 f'r={r:.4f}  bias={err.mean():+.3f}  (reference 4.0342) ====')

    ntest = 340
    proj = t['total'] / t['n'] * ntest
    core.log(f'  ==== TIMING: wall {t["total"]:.1f}s for {t["n"]} exams ({t["nviews"]} videos) ====')
    core.log(f'       gpu {t["gpu"]:.1f}s | apply {t["apply"]*1000:.1f}ms')
    core.log(f'       -> test ({ntest} exams) projection: {proj:.0f}s = {proj/60:.1f} min  '
             f'(15-min budget: {"OK" if proj < 900 else "OVER"})')


if __name__ == '__main__':
    main()

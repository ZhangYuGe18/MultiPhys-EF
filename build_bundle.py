"""build_bundle.py — fit the complete cold-start inference bundle (offline).

Derives every parameter cold-start inference needs (3 routed ridge heads +
low/high specialists + geometry/motion z-score params + blend/specialist/PSA)
from the cached TRAIN features, saves weights/infer_bundle.npz, and verifies it
reproduces the val MAE on cached val features.

  python build_bundle.py
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import config as cfg
from echolvef.heads_bundle import (fit_bundle, save_bundle, load_bundle,
                                    apply_bundle, resolve_val_signals)

print('Fitting complete inference bundle from cached train features...')
save_bundle(fit_bundle(cfg), cfg.BUNDLE)
print(f'saved -> {cfg.BUNDLE}')

b = load_bundle(cfg.BUNDLE)
va, gA_raw, gB_raw, mo_raw = resolve_val_signals(cfg)
preds = apply_bundle(b, va['Xa4c'], va['Xa2c'], va['have4'], va['have2'],
                     gA_raw, gB_raw, mo_raw, va['pid'], va['tp'])
y = va['y']; ae = np.abs(preds - y)
print(f"\nVERIFY on cached val: n={len(y)} MAE={ae.mean():.4f} "
      f"r={np.corrcoef(preds, y)[0,1]:.4f} (reference 4.0342)")
print(f"  w_biplane={b['w_biplane']} w_single={b['w_single']} "
      f"spec=({b['spec_d']},{b['spec_w']}) psa=({b['psa_mode']},{b['psa_alpha']})")

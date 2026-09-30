"""
Complete inference bundle: everything needed to turn per-exam features into a final
LVEF, WITHOUT any refitting at inference time.

The shipped weights/heads.npz only has the 3 routed ridge heads + selected
(w, specialist, PSA) params. For standalone inference we also need the low/high
specialist heads and the geometry/motion z-score parameters. `fit_bundle` derives
the complete bundle once (offline, from cached train features), `apply_bundle`
turns features -> predictions (reproduces the val pipeline exactly).
"""
import numpy as np
from . import fusion as Fu


def _resolve_per_exam(G, pid_key, tp_key, c4, c2, cbi, keys, have4, both):
    """biplane-else-single raw signal per exam, aligned to `keys`."""
    return np.where(both, Fu._align(G, pid_key, tp_key, cbi, keys),
                    np.where(have4, Fu._align(G, pid_key, tp_key, c4, keys),
                             Fu._align(G, pid_key, tp_key, c2, keys)))


def fit_bundle(cfg):
    """Derive the complete inference bundle from cached TRAIN features + heads.npz."""
    L = lambda p: np.load(p, allow_pickle=True)
    tr = L(cfg.F_APP_TRAIN); cam = L(cfg.F_CAMUS)
    gA = L(cfg.F_GEOM_A); gB = L(cfg.F_GEOM_B); mo = L(cfg.F_MOTION)
    H = L(cfg.HEADS)
    A = cfg.RIDGE_ALPHAS
    cg = cam['qual'] == 'Good'
    Caf, C2, Cef = cam['X4ch'][cg], cam['X2ch'][cg], cam['ef'][cg]
    ytr = tr['y']

    # low / high specialist heads (A4C), fit-on-all (same recipe as fusion)
    lm, hm, cl, ch = ytr < 55, ytr > 65, Cef < 55, Cef > 65
    _, _, Hlow = Fu.oof_and_val(tr['Xa4c'][lm], ytr[lm], tr['pid'][lm], Caf[cl], Cef[cl], tr['Xa4c'][:1], A)
    _, _, Hhigh = Fu.oof_and_val(tr['Xa4c'][hm], ytr[hm], tr['pid'][hm], Caf[ch], Cef[ch], tr['Xa4c'][:1], A)

    # geometry / motion z-score params (train), matching fusion._zscore_to_ef
    btr = tr['have4'] & tr['have2']; kt = Fu._key(tr['pid'], tr['tp'])
    gAt = _resolve_per_exam(gA, 'ptr', 'tptr', 'tr_4', 'tr_2', 'tr_bi', kt, tr['have4'], btr)
    gBt = _resolve_per_exam(gB, 'ptr', 'tptr', 'tr_4', 'tr_2', 'tr_bi', kt, tr['have4'], btr)
    mot = np.where(tr['have4'], Fu._align(mo, 'ptr', 'tptr', 'tr_4', kt),
                   Fu._align(mo, 'ptr', 'tptr', 'tr_2', kt))

    def zp(sig):
        m = np.isfinite(sig)
        return (float(sig[m].mean()), float(sig[m].std() + 1e-9), float(ytr[m].mean()), float(ytr[m].std()))

    return dict(
        bi=(H['bi_coef'], H['bi_mu'], H['bi_sd']),
        a4=(H['a4_coef'], H['a4_mu'], H['a4_sd']),
        a2=(H['a2_coef'], H['a2_mu'], H['a2_sd']),
        low=(Hlow['coef'], Hlow['mu'], Hlow['sd']),
        high=(Hhigh['coef'], Hhigh['mu'], Hhigh['sd']),
        w_biplane=float(H['w_biplane']), w_single=float(H['w_single']),
        spec_d=float(H['spec_d']), spec_w=float(H['spec_w']),
        psa_mode=str(H['psa_mode']), psa_alpha=float(H['psa_alpha']),
        gA=zp(gAt), gB=zp(gBt), mo=zp(mot),
    )


def save_bundle(bundle, path):
    flat = {}
    for k, v in bundle.items():
        if isinstance(v, tuple):
            for i, x in enumerate(v):
                flat[f'{k}__{i}'] = np.asarray(x)
        else:
            flat[k] = np.asarray(v)
    np.savez(path, **flat)


def load_bundle(path):
    d = np.load(path, allow_pickle=True)
    groups = {}
    for k in d.files:
        if '__' in k:
            base, i = k.split('__'); groups.setdefault(base, {})[int(i)] = d[k]
        else:
            groups[k] = d[k]
    out = {}
    for k, v in groups.items():
        if isinstance(v, dict):
            out[k] = tuple(v[i] for i in sorted(v))
        else:
            out[k] = v.item() if v.ndim == 0 else v
    return out


def _pred(head, X):
    coef, mu, sd = head
    return Fu.ridge_predict((X - mu) / sd, coef)


def _zs(raw, p):
    mu, sd, ym, ys = p
    return np.where(np.isfinite(raw), (raw - mu) / sd * ys + ym, np.nan)


def apply_bundle(b, Xa4c, Xa2c, have4, have2, gA_raw, gB_raw, mo_raw, pid, tp):
    """Per-exam features -> final LVEF. Reproduces the val pipeline exactly."""
    n = len(have4); both = have4 & have2
    app = np.full(n, np.nan)
    bi = np.where(both)[0]
    if len(bi):
        app[bi] = _pred(b['bi'], np.hstack([Xa4c[bi], Xa2c[bi]]))
    a4o = np.where((~both) & have4)[0]
    if len(a4o):
        app[a4o] = _pred(b['a4'], Xa4c[a4o])
    a2o = np.where((~both) & have2 & ~have4)[0]
    if len(a2o):
        app[a2o] = _pred(b['a2'], Xa2c[a2o])

    fin4 = np.all(np.isfinite(Xa4c), 1)
    low = np.full(n, np.nan); high = np.full(n, np.nan)
    if fin4.any():
        low[fin4] = _pred(b['low'], Xa4c[fin4]); high[fin4] = _pred(b['high'], Xa4c[fin4])

    zgA, zgB, zmo = _zs(gA_raw, b['gA']), _zs(gB_raw, b['gB']), _zs(mo_raw, b['mo'])
    fill = lambda s, bb: np.where(np.isfinite(s), s, bb)
    con = np.nanmean(np.vstack([fill(zgA, app), fill(zgB, app), fill(zmo, app)]), 0)
    base = np.where(both, (1 - b['w_biplane']) * app + b['w_biplane'] * con,
                    (1 - b['w_single']) * app + b['w_single'] * con)

    d, w = b['spec_d'], b['spec_w']; p = base.copy()
    for i in np.where(~both)[0]:
        if np.isfinite(low[i]) and low[i] < base[i] - d and low[i] < 55:
            p[i] = (1 - w) * base[i] + w * low[i]
        elif np.isfinite(high[i]) and high[i] > base[i] + d and high[i] > 65:
            p[i] = (1 - w) * base[i] + w * high[i]
    v3 = np.clip(p, 0, 100)

    bi_pred = np.where(both, v3, np.nan)
    final = Fu._apply_psa(pid, tp, both, v3, bi_pred, b['psa_mode'], b['psa_alpha'])
    return np.clip(final, 0, 100)


def resolve_val_signals(cfg):
    """For verification: resolve cached val geom/motion to per-exam raw signals."""
    va = np.load(cfg.F_APP_VAL, allow_pickle=True)
    gA = np.load(cfg.F_GEOM_A, allow_pickle=True); gB = np.load(cfg.F_GEOM_B, allow_pickle=True)
    mo = np.load(cfg.F_MOTION, allow_pickle=True)
    kv = Fu._key(va['pid'], va['tp']); bva = va['have4'] & va['have2']
    gA_raw = _resolve_per_exam(gA, 'pva', 'tpva', 'va_4', 'va_2', 'va_bi', kv, va['have4'], bva)
    gB_raw = _resolve_per_exam(gB, 'pva', 'tpva', 'va_4', 'va_2', 'va_bi', kv, va['have4'], bva)
    mo_raw = np.where(va['have4'], Fu._align(mo, 'pva', 'tpva', 'va_4', kv),
                      Fu._align(mo, 'pva', 'tpva', 'va_2', kv))
    return va, gA_raw, gB_raw, mo_raw

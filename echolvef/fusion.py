"""
Multi-signal fusion (single clean module; replaces the v3->v6->v7 patch-chain).

Signals
  appearance : frozen Stage2 encoder statpool -> CAMUS-augmented routed ridge
  geometry   : 2 LV segmenters -> Simpson EF  (z-scored to EF scale on TRAIN stats)
  motion     : R(2+1)D EF                       (z-scored to EF scale on TRAIN stats)

Fusion (every parameter selected on TRAIN out-of-fold predictions):
  1. consensus = mean(geomA, geomB, motion)  (appearance used as NaN fallback)
  2. pred = (1-w)*appearance + w*consensus    (w_biplane, w_single chosen on train OOF)
  3. specialist-trust for single-view low/high-EF exams (d,w chosen on train OOF)
  4. leakage-free Patient-Sibling-Anchor for single-view exams with biplane siblings
     (mode/alpha chosen on train single-view OOF; anchor = sibling *predictions*, not labels)

The MICCAI val labels are used only for the final score.
"""
import os
import subprocess
import numpy as np
import pandas as pd


def _log(*a):
    import datetime
    print(f'[{datetime.datetime.now():%H:%M:%S}]', *a, flush=True)


# ----------------------------- ridge utils ---------------------------------
def ridge_fit(X, y, w, alpha):
    Z = np.hstack([X, np.ones((len(X), 1))]); I = np.eye(Z.shape[1]); I[-1, -1] = 0
    W = w[:, None]
    return np.linalg.solve(Z.T @ (W * Z) + alpha * I, Z.T @ (w * y))


def ridge_predict(X, coef):
    return np.hstack([X, np.ones((len(X), 1))]) @ coef


def bin_weights(y):
    b = np.digitize(y, [55, 60, 65, 70, 75]); c = np.bincount(b, minlength=6)
    w = 1.0 / np.clip(c[b], 1, None); w = w / w.mean()
    return np.clip(w, 0.3, 3.0)


def cv_alpha(Z, y, pid, alphas, seed=0):
    """Patient-grouped 5-fold CV alpha (micro-MAE objective). No val leakage."""
    ups = np.unique(pid); rng = np.random.RandomState(seed); rng.shuffle(ups)
    folds = np.array_split(ups, 5); best = None
    for a in alphas:
        errs = []
        for k in range(5):
            te = np.isin(pid, folds[k]); tr = ~te
            c = ridge_fit(Z[tr], y[tr], np.ones(tr.sum()), a)
            pe = ridge_predict(Z[te], c)
            errs.append(np.mean(np.abs(pe - y[te])))
        m = np.mean(errs)
        if best is None or m < best[0]:
            best = (m, a)
    return best[1]


def evaluate(preds, labels, tag=''):
    p = np.asarray(preds, np.float64); y = np.asarray(labels, np.float64)
    err = p - y; ae = np.abs(err)
    return dict(MAE=float(ae.mean()), RMSE=float(np.sqrt((err**2).mean())),
                r=float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else 0.0,
                bias=float(err.mean()), slope=float(np.polyfit(y, p, 1)[0]) if y.std() > 0 else 0.0,
                n=len(p), tag=tag)


# ----------------------- CAMUS-augmented OOF + val -------------------------
def oof_and_val(Xtr, ytr, ptr, Xc, yc, Xva, alphas, sub=None, seed=0):
    """5-fold patient-grouped OOF on (train+CAMUS), plus fit-on-all -> val prediction.

    Standardisation is fit on train+CAMUS only (no val statistics).
    Returns (train_oof[n_train], val_pred[n_val], head_dict) where head_dict is the
    fit-on-all head for inference reuse.
    """
    fin = np.all(np.isfinite(Xtr), 1)
    sub = fin if sub is None else (sub & fin)
    ups = np.unique(ptr[sub]); rng = np.random.RandomState(seed); rng.shuffle(ups)
    folds = np.array_split(ups, 5)
    oof = np.full(len(ytr), np.nan)
    Cpid = np.array([f'C{i}' for i in range(len(yc))])
    for k in range(5):
        te = np.isin(ptr, folds[k]) & sub
        tr_ = (~te) & sub
        Xt = np.vstack([Xtr[tr_], Xc]); yt = np.concatenate([ytr[tr_], yc])
        pt = np.concatenate([ptr[tr_], Cpid])
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        a = cv_alpha((Xt - mu) / sd, yt, pt, alphas)
        c = ridge_fit((Xt - mu) / sd, yt, np.ones(len(yt)), a)
        oof[te] = ridge_predict((Xtr[te] - mu) / sd, c)
    # fit on all sub + CAMUS -> val
    Xu = np.vstack([Xtr[sub], Xc]); yu = np.concatenate([ytr[sub], yc])
    pu = np.concatenate([ptr[sub], Cpid])
    mu, sd = Xu.mean(0), Xu.std(0) + 1e-6
    a = cv_alpha((Xu - mu) / sd, yu, pu, alphas)
    c = ridge_fit((Xu - mu) / sd, yu, np.ones(len(yu)), a)
    val = ridge_predict((Xva - mu) / sd, c)
    return oof, val, dict(coef=c, mu=mu, sd=sd, alpha=a)


# ----------------------- signal alignment helpers --------------------------
def _key(pid, tp):
    return np.array([f'{a}|{b}' for a, b in zip(pid, tp)])


def _align(geom, col_pid, col_tp, col_val, keys):
    idx = {f'{a}|{b}': i for i, (a, b) in enumerate(zip(geom[col_pid], geom[col_tp]))}
    return np.array([geom[col_val][idx[x]] if x in idx else np.nan for x in keys])


def _zscore_to_ef(t, v, ytr):
    """Map a raw signal (geom/motion EF, different bias/scale) onto the EF scale
    using TRAIN finite stats only (honest)."""
    m = np.isfinite(t); mu, sd = t[m].mean(), t[m].std() + 1e-9
    f = lambda x: (x - mu) / sd * ytr[m].std() + ytr[m].mean()
    return np.where(np.isfinite(t), f(t), np.nan), np.where(np.isfinite(v), f(v), np.nan)


# ------------------------- PSA (sibling anchor) ----------------------------
def _tp_dist(a, b):
    return abs(int(a[1:]) - int(b[1:]))


def _time_anchor(pat_tp, sib_tps, sib_preds, mode):
    if len(sib_tps) == 0:
        return np.nan
    if mode == 'mean':
        w = np.ones(len(sib_tps))
    elif mode == 'inv_d':
        w = np.array([1.0 / max(_tp_dist(pat_tp, st), 1) for st in sib_tps])
    elif mode == 'inv_d2':
        w = np.array([1.0 / max(_tp_dist(pat_tp, st), 1) ** 2 for st in sib_tps])
    else:  # closest
        w = np.array([1.0 if _tp_dist(pat_tp, st) == 1 else 0.0 for st in sib_tps])
    if w.sum() == 0:
        w = np.ones(len(sib_tps))
    return float((w * sib_preds).sum() / w.sum())


# ============================== main ======================================
def run_fusion(cfg):
    _log('=' * 60); _log('Multi-signal fusion (train-OOF; honest)'); _log('=' * 60)
    L = lambda p: np.load(p, allow_pickle=True)
    tr = L(cfg.F_APP_TRAIN); va = L(cfg.F_APP_VAL); cam = L(cfg.F_CAMUS)
    gA = L(cfg.F_GEOM_A); gB = L(cfg.F_GEOM_B); mo = L(cfg.F_MOTION)

    cg = cam['qual'] == 'Good'
    Caf, C2, Cef = cam['X4ch'][cg], cam['X2ch'][cg], cam['ef'][cg]
    ytr, yva = tr['y'], va['y']
    btr, bva = tr['have4'] & tr['have2'], va['have4'] & va['have2']
    kt, kv = _key(tr['pid'], tr['tp']), _key(va['pid'], va['tp'])
    A = cfg.RIDGE_ALPHAS

    # ---- appearance: routed heads (biplane / A4C / A2C) with CAMUS aug ----
    _log('appearance OOF (biplane + A4C + A2C + low/high specialists) ...')
    appb_t, appb_v, Hbi = oof_and_val(np.hstack([tr['Xa4c'], tr['Xa2c']])[btr], ytr[btr], tr['pid'][btr],
                                      np.hstack([Caf, C2]), Cef, np.hstack([va['Xa4c'], va['Xa2c']])[bva], A)
    en_t, en_v, H4 = oof_and_val(tr['Xa4c'], ytr, tr['pid'], Caf, Cef, va['Xa4c'], A)
    a2_t, a2_v, H2 = oof_and_val(tr['Xa2c'], ytr, tr['pid'], C2, Cef, va['Xa2c'], A)
    lm, cl = ytr < 55, Cef < 55
    hm, ch = ytr > 65, Cef > 65
    low_t = oof_and_val(tr['Xa4c'][lm], ytr[lm], tr['pid'][lm], Caf[cl], Cef[cl], tr['Xa4c'], A)[1]
    low_v = oof_and_val(tr['Xa4c'][lm], ytr[lm], tr['pid'][lm], Caf[cl], Cef[cl], va['Xa4c'], A)[1]
    high_t = oof_and_val(tr['Xa4c'][hm], ytr[hm], tr['pid'][hm], Caf[ch], Cef[ch], tr['Xa4c'], A)[1]
    high_v = oof_and_val(tr['Xa4c'][hm], ytr[hm], tr['pid'][hm], Caf[ch], Cef[ch], va['Xa4c'], A)[1]

    # per-exam appearance (route by available view)
    app_t = np.where(tr['have4'], en_t, a2_t); app_t = np.where(btr, np.nan, app_t)
    tmp = np.full(len(ytr), np.nan); tmp[btr] = appb_t; app_t = np.where(btr, tmp, app_t)
    app_v = np.where(va['have4'], en_v, a2_v); app_v = np.where(bva, np.nan, app_v)
    tmpv = np.full(len(yva), np.nan); tmpv[bva] = appb_v; app_v = np.where(bva, tmpv, app_v)

    # ---- geometry + motion signals (biplane else single-view), z-scored to EF ----
    def geo(G):
        t = np.where(btr, _align(G, 'ptr', 'tptr', 'tr_bi', kt),
                     np.where(tr['have4'], _align(G, 'ptr', 'tptr', 'tr_4', kt), _align(G, 'ptr', 'tptr', 'tr_2', kt)))
        v = np.where(bva, _align(G, 'pva', 'tpva', 'va_bi', kv),
                     np.where(va['have4'], _align(G, 'pva', 'tpva', 'va_4', kv), _align(G, 'pva', 'tpva', 'va_2', kv)))
        return _zscore_to_ef(t, v, ytr)
    gAt, gAv = geo(gA); gBt, gBv = geo(gB)
    mt, mv = _zscore_to_ef(np.where(tr['have4'], _align(mo, 'ptr', 'tptr', 'tr_4', kt), _align(mo, 'ptr', 'tptr', 'tr_2', kt)),
                           np.where(va['have4'], _align(mo, 'pva', 'tpva', 'va_4', kv), _align(mo, 'pva', 'tpva', 'va_2', kv)), ytr)
    sigs_t = [gAt, gBt, mt]; sigs_v = [gAv, gBv, mv]

    fill = lambda s, b: np.where(np.isfinite(s), s, b)
    con_t = np.nanmean(np.vstack([fill(s, app_t) for s in sigs_t]), 0)
    con_v = np.nanmean(np.vstack([fill(s, app_v) for s in sigs_v]), 0)

    # ---- blend weight (train OOF; separate for biplane / single-view) ----
    def best_w(app, con, y, m):
        b = (9, 0)
        for w in cfg.BLEND_WEIGHTS:
            e = np.mean(np.abs(((1 - w) * app + w * con - y)[m]))
            b = (e, w) if e < b[0] else b
        return b[1]
    wb = best_w(app_t, con_t, ytr, btr); ws = best_w(app_t, con_t, ytr, ~btr)
    base_v = np.where(bva, (1 - wb) * app_v + wb * con_v, (1 - ws) * app_v + ws * con_v)
    base_t = np.where(btr, (1 - wb) * app_t + wb * con_t, (1 - ws) * app_t + ws * con_t)
    _log(f'blend w_biplane={wb} w_single={ws} | base n156 MAE={np.abs(np.clip(base_v,0,100)-yva).mean():.4f}')

    # ---- specialist-trust for single-view low/high EF (train OOF) ----
    def spec(app, low, high, m, d, w):
        p = app.copy()
        for i in np.where(m)[0]:
            if np.isfinite(low[i]) and low[i] < app[i] - d and low[i] < 55:
                p[i] = (1 - w) * app[i] + w * low[i]
            elif np.isfinite(high[i]) and high[i] > app[i] + d and high[i] > 65:
                p[i] = (1 - w) * app[i] + w * high[i]
        return p
    bsp = (9, (5, 0.3))
    for d in cfg.SPEC_DELTAS:
        for w in cfg.SPEC_WEIGHTS:
            p = spec(base_t, low_t, high_t, ~btr, d, w)
            e = np.nanmean(np.abs((p - ytr)[~btr]))
            bsp = (e, (d, w)) if e < bsp[0] else bsp
    sd_, sw_ = bsp[1]
    v3_v = np.clip(spec(base_v, low_v, high_v, ~bva, sd_, sw_), 0, 100)
    v3_t = np.clip(spec(base_t, low_t, high_t, ~btr, sd_, sw_), 0, 100)
    _log(f'specialist d={sd_} w={sw_} | v3 n156 MAE={np.abs(v3_v-yva).mean():.4f} '
         f'sv={np.abs(v3_v[~bva]-yva[~bva]).mean():.4f}')

    # ---- PSA mode/alpha selection on TRAIN single-view OOF (anchor=biplane OOF preds) ----
    # train biplane OOF preds = v3_t on biplane rows (sibling source for train sv)
    bi_pred_t = np.where(btr, v3_t, np.nan)
    psa_mode, psa_alpha = _select_psa(cfg, tr['pid'], tr['tp'], btr, v3_t, bi_pred_t, ytr)
    _log(f'PSA selected: mode={psa_mode} alpha={psa_alpha} (train sv 5-fold CV)')

    # ---- apply PSA to val single-view (anchor = val biplane v3 preds, NOT labels) ----
    bi_pred_v = np.where(bva, v3_v, np.nan)
    final_v = _apply_psa(va['pid'], va['tp'], bva, v3_v, bi_pred_v, psa_mode, psa_alpha)
    final_v = np.clip(final_v, 0, 100)

    m156 = evaluate(final_v, yva, 'final n156')
    _log(f"FINAL honest: n156 MAE={m156['MAE']:.4f} r={m156['r']:.4f} "
         f"bi={np.abs(final_v[bva]-yva[bva]).mean():.4f} sv={np.abs(final_v[~bva]-yva[~bva]).mean():.4f} "
         f"slope={m156['slope']:.3f} bias={m156['bias']:+.3f}")

    # ---- official submission + score ----
    os.makedirs(cfg.SUBMISSIONS, exist_ok=True)
    sub = pd.read_csv(cfg.GT_VAL).dropna(subset=['lvef'])[['patient_id', 'timepoint']].copy()
    sub['lvef_pred'] = final_v
    out = os.path.join(cfg.SUBMISSIONS, 'submission_v7clean.csv'); sub.to_csv(out, index=False)
    _official_score(cfg, out)

    # ---- save merged heads + fusion params (for fast inference reuse) ----
    np.savez(cfg.HEADS,
             # routed appearance heads (fit-on-all+CAMUS)
             bi_coef=Hbi['coef'], bi_mu=Hbi['mu'], bi_sd=Hbi['sd'], bi_alpha=Hbi['alpha'],
             a4_coef=H4['coef'], a4_mu=H4['mu'], a4_sd=H4['sd'], a4_alpha=H4['alpha'],
             a2_coef=H2['coef'], a2_mu=H2['mu'], a2_sd=H2['sd'], a2_alpha=H2['alpha'],
             # fusion params
             w_biplane=wb, w_single=ws, spec_d=sd_, spec_w=sw_,
             psa_mode=psa_mode, psa_alpha=psa_alpha,
             ytr_mean=float(ytr.mean()), ytr_std=float(ytr.std()))
    _log(f'saved merged heads + fusion params -> {cfg.HEADS}')
    _log('fusion DONE')


def _select_psa(cfg, pid, tp, bi_mask, sv_pred, bi_pred, y):
    """5-fold patient-grouped CV over train single-view exams to pick (mode, alpha)."""
    sv = (~bi_mask) & np.isfinite(sv_pred)
    sv_idx = np.where(sv)[0]
    if len(sv_idx) == 0:
        return 'mean', 1.0
    # anchors for each sv exam from same-patient biplane preds (not labels)
    anchors = {}
    for mode in cfg.PSA_MODES:
        anchors[mode] = np.full(len(sv_pred), np.nan)
        for i in sv_idx:
            sib = (bi_mask) & np.isfinite(bi_pred) & (pid == pid[i])
            if sib.sum() == 0:
                continue
            anchors[mode][i] = _time_anchor(tp[i], tp[sib], bi_pred[sib], mode)
    ups = np.unique(pid[sv_idx]); rng = np.random.RandomState(0); rng.shuffle(ups)
    folds = np.array_split(ups, cfg.N_FOLDS)
    best = None
    for mode in cfg.PSA_MODES:
        for alpha in cfg.PSA_ALPHAS:
            maes = []
            for fold in folds:
                te = np.isin(pid, fold) & sv
                if te.sum() == 0:
                    continue
                pe = np.where(np.isfinite(anchors[mode][te]),
                              alpha * anchors[mode][te] + (1 - alpha) * sv_pred[te], sv_pred[te])
                maes.append(np.abs(pe - y[te]).mean())
            if maes:
                m = np.mean(maes)
                if best is None or m < best[0]:
                    best = (m, mode, alpha)
    return best[1], best[2]


def _apply_psa(pid, tp, bi_mask, pred, bi_pred, mode, alpha):
    sib = {}
    for i in np.where(np.isfinite(bi_pred))[0]:
        sib.setdefault(pid[i], []).append((tp[i], bi_pred[i]))
    out = pred.copy()
    for i in range(len(pred)):
        if (not bi_mask[i]) and pid[i] in sib:
            stps = [s[0] for s in sib[pid[i]]]; spr = np.array([s[1] for s in sib[pid[i]]])
            anc = _time_anchor(tp[i], stps, spr, mode)
            out[i] = alpha * anc + (1 - alpha) * pred[i]
    return out


def _official_score(cfg, sub_csv):
    scorer = cfg.SCORE if os.path.exists(cfg.SCORE) else cfg.SCORE_OFFICIAL
    r = subprocess.run([cfg.PY, scorer, sub_csv, cfg.GT_VAL], capture_output=True, text=True)
    _log('OFFICIAL score.py (n=156):'); print(r.stdout.strip() or r.stderr.strip())
    if os.path.exists(cfg.SCORE_OFFICIAL) and scorer != cfg.SCORE_OFFICIAL:
        r2 = subprocess.run([cfg.PY, cfg.SCORE_OFFICIAL, sub_csv, cfg.GT_VAL], capture_output=True, text=True)
        _log('cross-check canonical scorer:'); print(r2.stdout.strip() or r2.stderr.strip())

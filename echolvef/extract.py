"""
Feature extraction (all three modalities + CAMUS augmentation).

  appearance : my honestly-trained Stage2 encoder -> statpool(mean|max|std) 1536-d/view
  geometry   : 2 LV segmenters -> ED/ES by cavity area -> Simpson disks -> EF   (label-free)
  motion     : EchoNet R(2+1)D-18 -> EF                                          (label-free)
  CAMUS      : appearance statpool on CAMUS half-sequences (wide-distribution aug)

Appearance + CAMUS depend on the backbone, so they are re-extracted from MY
backbone.  Geometry + motion are label-free w.r.t. MICCAI and reuse the
pretrained aux weights.  All preprocessing matches the audited pipeline.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision
from scipy.ndimage import uniform_filter1d, label as cc_label

from .models import Stage2MTLModel, load_segmenter
from .data import (extract_uniform_clips, N_CLIPS, miccai_exam_rows,
                   parse_cfg, load_camus_seq)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _log(*a):
    import datetime
    print(f'[{datetime.datetime.now():%H:%M:%S}]', *a, flush=True)


def clahe_norm(gray_u8):
    return (_clahe.apply(gray_u8.astype(np.uint8)).astype(np.float32) / 255.)


# ============================ appearance ===================================
@torch.no_grad()
def _statpool(enc, arr, device):
    clips = extract_uniform_clips(arr.astype(np.float32), N_CLIPS, False).to(device)
    with torch.autocast('cuda'):
        f = enc(clips).reshape(N_CLIPS, 512).float()
    return torch.cat([f.mean(0), f.amax(0), f.std(0)]).cpu().numpy()


def _appearance_split(enc, cfg, split, device):
    cache = cfg.CACHE_TR if split == 'train' else cfg.CACHE_VA
    rows = miccai_exam_rows(cfg.LABELS, split)
    D = 1536; Xa4, Xa2, h4, h2, y, pid, tp = [], [], [], [], [], [], []
    for r in rows:
        s4 = s2 = None
        if r['f4'] and os.path.exists(os.path.join(cache, r['f4'])):
            s4 = _statpool(enc, np.load(os.path.join(cache, r['f4'])), device)
        if r['f2'] and os.path.exists(os.path.join(cache, r['f2'])):
            s2 = _statpool(enc, np.load(os.path.join(cache, r['f2'])), device)
        Xa4.append(s4 if s4 is not None else np.full(D, np.nan)); h4.append(s4 is not None)
        Xa2.append(s2 if s2 is not None else np.full(D, np.nan)); h2.append(s2 is not None)
        y.append(r['y']); pid.append(r['pid']); tp.append(r['tp'])
    out = cfg.F_APP_TRAIN if split == 'train' else cfg.F_APP_VAL
    np.savez(out, Xa4c=np.array(Xa4, np.float32), Xa2c=np.array(Xa2, np.float32),
             have4=np.array(h4), have2=np.array(h2), y=np.array(y, np.float32),
             pid=np.array(pid), tp=np.array(tp))
    _log(f'appearance[{split}]: n={len(y)} both={int((np.array(h4)&np.array(h2)).sum())} -> {out}')


def _appearance_camus(enc, cfg, device):
    X4, X2, ef, qual, pid = [], [], [], [], []
    pats = sorted([d for d in os.listdir(cfg.CAMUS) if d.startswith('patient')])
    for i, pat in enumerate(pats):
        try:
            c4 = parse_cfg(f'{cfg.CAMUS}/{pat}/Info_4CH.cfg')
            s4 = load_camus_seq(f'{cfg.CAMUS}/{pat}/{pat}_4CH_half_sequence.nii.gz')
            s2 = load_camus_seq(f'{cfg.CAMUS}/{pat}/{pat}_2CH_half_sequence.nii.gz')
        except Exception:
            continue
        X4.append(_statpool(enc, s4, device)); X2.append(_statpool(enc, s2, device))
        # CAMUS Info_*.cfg uses key 'EF' (some forks use 'LVef'); ground-truth LVEF, backbone-independent
        ef.append(float(c4.get('EF', c4.get('LVef', 'nan')))); qual.append(c4.get('ImageQuality', 'Medium')); pid.append(pat)
        if (i + 1) % 100 == 0:
            _log(f'  CAMUS {i+1}/{len(pats)}')
    np.savez(cfg.F_CAMUS, X4ch=np.array(X4, np.float32), X2ch=np.array(X2, np.float32),
             ef=np.array(ef, np.float32), qual=np.array(qual), pid=np.array(pid))
    _log(f'appearance[CAMUS]: n={len(ef)} Good={int((np.array(qual)=="Good").sum())} -> {cfg.F_CAMUS}')


# ============================ geometry =====================================
def _lv_disks(mask, n=20):
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return None
    P = np.stack([xs, ys], 1).astype(np.float32); P -= P.mean(0)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    t = P @ vt[0]; w = P @ vt[1]; L = t.max() - t.min()
    edges = np.linspace(t.min(), t.max(), n + 1); d = np.zeros(n)
    for i in range(n):
        mm = (t >= edges[i]) & (t < edges[i + 1])
        if mm.sum() >= 2:
            d[i] = w[mm].max() - w[mm].min()
    if d[:n // 2].mean() > d[n // 2:].mean():
        d = d[::-1]
    return L, d


def _simpson_single(mk, n=20):
    a = _lv_disks(mk, n)
    return float(np.sum(np.pi / 4 * a[1] ** 2 * (a[0] / n))) if a else None


def _simpson_bi(m4, m2, n=20):
    a4 = _lv_disks(m4, n); a2 = _lv_disks(m2, n)
    if not (a4 and a2):
        return None
    L = (a4[0] + a2[0]) / 2
    return float(np.sum(np.pi / 4 * a4[1] * a2[1] * (L / n)))


def _largest_cc(mk):
    lab, nn_ = cc_label(mk)
    if nn_ <= 1:
        return mk
    s = np.bincount(lab.ravel()); s[0] = 0
    return lab == s.argmax()


@torch.no_grad()
def _ed_es_masks(seg, npy, device):
    if not os.path.exists(npy):
        return None
    raw = np.load(npy)[..., 0]
    arr = np.stack([clahe_norm(raw[t]) for t in range(len(raw))])
    masks = []
    for i in range(0, len(arr), 64):
        x = torch.from_numpy(arr[i:i + 64][:, None].astype(np.float32)).to(device)
        with torch.autocast('cuda'):
            masks.append((torch.sigmoid(seg(x))[:, 0] > 0.5).cpu().numpy())
    masks = np.concatenate(masks)
    masks = np.stack([_largest_cc(masks[t]) for t in range(len(masks))])
    ar = masks.reshape(len(masks), -1).sum(1)
    if ar.max() < 20:
        return None
    a = uniform_filter1d(ar.astype(np.float32), 3)
    return masks[int(a.argmax())], masks[int(a.argmin())]


def _ef(ve, vs):
    return (ve - vs) / ve * 100 if (ve and vs and ve > vs) else np.nan


def _geometry(seg, cfg, split, device):
    cache = cfg.CACHE_TR if split == 'train' else cfg.CACHE_VA
    rows = miccai_exam_rows(cfg.LABELS, split)
    g4, g2, gbi, y, pid, tp = [], [], [], [], [], []
    for r in rows:
        e4 = _ed_es_masks(seg, f'{cache}/{r["f4"]}', device) if r['f4'] else None
        e2 = _ed_es_masks(seg, f'{cache}/{r["f2"]}', device) if r['f2'] else None
        g4.append(_ef(_simpson_single(e4[0]), _simpson_single(e4[1])) if e4 else np.nan)
        g2.append(_ef(_simpson_single(e2[0]), _simpson_single(e2[1])) if e2 else np.nan)
        gbi.append(_ef(_simpson_bi(e4[0], e2[0]), _simpson_bi(e4[1], e2[1])) if (e4 and e2) else np.nan)
        y.append(r['y']); pid.append(r['pid']); tp.append(r['tp'])
    return dict(g4=np.array(g4), g2=np.array(g2), gbi=np.array(gbi),
                y=np.array(y, np.float32), pid=np.array(pid), tp=np.array(tp))


def _geometry_both(cfg, ckpt, out, device):
    seg, dice = load_segmenter(ckpt, device)
    _log(f'segmenter {os.path.basename(ckpt)} dice={dice:.3f}')
    gt = _geometry(seg, cfg, 'train', device); gv = _geometry(seg, cfg, 'val', device)
    np.savez(out, tr_4=gt['g4'], tr_2=gt['g2'], tr_bi=gt['gbi'], ytr=gt['y'], ptr=gt['pid'], tptr=gt['tp'],
             va_4=gv['g4'], va_2=gv['g2'], va_bi=gv['gbi'], yva=gv['y'], pva=gv['pid'], tpva=gv['tp'])
    _log(f'geometry -> {out}')


# ============================ motion =======================================
def _motion(cfg, device):
    MEAN, STD = 33.741, 51.046
    m = torchvision.models.video.r2plus1d_18(num_classes=1)
    sd = torch.load(cfg.R2P1D, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd); m = m.to(device).eval()
    _log('R(2+1)D EchoNet EF model loaded')

    @torch.no_grad()
    def pred_video(npy, n_clips=4, T=32):
        if not os.path.exists(npy):
            return np.nan
        arr = np.load(npy)
        f = np.stack([cv2.resize(arr[i], (112, 112)) for i in range(len(arr))]).astype(np.float32)
        f = (f - MEAN) / STD; F = len(f); outs = []
        for c in range(n_clips):
            if F >= T:
                s0 = int(c * max(1, (F - T) / max(1, n_clips - 1))); s0 = min(s0, F - T); clip = f[s0:s0 + T]
            else:
                clip = np.tile(f, (T // F + 1, 1, 1, 1))[:T]
            x = torch.from_numpy(clip).permute(3, 0, 1, 2)[None].to(device)
            outs.append(float(m(x).item()))
        return float(np.mean(outs))

    def run(split):
        cache = cfg.CACHE_TR if split == 'train' else cfg.CACHE_VA
        rows = miccai_exam_rows(cfg.LABELS, split)
        p4, p2, y, pid, tp = [], [], [], [], []
        for r in rows:
            p4.append(pred_video(f'{cache}/{r["f4"]}') if r['f4'] else np.nan)
            p2.append(pred_video(f'{cache}/{r["f2"]}') if r['f2'] else np.nan)
            y.append(r['y']); pid.append(r['pid']); tp.append(r['tp'])
        return np.array(p4), np.array(p2), np.array(y, np.float32), np.array(pid), np.array(tp)

    a = run('train'); b = run('val')
    np.savez(cfg.F_MOTION, tr_4=a[0], tr_2=a[1], ytr=a[2], ptr=a[3], tptr=a[4],
             va_4=b[0], va_2=b[1], yva=b[2], pva=b[3], tpva=b[4])
    mm = np.isfinite(b[0])
    _log(f'motion val A4C r={np.corrcoef(b[0][mm], b[2][mm])[0,1]:.3f} -> {cfg.F_MOTION}')


# ============================ orchestrator =================================
def run_extract(cfg, device, force=False):
    # appearance (depends on MY backbone)
    if force or not (os.path.exists(cfg.F_APP_TRAIN) and os.path.exists(cfg.F_APP_VAL) and os.path.exists(cfg.F_CAMUS)):
        enc_model = Stage2MTLModel(cfg.ENCODER, cfg.BACKBONE, r=cfg.S2_LORA_R).to(device)
        ck = torch.load(cfg.BACKBONE, map_location='cpu', weights_only=False)
        enc_model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
        enc_model.eval(); enc = enc_model.enc
        _log('Stage2 backbone loaded for appearance extraction')
        _appearance_split(enc, cfg, 'train', device)
        _appearance_split(enc, cfg, 'val', device)
        _appearance_camus(enc, cfg, device)
        del enc_model, enc; torch.cuda.empty_cache()
    else:
        _log('appearance/CAMUS features present, skip')
    # geometry (label-free)
    if force or not os.path.exists(cfg.F_GEOM_A):
        _geometry_both(cfg, cfg.SEG_A, cfg.F_GEOM_A, device)
    if force or not os.path.exists(cfg.F_GEOM_B):
        _geometry_both(cfg, cfg.SEG_B, cfg.F_GEOM_B, device)
    # motion (label-free)
    if force or not os.path.exists(cfg.F_MOTION):
        _motion(cfg, device)
    _log('extract DONE')

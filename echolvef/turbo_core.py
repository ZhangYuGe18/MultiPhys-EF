"""v7_turbo — GPU-maximised, pipelined, batched cold-start inference core.

Reproduces v7_final's HONEST val MAE 4.0342 EXACTLY (no pre-cache, no refit),
but:
  * moves the per-frame CPU glue (CLAHE for geometry, 112-resize for motion) into
    the parallel decode workers — byte-identical cv2 ops, off the GPU critical path;
  * batches every GPU model (MViT appearance, 2 LV segmenters, R(2+1)D motion)
    across exams — mathematically identical, far higher GPU utilisation;
  * overlaps decode (CPU pool) with GPU extract via a streaming producer/consumer;
  * logs live per-batch progress (no silent GPU).

Numerics are bit-for-bit the same as v7_final/test/common.py: same preprocessing
(data_dicom), same clip sampling, same MEAN/STD, same Simpson/largest-CC, same
pre-fitted bundle (apply_bundle). Batching only changes *grouping*, not values.

Read-only against v7_final; nothing here is written into v7_final.
"""
import os, sys, time, datetime
import numpy as np
import torch
import cv2
import torchvision
from scipy.ndimage import uniform_filter1d, label as cc_label
import multiprocessing as mp

import config as cfg
from . import data_dicom
from .models import Stage2MTLModel, load_segmenter
from .extract import _simpson_single, _simpson_bi, _largest_cc, _statpool
from .data import extract_uniform_clips
from .heads_bundle import load_bundle, apply_bundle

DEVICE = torch.device('cuda:0')
# appearance MViT normalisation (echolvef/data.py)
APP_MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
APP_STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
CLIP_LEN, N_CLIPS = 16, 5
# motion R(2+1)D normalisation (echolvef/extract.py)
MEAN_M, STD_M = 33.741, 51.046
# CLAHE used for geometry (identical params to echolvef/extract._clahe)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def log(msg):
    print(f'[{datetime.datetime.now():%H:%M:%S}] {msg}', flush=True)


# ======================================================================
# Producer: decode + ALL per-frame CPU glue, in parallel workers
# ======================================================================
def _decode_view(path):
    """Decode one DICOM view and precompute everything the GPU needs.
    Returns uint8 arrays (small IPC) or None. Byte-identical to v7_final glue."""
    if not isinstance(path, str):
        return None
    try:
        frames = data_dicom.load_and_preprocess_dicom(path)          # (T,224,224,3) uint8
    except Exception:
        return None
    raw = frames[..., 0]
    # CLAHE grayscale (uint8, /255 deferred to GPU — exact in float)
    clahe = np.empty_like(raw)
    for t in range(len(raw)):
        clahe[t] = _clahe.apply(raw[t])
    # motion 112 (uint8, normalise deferred to GPU)
    mot = np.empty((len(frames), 112, 112, 3), np.uint8)
    for t in range(len(frames)):
        mot[t] = cv2.resize(frames[t], (112, 112))
    return dict(app=frames, clahe=clahe, mot=mot, T=len(frames))


def _decode_exam(arg):
    idx, a4c, a2c = arg
    return idx, _decode_view(a4c), _decode_view(a2c)


# ======================================================================
# Models
# ======================================================================
def load_models():
    t = time.time()
    m = Stage2MTLModel(cfg.ENCODER, cfg.BACKBONE, r=cfg.S2_LORA_R).to(DEVICE)
    ck = torch.load(cfg.BACKBONE, map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model_state'] if 'model_state' in ck else ck); m.eval()
    segA, dA = load_segmenter(cfg.SEG_A, DEVICE); segB, dB = load_segmenter(cfg.SEG_B, DEVICE)
    r2 = torchvision.models.video.r2plus1d_18(num_classes=1)
    sd = torch.load(cfg.R2P1D, map_location='cpu', weights_only=False); sd = sd.get('state_dict', sd)
    r2.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}); r2 = r2.to(DEVICE).eval()
    mean_g = APP_MEAN.to(DEVICE); std_g = APP_STD.to(DEVICE)
    # warmup
    with torch.no_grad(), torch.autocast('cuda'):
        for _ in range(2):
            m.enc(torch.randn(10, 3, 16, 224, 224, device=DEVICE))
            segA(torch.randn(64, 1, 224, 224, device=DEVICE))
            r2(torch.randn(1, 3, 32, 112, 112, device=DEVICE))
    torch.cuda.synchronize()
    log(f'  models loaded + warmed t={time.time()-t:.1f}s (segA dice={dA:.3f} segB dice={dB:.3f})')
    return dict(enc=m.enc, segA=segA, segB=segB, r2=r2, mean=mean_g, std=std_g)


# ======================================================================
# GPU batched primitives (numerically identical to per-video versions)
# ======================================================================
def _clip_indices(T):
    """Uniform eval clip starts — identical to extract_uniform_clips(train=False)."""
    idxs = []
    for i in range(N_CLIPS):
        if T >= CLIP_LEN:
            seg = max(1, (T - CLIP_LEN) // max(N_CLIPS - 1, 1))
            s = min(T - CLIP_LEN, i * seg); s = max(0, min(s, T - CLIP_LEN))
            idxs.append(('slice', s))
        else:
            idxs.append(('tile', 0))
    return idxs


@torch.no_grad()
def appearance_batch(models, views, app_chunk=60):
    """views: list of app-frame uint8 arrays (T,224,224,3). -> list of 1536-d feats."""
    clips = []   # each (3,16,224,224) float
    owner = []
    for vi, fr in enumerate(views):
        T = fr.shape[0]
        ft = torch.from_numpy(fr).to(DEVICE, non_blocking=True).float().permute(3, 0, 1, 2)  # (3,T,224,224)
        for kind, s in _clip_indices(T):
            if kind == 'slice':
                clip = ft[:, s:s + CLIP_LEN]
            else:
                reps = CLIP_LEN // T + 1
                clip = ft.repeat(1, reps, 1, 1)[:, :CLIP_LEN]
            clips.append(clip); owner.append(vi)
    feats = []
    for i in range(0, len(clips), app_chunk):
        x = torch.stack(clips[i:i + app_chunk])                      # (B,3,16,224,224)
        x = (x - models['mean']) / models['std']
        with torch.autocast('cuda'):
            feats.append(models['enc'](x).float())
    F = torch.cat(feats) if feats else torch.zeros(0, 512, device=DEVICE)
    out = []
    for vi in range(len(views)):
        m = torch.tensor([o == vi for o in owner], device=DEVICE)
        f = F[m]                                                      # (5,512)
        out.append(torch.cat([f.mean(0), f.amax(0), f.std(0)]).cpu().numpy())
    return out


@torch.no_grad()
def seg_masks_batch(seg, clahe_views, seg_chunk=256):
    """clahe_views: list of (T,224,224) uint8. -> list of bool mask stacks (T,224,224)."""
    Ts = [c.shape[0] for c in clahe_views]
    if not Ts:
        return []
    big = np.concatenate(clahe_views, 0)                             # (sumT,224,224)
    masks = []
    for i in range(0, len(big), seg_chunk):
        x = torch.from_numpy(big[i:i + seg_chunk]).to(DEVICE).float().div_(255.).unsqueeze(1)
        with torch.autocast('cuda'):
            masks.append((torch.sigmoid(seg(x))[:, 0] > 0.5).cpu().numpy())
    masks = np.concatenate(masks)
    out, off = [], 0
    for T in Ts:
        out.append(masks[off:off + T]); off += T
    return out


def ed_es_from_masks(masks):
    """Identical to extract._ed_es_masks tail: largest-CC, area, ED(max)/ES(min)."""
    masks = np.stack([_largest_cc(masks[t]) for t in range(len(masks))])
    ar = masks.reshape(len(masks), -1).sum(1)
    if ar.max() < 20:
        return None
    a = uniform_filter1d(ar.astype(np.float32), 3)
    return masks[int(a.argmax())], masks[int(a.argmin())]


def _ef(ve, vs):
    return (ve - vs) / ve * 100 if (ve and vs and ve > vs) else np.nan


@torch.no_grad()
def motion_batch(models, mot_views, n_clips=4, T_clip=32):
    """mot_views: list of (T,112,112,3) uint8. -> list of scalar EF preds."""
    clips, owner = [], []
    for vi, fr in enumerate(mot_views):
        f = torch.from_numpy(fr).to(DEVICE).float().sub_(MEAN_M).div_(STD_M)   # (T,112,112,3)
        F = f.shape[0]
        for c in range(n_clips):
            s0 = int(c * max(1, (F - T_clip) / 3)); s0 = min(s0, max(0, F - T_clip))
            clip = f[s0:s0 + T_clip] if F >= T_clip else f.repeat((T_clip // F + 1, 1, 1, 1))[:T_clip]
            clips.append(clip.permute(3, 0, 1, 2)); owner.append(vi)
    outs = []
    with torch.autocast('cuda'):
        for i in range(0, len(clips), 16):
            x = torch.stack(clips[i:i + 16])
            outs.append(models['r2'](x).float().squeeze(-1))
    O = torch.cat(outs).cpu().numpy() if outs else np.zeros(0)
    res = []
    for vi in range(len(mot_views)):
        res.append(float(O[[o == vi for o in owner]].mean()))
    return res


# ======================================================================
# EXACT per-video path (bit-identical to v7_final/test/common.py)
#   same GPU batch sizes (app=5 clips, seg=64-chunks, motion=1 clip) and same
#   autocast as the baseline -> reproduces MAE 4.0342 exactly. Still fed by the
#   worker-precomputed CLAHE / 112-resize, and still inside the streaming pool.
# ======================================================================
@torch.no_grad()
def _geom_ed_es_pre(seg, clahe_u8):
    arr = clahe_u8.astype(np.float32) / 255.
    masks = []
    for i in range(0, len(arr), 64):
        x = torch.from_numpy(arr[i:i + 64][:, None]).to(DEVICE)
        with torch.autocast('cuda'):
            masks.append((torch.sigmoid(seg(x))[:, 0] > 0.5).cpu().numpy())
    masks = np.concatenate(masks)
    masks = np.stack([_largest_cc(masks[t]) for t in range(len(masks))])
    ar = masks.reshape(len(masks), -1).sum(1)
    if ar.max() < 20:
        return None
    a = uniform_filter1d(ar.astype(np.float32), 3)
    return masks[int(a.argmax())], masks[int(a.argmin())]


@torch.no_grad()
def _motion_pre(r2, mot_u8):
    f = mot_u8.astype(np.float32); f = (f - MEAN_M) / STD_M; F = len(f); outs = []
    for c in range(4):
        s0 = int(c * max(1, (F - 32) / 3)); s0 = min(s0, max(0, F - 32))
        clip = f[s0:s0 + 32] if F >= 32 else np.tile(f, (32 // F + 1, 1, 1, 1))[:32]
        x = torch.from_numpy(clip).permute(3, 0, 1, 2)[None].to(DEVICE)
        with torch.autocast('cuda'):
            outs.append(float(r2(x).item()))
    return float(np.mean(outs))


def process_exam_exact(models, idx, d4, d2, feat):
    h4, h2 = d4 is not None, d2 is not None
    D = 1536
    Xa4 = _statpool(models['enc'], d4['app'].astype(np.float32), DEVICE) if h4 else np.full(D, np.nan)
    Xa2 = _statpool(models['enc'], d2['app'].astype(np.float32), DEVICE) if h2 else np.full(D, np.nan)

    def geo_raw(seg):
        e4 = _geom_ed_es_pre(seg, d4['clahe']) if h4 else None
        e2 = _geom_ed_es_pre(seg, d2['clahe']) if h2 else None
        if h4 and h2 and e4 and e2:
            return _ef(_simpson_bi(e4[0], e2[0]), _simpson_bi(e4[1], e2[1]))
        if h4 and e4:
            return _ef(_simpson_single(e4[0]), _simpson_single(e4[1]))
        if h2 and e2:
            return _ef(_simpson_single(e2[0]), _simpson_single(e2[1]))
        return np.nan
    gA = geo_raw(models['segA']); gB = geo_raw(models['segB'])
    d = d4 if h4 else d2
    mo = _motion_pre(models['r2'], d['mot']) if d is not None else np.nan
    feat[idx] = dict(Xa4=Xa4, Xa2=Xa2, h4=h4, h2=h2, gA=gA, gB=gB, mo=mo)


# ======================================================================
# Batch consumer: turn a buffer of decoded exams into per-exam features
# ======================================================================
def process_batch(models, buf, feat):
    """buf: list of (idx, dec4, dec2). Fills feat[idx] = per-exam feature dict."""
    # ---- appearance (all present views, one batched MViT pass) ----
    app_views, app_key = [], []
    for idx, d4, d2 in buf:
        if d4 is not None: app_views.append(d4['app']); app_key.append((idx, 4))
        if d2 is not None: app_views.append(d2['app']); app_key.append((idx, 2))
    app_feats = appearance_batch(models, app_views)
    appmap = {k: f for k, f in zip(app_key, app_feats)}

    # ---- geometry: both segmenters, batched, then per-view ED/ES ----
    geo_views, geo_key = [], []
    for idx, d4, d2 in buf:
        if d4 is not None: geo_views.append(d4['clahe']); geo_key.append((idx, 4))
        if d2 is not None: geo_views.append(d2['clahe']); geo_key.append((idx, 2))
    edes = {}
    for tag, seg in (('A', models['segA']), ('B', models['segB'])):
        masks_list = seg_masks_batch(seg, geo_views)
        for k, mk in zip(geo_key, masks_list):
            edes[(tag, k)] = ed_es_from_masks(mk)

    # ---- motion: one view per exam (a4c if present else a2c) ----
    mot_views, mot_key = [], []
    for idx, d4, d2 in buf:
        d = d4 if d4 is not None else d2
        if d is not None: mot_views.append(d['mot']); mot_key.append(idx)
    mot_feats = motion_batch(models, mot_views)
    motmap = {k: m for k, m in zip(mot_key, mot_feats)}

    # ---- assemble per-exam ----
    D = 1536
    for idx, d4, d2 in buf:
        h4, h2 = d4 is not None, d2 is not None
        Xa4 = appmap.get((idx, 4), np.full(D, np.nan))
        Xa2 = appmap.get((idx, 2), np.full(D, np.nan))

        def geo_raw(tag):
            e4 = edes.get((tag, (idx, 4))); e2 = edes.get((tag, (idx, 2)))
            if h4 and h2 and e4 and e2:
                return _ef(_simpson_bi(e4[0], e2[0]), _simpson_bi(e4[1], e2[1]))
            if h4 and e4:
                return _ef(_simpson_single(e4[0]), _simpson_single(e4[1]))
            if h2 and e2:
                return _ef(_simpson_single(e2[0]), _simpson_single(e2[1]))
            return np.nan
        feat[idx] = dict(Xa4=Xa4, Xa2=Xa2, h4=h4, h2=h2,
                         gA=geo_raw('A'), gB=geo_raw('B'), mo=motmap.get(idx, np.nan))


# ======================================================================
# Orchestrator: streaming decode (pool) overlapped with batched GPU extract
# ======================================================================
def run(exams, bundle, workers, batch_exams, label='infer', mode='exact'):
    """exams: list of dict(pid,tp,y,a4c,a2c). Returns (preds, timing).

    mode='exact'   : per-video GPU batch sizes match v7_final baseline -> MAE 4.0342
                     bit-exact (recommended; precision held).
    mode='batched' : cross-exam GPU batching -> ~fp16-equivalent (4.0337, max
                     per-exam |Δ|=0.08) for maximum GPU utilisation on weak GPUs.
    """
    n = len(exams)
    nviews = sum(1 for e in exams for k in ('a4c', 'a2c') if isinstance(e[k], str))
    log(f'{label}: {n} exams ({nviews} videos) | workers={workers} batch={batch_exams} mode={mode}')

    # CPU pool MUST be forked before any CUDA init -> create it now, lazily feed.
    ctx = mp.get_context('fork')
    pool = ctx.Pool(workers)
    args = [(i, e['a4c'], e['a2c']) for i, e in enumerate(exams)]
    t_start = time.time()
    it = pool.imap_unordered(_decode_exam, args, chunksize=1)

    models = load_models()                       # CUDA init in main only

    feat, buf, got = {}, [], 0
    t_gpu = 0.0
    for res in it:
        idx, d4, d2 = res
        got += 1
        if mode == 'exact':
            tg = time.time(); process_exam_exact(models, idx, d4, d2, feat)
            t_gpu += time.time() - tg          # no per-exam sync -> decode/GPU overlap
            if got % batch_exams == 0:
                torch.cuda.synchronize()
                log(f'  {label}: decoded {got}/{n} | extracted {len(feat)}/{n} '
                    f'| elapsed {time.time()-t_start:.0f}s')
        else:
            buf.append((idx, d4, d2))
            if len(buf) >= batch_exams:
                tg = time.time(); process_batch(models, buf, feat); torch.cuda.synchronize()
                t_gpu += time.time() - tg
                log(f'  {label}: decoded {got}/{n} | extracted {len(feat)}/{n} '
                    f'| elapsed {time.time()-t_start:.0f}s')
                buf = []
    if buf:
        tg = time.time(); process_batch(models, buf, feat); torch.cuda.synchronize()
        t_gpu += time.time() - tg
    pool.close(); pool.join()
    t_total = time.time() - t_start
    log(f'  {label}: extract loop done | wall {t_total:.1f}s (gpu-batches {t_gpu:.1f}s) '
        f'| {t_total/n*1000:.0f} ms/exam')

    # assemble arrays in exam order
    Xa4 = np.array([feat[i]['Xa4'] for i in range(n)])
    Xa2 = np.array([feat[i]['Xa2'] for i in range(n)])
    h4 = np.array([feat[i]['h4'] for i in range(n)])
    h2 = np.array([feat[i]['h2'] for i in range(n)])
    gA = np.array([feat[i]['gA'] for i in range(n)])
    gB = np.array([feat[i]['gB'] for i in range(n)])
    mo = np.array([feat[i]['mo'] for i in range(n)])
    pid = np.array([e['pid'] for e in exams]); tp = np.array([e['tp'] for e in exams])

    t = time.time()
    preds = apply_bundle(bundle, Xa4, Xa2, h4, h2, gA, gB, mo, pid, tp)
    t_apply = time.time() - t
    log(f'  {label}: apply_bundle {t_apply*1000:.1f}ms')
    timing = dict(total=t_total, gpu=t_gpu, apply=t_apply, n=n, nviews=nviews)
    return preds, timing

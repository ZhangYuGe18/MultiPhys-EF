"""
Backbone training (from the pretrained EchoPrime base).

  Stage1  run_stage1()  : EchoNet ED/ES contrastive.  Selection = EchoNet VAL MAE.
                          EchoNet is label-free w.r.t. MICCAI, so this is clean.
  Stage2  run_stage2_honest() : MICCAI MTL with LoRA.  Selection = patient-grouped
                          INNER-VAL split of MICCAI *train*.  The MICCAI val set is
                          never read here — that is the leakage fix vs. the original
                          run_stage2 (which selected the best epoch by MICCAI val MAE).

The MICCAI val MAE is *logged each epoch for transparency only* and explicitly
NOT used for any save/selection decision.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import Stage1ContrastiveModel, Stage2MTLModel
from .data import (EchoNetContrastive, MICCAIDatasetMTL, build_ed_es_map,
                   miccai_biplane_patient_groups, N_CLIPS)


def _log(*a):
    import datetime
    print(f'[{datetime.datetime.now():%H:%M:%S}]', *a, flush=True)


def ef_metrics(preds, labels):
    p, l = np.array(preds, np.float32), np.array(labels, np.float32)
    mae = float(np.mean(np.abs(p - l)))
    rmse = float(np.sqrt(np.mean((p - l) ** 2)))
    r = float(np.corrcoef(p, l)[0, 1]) if p.std() > 0 and l.std() > 0 else 0.0
    w5 = float(np.mean(np.abs(p - l) <= 5.0)) * 100
    return mae, rmse, r, w5


def bio_metrics(preds, labels):
    mask = np.array(labels) >= 0
    if mask.sum() == 0:
        return 0.0, 0
    p = np.array(preds)[mask]; l = np.array(labels)[mask]
    return float(((p[:, 1] > p[:, 0]) == l).mean()), int(mask.sum())


# ============================ Stage 1 ======================================
def run_stage1(cfg, device):
    _log('=' * 60); _log('Stage1: EchoNet ED/ES contrastive (EchoNet-val selection)'); _log('=' * 60)
    ed_es_map = build_ed_es_map(cfg.ECHONET)
    tr_ds = EchoNetContrastive('TRAIN', ed_es_map, cfg.ECHONET, cfg.ECHO_CACHE_FULL)
    va_ds = EchoNetContrastive('VAL', ed_es_map, cfg.ECHONET, cfg.ECHO_CACHE_FULL)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.S1_BS, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=cfg.S1_BS, shuffle=False, num_workers=2, pin_memory=True)

    model = Stage1ContrastiveModel(cfg.ENCODER).to(device)
    opt = torch.optim.AdamW(model.get_param_groups(cfg.S1_BASE_LR, cfg.S1_HEAD_LR), weight_decay=cfg.S2_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.S1_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    ef_loss_fn = nn.SmoothL1Loss(beta=5.0)
    best_mae = float('inf')
    logf = os.path.join(cfg.LOGS, 'stage1.csv')
    with open(logf, 'w') as f:
        f.write('epoch,train_cos,val_cos,val_mae,val_rmse,val_r,val_w5,time_s\n')

    for ep in range(1, cfg.S1_EPOCHS + 1):
        model.train(); t0 = time.time(); total_cos, n_cos = 0.0, 0
        for uniform, ed_clip, es_clip, efs, has_annot in tr_dl:
            uniform, ed_clip, es_clip = uniform.to(device), ed_clip.to(device), es_clip.to(device)
            efs, has_annot = efs.to(device), has_annot.to(device)
            opt.zero_grad()
            with torch.autocast('cuda'):
                loss_ef = ef_loss_fn(model(uniform), efs)
                if has_annot.any():
                    mask = has_annot.bool()
                    f_ed = model.encode_one(ed_clip[mask]); f_es = model.encode_one(es_clip[mask])
                    cos = F.cosine_similarity(f_ed, f_es, dim=-1)
                    loss_c = F.relu(cos - cfg.S1_MARGIN).mean()
                    total_cos += cos.detach().mean().item() * mask.sum().item(); n_cos += mask.sum().item()
                else:
                    loss_c = torch.tensor(0.0, device=device)
                loss = loss_ef + cfg.S1_LAMBDA_C * loss_c
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        train_cos = total_cos / max(n_cos, 1)

        model.eval(); preds, labels = [], []; vcs, vcn = 0.0, 0
        with torch.no_grad():
            for uniform, ed_clip, es_clip, efs, has_annot in va_dl:
                with torch.autocast('cuda'):
                    p = model(uniform.to(device))
                preds.extend(p.cpu().tolist()); labels.extend(efs.tolist())
                if has_annot.any():
                    mask = has_annot.bool()
                    c = F.cosine_similarity(model.encode_one(ed_clip[mask].to(device)),
                                            model.encode_one(es_clip[mask].to(device)), dim=-1)
                    vcs += c.mean().item() * mask.sum().item(); vcn += mask.sum().item()
        mae, rmse, r, w5 = ef_metrics(preds, labels); val_cos = vcs / max(vcn, 1)
        flag = ' <- best' if mae < best_mae else ''
        if mae < best_mae:
            best_mae = mae
            torch.save({'epoch': ep, 'model_state': model.state_dict(), 'mae': mae, 'cosine_sim': val_cos},
                       cfg.BACKBONE_S1)
        _log(f'[S1] ep{ep:3d} | MAE={mae:.3f} RMSE={rmse:.3f} r={r:.4f} W5={w5:.1f}% '
             f'| cos(tr={train_cos:.4f}, va={val_cos:.4f}){flag} [{time.time()-t0:.0f}s]')
        with open(logf, 'a') as f:
            f.write(f'{ep},{train_cos:.4f},{val_cos:.4f},{mae:.3f},{rmse:.3f},{r:.4f},{w5:.2f},{time.time()-t0:.0f}\n')
    _log(f'Stage1 done. best EchoNet-val MAE={best_mae:.3f} -> {cfg.BACKBONE_S1}')


# ============================ Stage 2 (HONEST) =============================
def _eval_loader(model, dl, device):
    model.eval(); ep_, el_ = [], []
    with torch.no_grad():
        for a4c, a2c, efs, bios in dl:
            with torch.autocast('cuda'):
                p, _ = model(a4c.to(device), a2c.to(device))
            ep_.extend(p.cpu().tolist()); el_.extend(efs.tolist())
    return ef_metrics(ep_, el_)


def run_stage2_honest(cfg, device):
    _log('=' * 60)
    _log('Stage2: MICCAI MTL — HONEST epoch selection on patient-grouped inner-val')
    _log('(MICCAI val is logged for transparency only, NEVER used for selection)')
    _log('=' * 60)

    tr_rows, va_rows = miccai_biplane_patient_groups(cfg.LABELS, cfg.S2_SEED, cfg.S2_INNER_VAL_FRAC)
    _log(f'inner-train rows={len(tr_rows)}  inner-val rows={len(va_rows)} (patient-grouped, seed={cfg.S2_SEED})')

    common = dict(miccai_root=cfg.MICCAI, cache_tr=cfg.CACHE_TR, cache_va=cfg.CACHE_VA, labels_dir=cfg.LABELS)
    in_tr = MICCAIDatasetMTL('train', train_aug=True, rows=tr_rows, **common)
    in_va = MICCAIDatasetMTL('train', train_aug=False, rows=va_rows, **common)
    # MICCAI official val — for transparency logging only
    off_va = MICCAIDatasetMTL('val', train_aug=False, **common)

    tr_dl = DataLoader(in_tr, batch_size=cfg.S2_BS, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    inva_dl = DataLoader(in_va, batch_size=cfg.S2_BS, shuffle=False, num_workers=2, pin_memory=True)
    offva_dl = DataLoader(off_va, batch_size=cfg.S2_BS, shuffle=False, num_workers=2, pin_memory=True)

    model = Stage2MTLModel(cfg.ENCODER, cfg.BACKBONE_S1, r=cfg.S2_LORA_R).to(device)
    opt = torch.optim.AdamW(model.param_groups(cfg.S2_LORA_LR, cfg.S2_ATTN_LR, cfg.S2_HEAD_LR, cfg.S2_WD))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.S2_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    ef_loss_fn = nn.SmoothL1Loss(beta=5.0)
    bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_inner = float('inf')
    logf = os.path.join(cfg.LOGS, 'stage2.csv')
    with open(logf, 'w') as f:
        f.write('epoch,inner_val_mae,inner_val_r,official_val_mae_TRANSPARENCY_ONLY,official_val_r,time_s,saved\n')

    for ep in range(1, cfg.S2_EPOCHS + 1):
        model.train(); t0 = time.time()
        for a4c, a2c, efs, bios in tr_dl:
            a4c, a2c, efs, bios = a4c.to(device), a2c.to(device), efs.to(device), bios.to(device)
            opt.zero_grad()
            with torch.autocast('cuda'):
                ef_pred, bio_pred = model(a4c, a2c)
                loss = ef_loss_fn(ef_pred, efs) + cfg.S2_LAM * bio_loss_fn(bio_pred, bios)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()

        in_mae, _, in_r, _ = _eval_loader(model, inva_dl, device)
        off_mae, _, off_r, _ = _eval_loader(model, offva_dl, device)  # transparency only
        saved = in_mae < best_inner
        if saved:
            best_inner = in_mae
            torch.save({'epoch': ep, 'model_state': model.state_dict(),
                        'inner_val_mae': in_mae, 'selection': 'inner_val_patient_grouped'}, cfg.BACKBONE)
        _log(f'[S2] ep{ep:3d} | inner-val MAE={in_mae:.3f} r={in_r:.3f} '
             f'| [transparency] official-val MAE={off_mae:.3f} r={off_r:.3f}'
             f'{" <- saved" if saved else ""} [{time.time()-t0:.0f}s]')
        with open(logf, 'a') as f:
            f.write(f'{ep},{in_mae:.3f},{in_r:.3f},{off_mae:.3f},{off_r:.3f},{time.time()-t0:.0f},{int(saved)}\n')

    _log(f'Stage2 done. deployed backbone = best inner-val epoch (MAE={best_inner:.3f}) -> {cfg.BACKBONE}')

"""v7_turbo train.py — one-command, from-pretrained reproduction.

Pipeline (all from the pretrained EchoPrime base; honest, leakage-free):

  s1       Stage1 EchoNet ED/ES contrastive             -> weights/backbone_s1.pt
  s2       Stage2 MICCAI MTL (honest inner-val select)   -> weights/backbone.pt
  extract  appearance(my backbone)+geometry+motion+CAMUS -> cache/*.npz
  fusion   routed CAMUS-aug ridge + consensus blend +
           specialist + leakage-free sibling-anchor       -> submissions/ + weights/heads.npz
           then writes weights/infer_bundle.npz for cold-start inference

Usage:
  python train.py --steps all
  python train.py --steps s1
  python train.py --steps extract,fusion
  python train.py --steps fusion          # re-fit heads from cached features (fast)

Every hyper-parameter is selected on TRAIN out-of-fold predictions or a
patient-grouped inner-val split of MICCAI train. The MICCAI val set is used
exactly once, for the final score.
"""
import os, sys, argparse, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config as cfg


def log(*a):
    print(f'[{datetime.datetime.now():%H:%M:%S}]', *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', default='all', help="comma list of: s1,s2,extract,fusion (or 'all')")
    ap.add_argument('--force', action='store_true', help='retrain/re-extract even if outputs exist')
    args = ap.parse_args()
    steps = ['s1', 's2', 'extract', 'fusion'] if args.steps == 'all' else \
        [s.strip() for s in args.steps.split(',')]

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'device={device}  steps={steps}')

    if 's1' in steps:
        if os.path.exists(cfg.BACKBONE_S1) and not args.force:
            log(f'[s1] {cfg.BACKBONE_S1} exists, skip (use --force to retrain)')
        else:
            from echolvef.train_backbone import run_stage1
            run_stage1(cfg, device)

    if 's2' in steps:
        if os.path.exists(cfg.BACKBONE) and not args.force:
            log(f'[s2] {cfg.BACKBONE} exists, skip (use --force to retrain)')
        else:
            from echolvef.train_backbone import run_stage2_honest
            run_stage2_honest(cfg, device)

    if 'extract' in steps:
        from echolvef.extract import run_extract
        run_extract(cfg, device, force=args.force)

    if 'fusion' in steps:
        from echolvef.fusion import run_fusion
        run_fusion(cfg)
        # also (re)build the complete cold-start inference bundle
        try:
            from echolvef.heads_bundle import fit_bundle, save_bundle
            save_bundle(fit_bundle(cfg), cfg.BUNDLE)
            log(f'[fusion] wrote inference bundle -> {cfg.BUNDLE}')
        except Exception as e:
            log(f'[fusion] bundle build skipped: {e}')


if __name__ == '__main__':
    main()

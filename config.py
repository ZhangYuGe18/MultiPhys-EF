"""
EchoLVEF v7-turbo — central configuration (paths + hyper-parameters).

Self-contained re-implementation of the multi-signal LVEF pipeline for
EchoRisk-MICCAI 2026 Task 1, with a GPU-maximised / pipelined cold-start
inference path (echolvef/turbo_core.py).  No imports from other project folders.

Honesty contract (unchanged from v7-clean):
  * Stage1 backbone trained on EchoNet only (label-free w.r.t. MICCAI).
  * Stage2 backbone epoch selected on a patient-grouped INNER-VAL split of MICCAI
    *train* — the MICCAI val set is NEVER used for any model/param choice.
  * All ridge alphas, blend weights, specialist & sibling-anchor params chosen on
    TRAIN out-of-fold predictions.
  * MICCAI val labels are touched exactly once: the final score report.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'echolvef')
WEIGHTS = os.path.join(ROOT, 'weights')
CACHE = os.path.join(ROOT, 'cache')            # cached APPEARANCE/GEOM/MOTION feature npz
SUBMISSIONS = os.path.join(ROOT, 'submissions')
LOGS = os.path.join(ROOT, 'logs')
for d in (WEIGHTS, CACHE, SUBMISSIONS, LOGS):
    os.makedirs(d, exist_ok=True)

PY = '/root/miniconda3/envs/echo/bin/python'

# ----------------------------------------------------------------------------
# Read-only official / public datasets (never written to)
# ----------------------------------------------------------------------------
ECHONET = '/root/autodl-tmp/EchoNet-Dynamic'
ECHO_CACHE_FULL = '/root/autodl-tmp/echo_cache_full'        # faithful (T,112,112,3) uint8 cache
MICCAI = '/root/autodl-tmp/MICCAI'
CACHE_TR = os.path.join(MICCAI, 'cache', 'train')           # MICCAI train DICOM->npy frames (for train/extract)
CACHE_VA = os.path.join(MICCAI, 'cache', 'val')            # MICCAI val   DICOM->npy frames
LABELS = os.path.join(MICCAI, 'task1_public', 'Task1', 'public', 'labels')
GT_TRAIN = os.path.join(LABELS, 'task1_train.csv')
GT_VAL = os.path.join(LABELS, 'task1_val.csv')
CAMUS = '/root/autodl-tmp/CAMUS_public/database_nifti'
# Raw DICOMs (used by the cold-start infer — no pre-cache needed for test)
DICOM_ROOT = os.path.join(MICCAI, 'task1_public', 'Task1', 'public', 'dicom')

SCORE = os.path.join(ROOT, 'score.py')
# optional canonical scorer for an extra cross-check (skipped if absent)
SCORE_OFFICIAL = '/root/EchoPrime-Mamba/evaluation/score.py'

# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------
ENCODER = os.path.join(WEIGHTS, 'echo_prime_encoder.pt')   # [reused] EchoPrime MViT-v2-s base (Stage1 init)
BACKBONE_S1 = os.path.join(WEIGHTS, 'backbone_s1.pt')      # [trained] Stage1 ED/ES contrastive
BACKBONE = os.path.join(WEIGHTS, 'backbone.pt')            # [trained] Stage2 honest (deployed appearance encoder)
SEG_A = os.path.join(WEIGHTS, 'lv_seg_a.pt')               # [reused, label-free] ResNet-UNet (dice 0.937)
SEG_B = os.path.join(WEIGHTS, 'lv_seg_b.pt')               # [reused, label-free] ResNet-UNet (dice 0.938)
R2P1D = os.path.join(WEIGHTS, 'echonet_r2plus1d_18.pt')    # [reused, label-free]
HEADS = os.path.join(WEIGHTS, 'heads.npz')                 # [trained] merged ridge heads + fusion params
BUNDLE = os.path.join(WEIGHTS, 'infer_bundle.npz')         # [trained] complete cold-start inference bundle

# ----------------------------------------------------------------------------
# Feature caches (regenerable by extract.py)
# ----------------------------------------------------------------------------
F_APP_TRAIN = os.path.join(CACHE, 'app_train.npz')
F_APP_VAL = os.path.join(CACHE, 'app_val.npz')
F_CAMUS = os.path.join(CACHE, 'app_camus.npz')
F_GEOM_A = os.path.join(CACHE, 'geom_a.npz')
F_GEOM_B = os.path.join(CACHE, 'geom_b.npz')
F_MOTION = os.path.join(CACHE, 'motion_r2p1d.npz')

# ----------------------------------------------------------------------------
# Backbone hyper-parameters
# ----------------------------------------------------------------------------
S1_EPOCHS = 10
S1_BS = 4
S1_BASE_LR = 1e-4
S1_HEAD_LR = 1e-3
S1_MARGIN = 0.70
S1_LAMBDA_C = 2.0          # contrastive weight; <2 leaves cos-sim stuck ~0.98

S2_EPOCHS = 40
S2_BS = 4
S2_LORA_R = 8
S2_LORA_LR = 1e-4
S2_ATTN_LR = 2e-4
S2_HEAD_LR = 5e-4
S2_LAM = 0.3              # biomarker MTL weight
S2_WD = 1e-2
S2_INNER_VAL_FRAC = 0.15  # patient-grouped holdout of MICCAI train for epoch selection
S2_SEED = 0

# ----------------------------------------------------------------------------
# Fusion hyper-parameters (search grids; all selected on TRAIN OOF)
# ----------------------------------------------------------------------------
RIDGE_ALPHAS = [30, 100, 300, 1000, 3000, 10000, 30000, 100000]
BLEND_WEIGHTS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
SPEC_DELTAS = [3, 5, 7]
SPEC_WEIGHTS = [0.3, 0.5, 0.7]
PSA_MODES = ['mean', 'inv_d', 'inv_d2', 'closest']
PSA_ALPHAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_FOLDS = 5

TARGET_MAE_REFERENCE = 4.0342   # honest v7 (cold-start, no val-tuning) val MAE

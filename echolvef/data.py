"""
Data: constants, clip sampling, EchoNet (Stage1) + MICCAI (Stage2) datasets,
CAMUS loader, and MICCAI per-exam frame helpers used for feature extraction.

All preprocessing (crop/scale, MEAN/STD, clip sampling) is byte-identical to the
audited EchoPrime-Mamba pipeline so frozen features are reproduced faithfully.
"""
import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

# Normalisation + clip geometry (EchoPrime MViT)
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
CLIP_LEN = 16
N_CLIPS = 5


# ------------------------------ clip sampling ------------------------------
def extract_uniform_clips(arr, n_clips, train):
    """(T,H,W,3) -> (n_clips,3,CLIP_LEN,H,W) normalized tensor."""
    T = arr.shape[0]; clips = []
    for i in range(n_clips):
        if T >= CLIP_LEN:
            seg = max(1, (T - CLIP_LEN) // max(n_clips - 1, 1)); s0 = i * seg
            if train:
                s = np.random.randint(s0, max(s0 + 1, min(T - CLIP_LEN, s0 + seg) + 1))
            else:
                s = min(T - CLIP_LEN, s0)
            s = max(0, min(s, T - CLIP_LEN))
            clip = arr[s:s + CLIP_LEN]
        else:
            clip = np.tile(arr, (CLIP_LEN // T + 1, 1, 1, 1))[:CLIP_LEN]
        clips.append(torch.from_numpy(clip.astype(np.float32)).permute(3, 0, 1, 2).sub(MEAN).div(STD))
    return torch.stack(clips)


def extract_clip_at(arr, frame_idx):
    """CLIP_LEN frames centred on frame_idx -> (3,CLIP_LEN,H,W)."""
    T = arr.shape[0]
    s = max(0, min(frame_idx - CLIP_LEN // 2, T - CLIP_LEN))
    clip = np.tile(arr, (CLIP_LEN // T + 1, 1, 1, 1))[:CLIP_LEN] if T < CLIP_LEN else arr[s:s + CLIP_LEN]
    return torch.from_numpy(clip.astype(np.float32)).permute(3, 0, 1, 2).sub(MEAN).div(STD)


def crop_and_scale(img, res=224):
    h, w = img.shape[:2]
    if w > h:   p = (w - h) // 2; img = img[:, p:p + h]
    elif h > w: p = (h - w) // 2; img = img[p:p + w, :]
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_CUBIC)


# ------------------------------ EchoNet (Stage1) ---------------------------
def build_ed_es_map(echonet_root):
    """{filename_no_ext: (ed_frame, es_frame)} from VolumeTracings (ED=max LV diameter, ES=min)."""
    vt = pd.read_csv(os.path.join(echonet_root, 'VolumeTracings.csv'))
    vt['length'] = np.sqrt((vt['X2'] - vt['X1']) ** 2 + (vt['Y2'] - vt['Y1']) ** 2)
    fl = vt.groupby(['FileName', 'Frame'])['length'].mean().reset_index()
    ed = fl.loc[fl.groupby('FileName')['length'].idxmax(), ['FileName', 'Frame']].rename(columns={'Frame': 'ed'})
    es = fl.loc[fl.groupby('FileName')['length'].idxmin(), ['FileName', 'Frame']].rename(columns={'Frame': 'es'})
    merged = ed.merge(es, on='FileName')
    merged['key'] = merged['FileName'].str.replace('.avi', '', regex=False)
    res = {row['key']: (int(row['ed']), int(row['es'])) for _, row in merged.iterrows()}
    print(f'[ED/ES map] {len(res)} videos parsed from VolumeTracings.csv')
    return res


def read_avi(path, echo_cache_full):
    """All frames -> (T,224,224,3) float32. Prefers faithful echo_cache_full (np.load), else cv2 decode."""
    key = os.path.basename(path).replace('.avi', '')
    cf = os.path.join(echo_cache_full, key + '.npy')
    if os.path.exists(cf):
        arr = np.load(cf)  # (T,112,112,3) uint8 == cv2 frames
        return np.stack([crop_and_scale(fr).astype(np.float32) for fr in arr])
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(crop_and_scale(frame).astype(np.float32))
    cap.release()
    return np.stack(frames) if frames else np.zeros((1, 224, 224, 3), dtype=np.float32)


class EchoNetContrastive(Dataset):
    """Returns (uniform_clips, ed_clip, es_clip, ef, has_annot). Stage1 ED/ES contrastive."""
    def __init__(self, split, ed_es_map, echonet_root, echo_cache_full):
        df = pd.read_csv(os.path.join(echonet_root, 'FileList.csv'))
        df = df[df['Split'].str.upper() == split.upper()].reset_index(drop=True)
        self.root = os.path.join(echonet_root, 'Videos')
        self.echo_cache_full = echo_cache_full
        self.train = (split.upper() == 'TRAIN')
        self.ed_es_map = ed_es_map or {}
        data, n_annot = [], 0
        for _, row in df.iterrows():
            fn = str(row['FileName']); fn_key = fn.replace('.avi', '')
            if not fn.endswith('.avi'):
                fn += '.avi'
            p = os.path.join(self.root, fn)
            if os.path.exists(p) or os.path.exists(os.path.join(echo_cache_full, fn_key + '.npy')):
                has = fn_key in self.ed_es_map
                n_annot += int(has)
                data.append((p, float(row['EF']), fn_key, has))
        self.data = data
        print(f'EchoNet [{split}]: {len(data)} samples ({n_annot} with ED/ES = {100*n_annot//max(len(data),1)}%)')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, ef, key, has_annot = self.data[i]
        arr = read_avi(path, self.echo_cache_full); T = arr.shape[0]
        uniform = extract_uniform_clips(arr, N_CLIPS, self.train)
        if has_annot:
            ed_f, es_f = self.ed_es_map[key]
            ed_clip = extract_clip_at(arr, max(0, min(ed_f, T - 1)))
            es_clip = extract_clip_at(arr, max(0, min(es_f, T - 1)))
        else:
            ed_clip = torch.zeros(3, CLIP_LEN, 224, 224)
            es_clip = torch.zeros(3, CLIP_LEN, 224, 224)
        return (uniform, ed_clip, es_clip,
                torch.tensor(ef, dtype=torch.float32), torch.tensor(int(has_annot), dtype=torch.bool))


# ------------------------------ MICCAI (Stage2) ----------------------------
def _load_clips(path, train):
    arr = np.load(path).astype(np.float32)
    return extract_uniform_clips(arr, N_CLIPS, train)


class MICCAIDatasetMTL(Dataset):
    """Biplane MICCAI exams (both views present). Returns (a4c, a2c, ef, bio).

    `rows` (optional) restricts to a subset of dataframe row-indices — used to
    build the patient-grouped inner-train / inner-val split for honest epoch
    selection (the MICCAI val csv is never loaded here for selection).
    """
    def __init__(self, split, miccai_root, cache_tr, cache_va, labels_dir, train_aug=None, rows=None):
        df = pd.read_csv(os.path.join(labels_dir, f'task1_{split}.csv'))
        df = df.dropna(subset=['lvef', 'video_a4c', 'video_a2c']).reset_index(drop=True)
        if rows is not None:
            df = df.iloc[rows].reset_index(drop=True)
        cache = cache_tr if split == 'train' else cache_va
        data = []
        for _, row in df.iterrows():
            p4 = os.path.join(cache, row['video_a4c'].replace('.dcm', '.npy'))
            p2 = os.path.join(cache, row['video_a2c'].replace('.dcm', '.npy'))
            if os.path.exists(p4) and os.path.exists(p2):
                bio = row.get('biomarker_elevated', np.nan)
                bio = int(bio) if not pd.isna(bio) else -1
                data.append((p4, p2, float(row['lvef']), bio, str(row['patient_id'])))
        self.data = data
        self.train = train_aug if train_aug is not None else (split == 'train')
        print(f'MICCAI [{split}]: {len(data)} biplane samples (train_aug={self.train})')

    def patients(self):
        return np.array([d[4] for d in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        p4, p2, ef, bio, _ = self.data[i]
        return (_load_clips(p4, self.train), _load_clips(p2, self.train),
                torch.tensor(ef, dtype=torch.float32), torch.tensor(bio, dtype=torch.long))


def miccai_biplane_patient_groups(labels_dir, seed, inner_val_frac):
    """Patient-grouped inner-train / inner-val row indices over the *biplane* MICCAI train set.

    Returns (inner_train_rows, inner_val_rows) as positional indices into the
    dropna(biplane) dataframe — so MICCAIDatasetMTL(rows=...) selects them.
    """
    df = pd.read_csv(os.path.join(labels_dir, 'task1_train.csv'))
    df = df.dropna(subset=['lvef', 'video_a4c', 'video_a2c']).reset_index(drop=True)
    pids = df['patient_id'].astype(str).to_numpy()
    ups = np.unique(pids)
    rng = np.random.RandomState(seed); rng.shuffle(ups)
    n_val = max(1, int(round(len(ups) * inner_val_frac)))
    val_pat = set(ups[:n_val].tolist())
    val_rows = np.where(np.isin(pids, list(val_pat)))[0]
    tr_rows = np.where(~np.isin(pids, list(val_pat)))[0]
    return tr_rows, val_rows


# ------------------------------ CAMUS --------------------------------------
def parse_cfg(p):
    d = {}
    for line in open(p):
        if ':' in line:
            k, v = line.split(':', 1); d[k.strip()] = v.strip()
    return d


def load_camus_seq(p):
    """CAMUS nifti half-sequence -> (T,224,224,3) uint8."""
    import nibabel as nib
    a = nib.load(p).get_fdata()
    if a.ndim == 2:
        a = a[..., None]
    a = np.transpose(a, (2, 0, 1))  # (T,H,W)
    fr = np.stack([cv2.resize(np.clip(f, 0, 255).astype(np.uint8), (224, 224)) for f in a])
    return np.repeat(fr[..., None], 3, -1)


# ------------------------------ MICCAI per-exam frames ---------------------
def miccai_exam_rows(labels_dir, split):
    """All labelled exams (incl. single-view) -> list of dicts {pid,tp,y,f4,f2}."""
    df = pd.read_csv(os.path.join(labels_dir, f'task1_{split}.csv')).dropna(subset=['lvef'])
    rows = []
    for _, r in df.iterrows():
        f4 = r.video_a4c.replace('.dcm', '.npy') if isinstance(r.video_a4c, str) else None
        f2 = r.video_a2c.replace('.dcm', '.npy') if isinstance(r.video_a2c, str) else None
        rows.append(dict(pid=str(r.patient_id), tp=str(r.timepoint), y=float(r.lvef), f4=f4, f2=f2))
    return rows

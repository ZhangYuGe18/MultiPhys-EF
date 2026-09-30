"""
Model definitions (self-contained).

  * Stage1ContrastiveModel : EchoPrime MViT-v2-s + temporal attention + EF head,
    used for the EchoNet ED/ES contrastive Stage1.
  * Stage2MTLModel         : same encoder + LoRA(blocks 12-15) + biplane temporal
    attention + EF/biomarker heads, used for MICCAI Stage2.
  * UNet / ResNetUNet      : LV-cavity segmenters (label-free geometry signal).

Architecture is identical to the audited EchoPrime-Mamba backbone so the
contrastive-pretrain + frozen-feature recipe is reproduced faithfully; only the
*training/selection* code (train_backbone.py) is changed to be leakage-free.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ----------------------------- LoRA ----------------------------------------
class LoRAQV(nn.Module):
    """LoRA on the Q and V projections of an MViT attention qkv linear."""
    def __init__(self, qkv, r=8, alpha=16.0):
        super().__init__()
        self.qkv = qkv
        d = qkv.out_features // 3
        self.d, self.scale = d, alpha / r
        self.lora_q_A = nn.Linear(qkv.in_features, r, bias=False)
        self.lora_q_B = nn.Linear(r, d, bias=False)
        self.lora_v_A = nn.Linear(qkv.in_features, r, bias=False)
        self.lora_v_B = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.lora_q_A.weight, std=0.02)
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.normal_(self.lora_v_A.weight, std=0.02)
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x):
        out = self.qkv(x); d = self.d
        return torch.cat([
            out[..., :d]   + self.lora_q_B(self.lora_q_A(x)) * self.scale,
            out[..., d:2*d],
            out[..., 2*d:] + self.lora_v_B(self.lora_v_A(x)) * self.scale,
        ], dim=-1)


def inject_lora(enc, blocks, r=8, alpha=16.0):
    for p in enc.parameters():
        p.requires_grad = False
    for idx in blocks:
        enc.blocks[idx].attn.qkv = LoRAQV(enc.blocks[idx].attn.qkv, r=r, alpha=alpha)
    return sum(p.numel() for p in enc.parameters() if p.requires_grad)


# ------------------------ Temporal attention -------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, feat_dim=512, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats):  # (B, n, 512) -> (B, 512)
        weights = torch.softmax(self.net(feats), dim=1)
        return (weights * feats).sum(dim=1)


def _mvit_encoder():
    enc = torchvision.models.video.mvit_v2_s()
    enc.head[-1] = nn.Linear(enc.head[-1].in_features, 512)
    return enc


# --------------------------- Stage1 model ----------------------------------
class Stage1ContrastiveModel(nn.Module):
    def __init__(self, enc_weights, unfreeze_blocks=(12, 13, 14, 15)):
        super().__init__()
        enc = _mvit_encoder()
        enc.load_state_dict(torch.load(enc_weights, map_location='cpu'))
        for p in enc.parameters():
            p.requires_grad = False
        for idx in unfreeze_blocks:
            for p in enc.blocks[idx].parameters():
                p.requires_grad = True
        for p in enc.norm.parameters():
            p.requires_grad = True
        for p in enc.head.parameters():
            p.requires_grad = True
        self.enc = enc
        self.temp_attn = TemporalAttention(512, 128)
        self.head = nn.Linear(512, 1)
        print(f'[S1] total={sum(p.numel() for p in self.parameters()):,} '
              f'trainable={sum(p.numel() for p in self.parameters() if p.requires_grad):,}')

    def encode_one(self, clip):                 # (B,3,16,H,W) -> (B,512)
        return self.enc(clip)

    def forward(self, uniform_clips):           # (B,n,3,16,H,W) -> (B,)
        B, n = uniform_clips.shape[:2]
        feats = self.enc(uniform_clips.reshape(B * n, *uniform_clips.shape[2:])).reshape(B, n, 512)
        return self.head(self.temp_attn(feats)).squeeze(-1)

    def get_param_groups(self, base_lr, head_lr):
        return [
            {'params': self.head.parameters(),           'lr': head_lr},
            {'params': self.temp_attn.parameters(),      'lr': head_lr},
            {'params': self.enc.head.parameters(),       'lr': head_lr},
            {'params': self.enc.norm.parameters(),       'lr': base_lr},
            {'params': self.enc.blocks[15].parameters(), 'lr': base_lr},
            {'params': self.enc.blocks[14].parameters(), 'lr': base_lr},
            {'params': self.enc.blocks[13].parameters(), 'lr': base_lr * 0.5},
            {'params': self.enc.blocks[12].parameters(), 'lr': base_lr * 0.5},
        ]


# --------------------------- Stage2 model ----------------------------------
class Stage2MTLModel(nn.Module):
    """Loads EchoPrime base, overlays Stage1 fine-tuned encoder weights, injects LoRA."""
    def __init__(self, base_enc_weights, s1_ckpt, r=8):
        super().__init__()
        enc = _mvit_encoder()
        enc.load_state_dict(torch.load(base_enc_weights, map_location='cpu'))
        ck = torch.load(s1_ckpt, map_location='cpu')
        s1_state = ck['model_state'] if 'model_state' in ck else ck
        enc_state = {k[4:]: v for k, v in s1_state.items() if k.startswith('enc.')}
        enc.load_state_dict(enc_state, strict=False)
        print(f"[S2] Loaded Stage1: ep{ck.get('epoch','?')} "
              f"cos={ck.get('cosine_sim', float('nan')):.4f} mae={ck.get('mae', float('nan')):.3f}")
        temp_attn = TemporalAttention(512, 128)
        attn_state = {k[10:]: v for k, v in s1_state.items() if k.startswith('temp_attn.')}
        if attn_state:
            temp_attn.load_state_dict(attn_state)
        lora_n = inject_lora(enc, blocks=(12, 13, 14, 15), r=r)
        self.enc = enc
        self.temp_attn = temp_attn
        self.ef_head = nn.Linear(1024, 1)
        self.bio_head = nn.Linear(1024, 2)
        print(f'[S2] LoRA={lora_n:,} trainable={sum(p.numel() for p in self.parameters() if p.requires_grad):,}')

    def encode(self, clips):                    # (B,n,3,16,H,W) -> (B,512)
        B, n = clips.shape[:2]
        return self.temp_attn(self.enc(clips.reshape(B * n, *clips.shape[2:])).reshape(B, n, 512))

    def forward(self, a4c, a2c):
        f = torch.cat([self.encode(a4c), self.encode(a2c)], dim=-1)
        return self.ef_head(f).squeeze(-1), self.bio_head(f)

    def param_groups(self, lora_lr, attn_lr, head_lr, wd):
        return [
            {'params': [p for p in self.enc.parameters() if p.requires_grad], 'lr': lora_lr, 'weight_decay': wd},
            {'params': list(self.temp_attn.parameters()), 'lr': attn_lr, 'weight_decay': wd},
            {'params': list(self.ef_head.parameters()),  'lr': head_lr, 'weight_decay': wd},
            {'params': list(self.bio_head.parameters()), 'lr': head_lr, 'weight_decay': wd},
        ]


# ------------------------------ Segmenters ---------------------------------
def _norm(kind, o):
    return nn.InstanceNorm2d(o, affine=True) if kind == 'instance' else nn.BatchNorm2d(o)


class DoubleConv(nn.Module):
    def __init__(s, i, o, norm='batch'):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1, bias=False), _norm(norm, o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1, bias=False), _norm(norm, o), nn.ReLU(inplace=True))

    def forward(s, x):
        return s.net(x)


class UNet(nn.Module):
    """Compact U-Net for LV-cavity segmentation (224x224, 1ch in, 1ch logit out)."""
    def __init__(s, ch=1, base=32, ncls=1, norm='batch'):
        super().__init__()
        DC = lambda i, o: DoubleConv(i, o, norm)
        s.d1 = DC(ch, base);       s.d2 = DC(base, base*2)
        s.d3 = DC(base*2, base*4); s.d4 = DC(base*4, base*8)
        s.bott = DC(base*8, base*16); s.pool = nn.MaxPool2d(2)
        s.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2); s.c4 = DC(base*16, base*8)
        s.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2);  s.c3 = DC(base*8, base*4)
        s.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2);  s.c2 = DC(base*4, base*2)
        s.u1 = nn.ConvTranspose2d(base*2, base, 2, 2);    s.c1 = DC(base*2, base)
        s.out = nn.Conv2d(base, ncls, 1)

    def forward(s, x):
        e1 = s.d1(x); e2 = s.d2(s.pool(e1)); e3 = s.d3(s.pool(e2)); e4 = s.d4(s.pool(e3))
        b = s.bott(s.pool(e4))
        d = s.c4(torch.cat([s.u4(b), e4], 1)); d = s.c3(torch.cat([s.u3(d), e3], 1))
        d = s.c2(torch.cat([s.u2(d), e2], 1)); d = s.c1(torch.cat([s.u1(d), e1], 1))
        return s.out(d)


def _dec(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.InstanceNorm2d(o, affine=True), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.InstanceNorm2d(o, affine=True), nn.ReLU(True))


class ResNetUNet(nn.Module):
    """ResNet34-UNet (ImageNet encoder + InstanceNorm decoder) for cross-domain LV seg."""
    def __init__(s, ncls=1, pretrained=True):
        super().__init__()
        bb = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        s.inp = nn.Conv2d(1, 3, 1)
        s.e0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        s.pool = bb.maxpool
        s.e1, s.e2, s.e3, s.e4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        s.up4 = _dec(512+256, 256); s.up3 = _dec(256+128, 128); s.up2 = _dec(128+64, 64)
        s.up1 = _dec(64+64, 32);    s.up0 = _dec(32, 32)
        s.out = nn.Conv2d(32, ncls, 1); s.us = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(s, x):
        x = s.inp(x); e0 = s.e0(x); e1 = s.e1(s.pool(e0)); e2 = s.e2(e1); e3 = s.e3(e2); e4 = s.e4(e3)
        d = s.up4(torch.cat([s.us(e4), e3], 1)); d = s.up3(torch.cat([s.us(d), e2], 1))
        d = s.up2(torch.cat([s.us(d), e1], 1));  d = s.up1(torch.cat([s.us(d), e0], 1))
        return s.out(s.up0(s.us(d)))


def load_segmenter(ckpt_path, device):
    """Load a UNet or ResNetUNet segmenter from a checkpoint (auto-detects arch)."""
    ck = torch.load(ckpt_path, map_location='cpu')
    if ck.get('arch') == 'resnet':
        m = ResNetUNet(1, pretrained=False).to(device)
    else:
        m = UNet(1, 32, 1, norm='instance').to(device)
    m.load_state_dict(ck['model']); m.eval()
    return m, float(ck.get('dice', float('nan')))

"""DICOM loading + ultrasound sector mask extraction.

The mask extraction is adapted from the EchoPrime utils (`mask_outside_ultrasound`),
which works on raw YBR_FULL_422 cine loops typical of GE / Philips scanners.

End-to-end pipeline (per DICOM file):
    pydicom.read -> pixel_array (T,H,W,3 uint8)
    -> mask outside ultrasound (zero-fill EKG / annotations / UI)
    -> crop to mask bounding box (square pad)
    -> resize to (target_size, target_size)
    -> uint8 [T, target_size, target_size, 3]
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mask extraction
# ---------------------------------------------------------------------------

def mask_outside_ultrasound(pixels: np.ndarray, n_probe: int = 4) -> tuple:
    """Estimate ultrasound sector mask and zero outside.

    Optimised version: uses only `n_probe` frames to build the mask
    (was: every frame in the video → ~10x slower).  Mask is applied
    via single broadcasted multiply, not per-frame loop.

    Args:
        pixels: uint8 [T, H, W, 3] (pydicom returns RGB for YBR_FULL_422 by default).
        n_probe: number of evenly-spaced frames to use when estimating the mask.

    Returns:
        masked: uint8 [T, H, W, 3] with non-sector zeroed.
        sector_mask: uint8 [H, W] with 1 inside sector.
    """
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3] uint8, got {pixels.shape} {pixels.dtype}")

    T, H, W, _ = pixels.shape
    probe_idxs = np.linspace(0, T - 1, num=min(n_probe, T), dtype=int)
    probe = pixels[probe_idxs]                                      # [n_probe, H, W, 3]

    # Vectorised "ever non-zero in any probe frame"
    gray = probe.max(axis=-1) > 0                                   # [n_probe, H, W] bool
    frame_sum = gray.any(axis=0).astype(np.uint8)                    # [H, W]
    frame_sum = cv2.erode(frame_sum, np.ones((3, 3), np.uint8), iterations=4)

    # Difference frame (first vs last) — uses cv2.absdiff, much faster than np.abs+cvt
    diff = cv2.absdiff(probe[0].max(axis=-1).astype(np.uint8),
                        probe[-1].max(axis=-1).astype(np.uint8))
    diff = (diff > 0).astype(np.uint8)
    diff[:20, :20] = 0

    overlap = ((frame_sum + diff) > 1).astype(np.uint8)
    overlap = cv2.dilate(overlap, np.ones((3, 3), np.uint8), iterations=4)

    flood = overlap.copy()
    cv2.floodFill(flood, None, (0, 0), 100)
    flood = np.where(flood != 100, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(flood, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)
        flood[:] = 0
        cv2.drawContours(flood, [hull], -1, 255, -1)               # fill the hull
        sector_mask = (flood > 0).astype(np.uint8)
    else:
        sector_mask = np.ones((H, W), dtype=np.uint8)

    # Single broadcasted multiply over all frames (was per-frame Python loop)
    masked = pixels * sector_mask[None, ..., None]
    return masked, sector_mask


def fallback_center_crop_mask(pixels: np.ndarray, frac: float = 0.85) -> tuple:
    """Conservative fallback if `mask_outside_ultrasound` raises."""
    T, H, W = pixels.shape[:3]
    ch, cw = int(H * frac), int(W * frac)
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y0 + ch, x0:x0 + cw] = 1
    return pixels * mask[None, ..., None], mask


# ---------------------------------------------------------------------------
# Crop + resize
# ---------------------------------------------------------------------------

def crop_to_mask_bbox(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return pixels
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return pixels[:, y0:y1, x0:x1]


def resize_video(pixels: np.ndarray, size: int = 224,
                  interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    T = pixels.shape[0]
    out = np.zeros((T, size, size, pixels.shape[-1]), dtype=pixels.dtype)
    for t in range(T):
        out[t] = cv2.resize(pixels[t], (size, size), interpolation=interpolation)
    return out


# ---------------------------------------------------------------------------
# Top-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_dicom_pixels(pixels: np.ndarray,
                              target_size: int = 224,
                              robust_mask: bool = True) -> np.ndarray:
    """pixel_array → uint8 [T, target_size, target_size, 3]."""
    if pixels.ndim == 3:                                    # [T, H, W] grayscale
        pixels = np.repeat(pixels[..., None], 3, axis=-1)
    if pixels.dtype != np.uint8:
        # Some DICOMs store 12/16-bit; rescale to uint8
        lo, hi = float(pixels.min()), float(pixels.max())
        pixels = ((pixels.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)

    if robust_mask:
        try:
            masked, mask = mask_outside_ultrasound(pixels)
        except Exception:
            masked, mask = fallback_center_crop_mask(pixels)
    else:
        masked, mask = fallback_center_crop_mask(pixels)

    cropped = crop_to_mask_bbox(masked, mask)
    out = resize_video(cropped, size=target_size)
    return out


def load_and_preprocess_dicom(path: str, target_size: int = 224,
                                robust_mask: bool = True) -> np.ndarray:
    """Lazy pydicom import to keep this module CPU-test-friendly."""
    import pydicom
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    return preprocess_dicom_pixels(pixels, target_size=target_size, robust_mask=robust_mask)

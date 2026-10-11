"""Mixup of training examples (Zhang et al. 2018), added 2026-09-26.

Each example of a batch is mixed with a partner from the same batch (a random permutation, so an
example is sometimes its own partner): input = w * own + (1 - w) * partner, and every target the
loss uses (emotion, and gender when that head is on) is mixed with the same weight. Each example
draws its own w from Beta(alpha, alpha). Mixing happens on the normalised, masked log-mel, after
SpecAugment; the mixed example counts the frames of the longer of the two, since padding is 0.
The draws come from the job's augmentation stream (numpy), so they follow the seed.
"""
import numpy as np


def draw(alpha: float, n: int, rng: np.random.Generator) -> tuple:
    """(weights [n] float32 from Beta(alpha, alpha), partner of each example [n])."""
    if alpha <= 0:
        raise ValueError("mixup needs alpha above 0")
    weights = rng.beta(alpha, alpha, size=n).astype(np.float32)
    partner = rng.permutation(n)
    return weights, partner


def mix(x, frames, targets: list, weights: np.ndarray, partner: np.ndarray) -> tuple:
    """x [batch, bands, frames] and frames [batch] on the device; targets: [batch, k] tensors.
    Returns (mixed x, frames of the longer example, mixed targets in the same order)."""
    import torch

    w = torch.as_tensor(weights, device=x.device)
    index = torch.as_tensor(partner, device=x.device, dtype=torch.long)
    wx = w.to(x.dtype)[:, None, None]
    mixed = wx * x + (1.0 - wx) * x[index]
    mixed_frames = torch.maximum(frames, frames[index])
    mixed_targets = []
    for t in targets:
        wt = w.to(t.dtype)[:, None]
        mixed_targets.append(wt * t + (1.0 - wt) * t[index])
    return mixed, mixed_frames, mixed_targets

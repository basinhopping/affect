"""SpecAugment-style masking of the normalised log-mel, training batches only.

Each example gets `freq_masks` bands of up to `freq_mask_width` mel bands and `time_masks` spans
of up to `time_mask_width` frames set to 0 (the mean, after normalisation). Time masks fall inside
the utterance's own frames, never in its padding. Zero masks means no augmentation. The settings
live in affect.augment.settings with the other augmentations.
"""
from affect.augment.settings import AugmentSettings


def _band_mask(n_masks: int, max_width: int, limit, size: int, generator):
    """[batch, size] bool: n_masks spans of width 0..max_width starting inside [0, limit - width]."""
    import torch

    batch = limit.shape[0]
    widths = torch.randint(0, max_width + 1, (batch, n_masks), generator=generator)
    room = (limit[:, None] - widths).clamp_min(0)
    starts = (torch.rand(batch, n_masks, generator=generator) * (room + 1)).floor().long()
    position = torch.arange(size)[None, None, :]
    inside = (position >= starts[..., None]) & (position < (starts + widths)[..., None])
    return inside.any(dim=1)


def spec_augment(x, valid_frames, settings: AugmentSettings, generator):
    """x: [batch, n_mels, frames] on any device; valid_frames: [batch] frame counts.
    generator: a CPU torch.Generator, so the masks follow the job's seed."""
    if settings.freq_masks == 0 and settings.time_masks == 0:
        return x
    import torch

    batch, n_mels, frames = x.shape
    masked = x
    if settings.freq_masks > 0 and settings.freq_mask_width > 0:
        limit = torch.full((batch,), n_mels, dtype=torch.long)
        band = _band_mask(settings.freq_masks, settings.freq_mask_width, limit, n_mels, generator)
        masked = masked.masked_fill(band.to(x.device)[:, :, None], 0.0)
    if settings.time_masks > 0 and settings.time_mask_width > 0:
        limit = valid_frames.detach().to("cpu").long().clamp_max(frames)
        span = _band_mask(settings.time_masks, settings.time_mask_width, limit, frames, generator)
        masked = masked.masked_fill(span.to(x.device)[:, None, :], 0.0)
    return masked

"""Augmentation settings: every augmentation a training batch can get, one dataclass (plan 5.3).

Order inside a training step: speed perturbation on the waveform (affect.augment.speed), then
features and normalisation, then SpecAugment masks (affect.augment.specaugment), then mixup
(affect.augment.mixup). Validation and test data are never augmented.

speed_factors and mixup_alpha were added on 2026-09-26. Settings written before then hold only the
four SpecAugment fields, which means no speed perturbation (speed_factors [1.0]) and no mixup
(mixup_alpha 0), and they keep their configuration hashes (affect.settings, ADDED_FIELDS).
"""
import dataclasses

from affect.augment.speed import speed_ratio


@dataclasses.dataclass(frozen=True)
class AugmentSettings:
    freq_masks: int          # frequency masks per training example (0 for none)
    freq_mask_width: int     # widest frequency mask, in mel bands
    time_masks: int          # time masks per training example (0 for none)
    time_mask_width: int     # widest time mask, in frames
    speed_factors: tuple[float, ...]
    # speed perturbation: every time a training example is used, one of these speeds is drawn with
    # equal probability and the waveform is resampled by it (1.1 is 10% faster and higher, as sox
    # "speed"; Aldeneh 2017 and Latif 2019 used 0.9, 1.0, 1.1). [1.0] for none.
    mixup_alpha: float
    # mixup: each training example is mixed with another example of its batch, inputs and targets
    # alike, with its own weight drawn from Beta(alpha, alpha) (Zhang et al. 2018). 0 for none.

    ADDED_FIELDS = (("2026-09-26", {"speed_factors": [1.0], "mixup_alpha": 0.0}),)


def check(settings: AugmentSettings, n_mels: int) -> None:
    for name in ("freq_masks", "freq_mask_width", "time_masks", "time_mask_width"):
        if getattr(settings, name) < 0:
            raise ValueError(f"augment.{name} must not be negative")
    if settings.freq_mask_width > n_mels:
        raise ValueError(f"augment.freq_mask_width {settings.freq_mask_width} exceeds n_mels {n_mels}")
    if not settings.speed_factors:
        raise ValueError("augment.speed_factors must list at least one speed ([1.0] for none)")
    if len(set(settings.speed_factors)) != len(settings.speed_factors):
        raise ValueError(f"augment.speed_factors repeat: {list(settings.speed_factors)}")
    for factor in settings.speed_factors:
        if not 0.5 <= factor <= 2.0:
            raise ValueError(f"augment.speed_factors: {factor} is outside 0.5 to 2.0")
        speed_ratio(factor)          # raises unless the speed is a ratio of small whole numbers
    if settings.mixup_alpha < 0:
        raise ValueError("augment.mixup_alpha must not be negative (0 for no mixup)")

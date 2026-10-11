"""CNN over the log-mel image (plan 4.2).

Blocks of [convolution, normalisation, activation] x convs_per_block, then max pooling over
frequency and time. After the last block the frequency axis is averaged, the time axis is
summarised by `pooling` over the utterance's own frames only (padding is masked out), and a
classifier gives one logit per class.

norm, norm_groups, norm_subbands and activation were added on 2026-09-26 (Dr. Luitel asked about
batch normalization). Settings written before then lack them, which means batch norm and ReLU,
the only layers the CNN had; those settings keep their configuration hashes (affect.settings,
ADDED_FIELDS), and the network they build is layer for layer the one built before, so the same
seed gives the same weights (a node test compares both). The front end here (conv_front_end) is
shared with the CNN + Transformer model.
"""
import dataclasses
from typing import Literal

from affect.registry import register_model

NORMS = ("batch", "subspectral", "group", "none")
ACTIVATIONS = ("relu", "leaky_relu", "elu", "gelu", "silu")


@dataclasses.dataclass(frozen=True)
class CNNSettings:
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + normalisation + activation layers per block
    kernel_size: int              # square kernel, odd
    pool_freq: int                # max-pool factor along frequency after each block
    pool_time: int                # max-pool factor along time after each block
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]
    # how the time axis is summarised: mean, max, both concatenated, mean and SD, or learned attention
    classifier_hidden: int | None  # width of one hidden layer before the output layer, or None for none
    dropout: float                # dropout before the classifier's layers
    norm: Literal["batch", "subspectral", "group", "none"]
    # normalisation after each convolution. batch: batch norm (Ioffe and Szegedy 2015). subspectral:
    # batch norm computed separately in norm_subbands frequency bands (Chang et al. 2021). group:
    # group norm with norm_groups groups per layer (Wu and He 2018; 1 group is layer norm). none: no
    # normalisation, and the convolutions get a bias instead.
    norm_groups: int | None       # groups for norm "group", dividing every block's channels; None otherwise
    norm_subbands: int | None     # frequency bands for norm "subspectral", at least 2; None otherwise
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]
    # after every normalisation and in the classifier's hidden layer. leaky_relu has slope 0.01 below
    # zero and elu has alpha 1 (the PyTorch values); gelu is the exact (erf) form.

    ADDED_FIELDS = (("2026-09-26", {"norm": "batch", "norm_groups": None, "norm_subbands": None,
                                    "activation": "relu"}),)


def check_front_end(settings, n_mels: int) -> None:
    """Checks of the convolutional front end, shared by every model that has one."""
    if not settings.channels or any(c < 1 for c in settings.channels):
        raise ValueError("model.channels must be a non-empty list of positive numbers")
    if settings.convs_per_block < 1:
        raise ValueError("model.convs_per_block must be at least 1")
    if settings.kernel_size < 1 or settings.kernel_size % 2 == 0:
        raise ValueError("model.kernel_size must be odd")
    if settings.pool_freq < 1 or settings.pool_time < 1:
        raise ValueError("model.pool_freq and pool_time must be at least 1")
    bands = n_mels
    for _ in settings.channels:
        bands //= settings.pool_freq
    if bands < 1:
        raise ValueError(f"model: {len(settings.channels)} blocks pooling frequency by {settings.pool_freq} "
                         f"leave no bands of {n_mels}")
    if settings.norm not in NORMS or settings.activation not in ACTIVATIONS:
        raise ValueError(f"model: norm must be one of {NORMS} and activation one of {ACTIVATIONS}")
    if settings.norm == "group":
        if settings.norm_groups is None or settings.norm_groups < 1 or settings.norm_subbands is not None:
            raise ValueError("model: norm 'group' needs norm_groups of at least 1 and norm_subbands None")
        bad = [c for c in settings.channels if c % settings.norm_groups]
        if bad:
            raise ValueError(f"model: norm_groups {settings.norm_groups} does not divide channels {bad}")
    elif settings.norm == "subspectral":
        if settings.norm_subbands is None or settings.norm_subbands < 2 or settings.norm_groups is not None:
            raise ValueError("model: norm 'subspectral' needs norm_subbands of at least 2 and norm_groups None")
        for block in range(len(settings.channels)):
            bands_here = n_mels // settings.pool_freq ** block
            if bands_here < settings.norm_subbands:
                raise ValueError(f"model: block {block + 1} sees {bands_here} frequency bands, fewer than "
                                 f"norm_subbands {settings.norm_subbands}")
    elif settings.norm_groups is not None or settings.norm_subbands is not None:
        raise ValueError(f"model: norm {settings.norm!r} takes norm_groups and norm_subbands None")


def check_head(settings) -> None:
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if settings.classifier_hidden is not None and settings.classifier_hidden < 1:
        raise ValueError("model.classifier_hidden must be positive or None")


def check(settings: CNNSettings, n_mels: int) -> None:
    check_front_end(settings, n_mels)
    check_head(settings)


def masked_pool(h, mask, kind: str, attention):
    """h: [batch, channels, frames] float32; mask: [batch, frames] bool, True on real frames."""
    import torch

    m = mask[:, None, :].to(h.dtype)
    count = m.sum(dim=-1).clamp_min(1.0)
    mean = (h * m).sum(dim=-1) / count
    if kind == "mean":
        return mean
    lowest = torch.finfo(h.dtype).min
    top = h.masked_fill(~mask[:, None, :], lowest).amax(dim=-1)
    if kind == "max":
        return top
    if kind == "meanmax":
        return torch.cat([mean, top], dim=1)
    if kind == "stats":
        var = (((h - mean[..., None]) ** 2) * m).sum(dim=-1) / count
        return torch.cat([mean, torch.sqrt(var + 1e-6)], dim=1)
    if kind == "attention":
        scores = attention(h.transpose(1, 2)).squeeze(-1).float()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(h.dtype)
        return (h * weights[:, None, :]).sum(dim=-1)
    raise ValueError(kind)


def pooled_width(width: int, pooling: str) -> int:
    return 2 * width if pooling in ("meanmax", "stats") else width


def activation_layer(name: str):
    from torch import nn

    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if name == "elu":
        return nn.ELU(alpha=1.0)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name!r}")


def _subspectral_class():
    import torch
    from torch import nn

    class SubSpectralNorm(nn.Module):
        """Batch norm computed separately in each of `subbands` frequency bands (Chang et al. 2021).
        Input [batch, channels, frequency, time]; torch.tensor_split cuts frequency as evenly as
        possible (a band count that does not divide it gives some bands one row more)."""

        def __init__(self, channels: int, subbands: int):
            super().__init__()
            self.subbands = subbands
            self.norms = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(subbands))

        def forward(self, x):
            parts = torch.tensor_split(x, self.subbands, dim=2)
            return torch.cat([norm(part) for norm, part in zip(self.norms, parts)], dim=2)

    return SubSpectralNorm


def conv_front_end(settings, n_mels: int):
    """nn.Sequential of the convolution blocks (n_mels is checked by check_front_end, not needed
    here). With norm "batch" and activation "relu" the layers and their order are exactly those
    of the CNN before 2026-09-26: Conv2d without bias, BatchNorm2d, ReLU(inplace) per convolution,
    then MaxPool2d per block."""
    from torch import nn

    subspectral = _subspectral_class() if settings.norm == "subspectral" else None
    layers, c_in = [], 1
    for c_out in settings.channels:
        for _ in range(settings.convs_per_block):
            layers.append(nn.Conv2d(c_in, c_out, settings.kernel_size, padding=settings.kernel_size // 2,
                                    bias=settings.norm == "none"))
            if settings.norm == "batch":
                layers.append(nn.BatchNorm2d(c_out))
            elif settings.norm == "subspectral":
                layers.append(subspectral(c_out, settings.norm_subbands))
            elif settings.norm == "group":
                layers.append(nn.GroupNorm(settings.norm_groups, c_out))
            layers.append(activation_layer(settings.activation))
            c_in = c_out
        layers.append(nn.MaxPool2d((settings.pool_freq, settings.pool_time)))
    return nn.Sequential(*layers)


def classifier_head(settings, width: int, n_classes: int):
    """Dropout, an optional hidden layer with the model's activation, and the output layer."""
    from torch import nn

    head = [nn.Dropout(settings.dropout)]
    if settings.classifier_hidden is not None:
        head += [nn.Linear(width, settings.classifier_hidden), activation_layer(settings.activation),
                 nn.Dropout(settings.dropout)]
        width = settings.classifier_hidden
    head.append(nn.Linear(width, n_classes))
    return nn.Sequential(*head)


def frames_after(valid_frames, n_blocks: int, pool_time: int):
    """Real frames left after n_blocks max poolings along time (at least 1)."""
    import torch

    frames = valid_frames
    for _ in range(n_blocks):
        frames = torch.div(frames, pool_time, rounding_mode="floor")
    return frames.clamp_min(1)


def _module_class():
    import torch
    from torch import nn

    class CNN(nn.Module):
        def __init__(self, settings: CNNSettings, n_mels: int, n_classes: int):
            super().__init__()
            check(settings, n_mels)
            self.blocks = conv_front_end(settings, n_mels)
            self.n_blocks = len(settings.channels)
            self.pool_time = settings.pool_time
            self.pooling = settings.pooling
            width = settings.channels[-1]
            self.attention = nn.Linear(width, 1) if settings.pooling == "attention" else None
            width = pooled_width(width, settings.pooling)
            self.embedding_dim = width            # size of embed()'s output, for extra heads
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, width, n_classes)

        def frames_after_blocks(self, valid_frames):
            return frames_after(valid_frames, self.n_blocks, self.pool_time)

        def embed(self, x, valid_frames):
            """x: [batch, n_mels, frames]; valid_frames: [batch]. The pooled summary [batch, embedding_dim]."""
            h = self.blocks(x.unsqueeze(1))           # [batch, channels, bands, frames']
            h = h.float().mean(dim=2)                  # [batch, channels, frames']
            frames = h.shape[-1]
            keep = self.frames_after_blocks(valid_frames).clamp_max(frames)
            mask = torch.arange(frames, device=h.device)[None, :] < keep[:, None]
            return masked_pool(h, mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            """x: [batch, n_mels, frames]; valid_frames: [batch] real frames before padding."""
            return self.classify(self.embed(x, valid_frames))

    return CNN


@register_model("cnn", settings=CNNSettings)
def build_cnn(settings: CNNSettings, n_mels: int, n_classes: int):
    return _module_class()(settings, n_mels, n_classes)

"""CNN + Transformer over the log-mel image (model family 2, added 2026-09-26).

The convolutional front end is the CNN's (affect.models.cnn.conv_front_end: blocks of
convolution, normalisation, activation, then max pooling). Its output is turned into one vector
per remaining time step: the frequency axis is averaged (freq_reduce "mean") or the channels of
every remaining frequency band are concatenated ("flatten"), and a linear layer maps that to
d_model. A Transformer encoder (Vaswani et al. 2017; torch.nn.TransformerEncoderLayer) then
attends across time; padded time steps are masked out of the attention and of the pooling. The
pooling and classifier are the CNN's. Every size is a setting; nothing is fixed in the code
except the sinusoidal position code's base of 10,000 (Vaswani et al.).
"""
import dataclasses
import math
from typing import Literal

from affect.models.cnn import (check_front_end, check_head, classifier_head, conv_front_end, frames_after,
                               masked_pool, pooled_width)
from affect.registry import register_model


@dataclasses.dataclass(frozen=True)
class CNNTransformerSettings:
    # convolutional front end, as in the CNN (affect.models.cnn.CNNSettings)
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + normalisation + activation layers per block
    kernel_size: int              # square kernel, odd
    pool_freq: int                # max-pool factor along frequency after each block
    pool_time: int                # max-pool factor along time after each block (the Transformer sees frames / pool_time ** blocks)
    norm: Literal["batch", "subspectral", "group", "none"]   # as in the CNN
    norm_groups: int | None       # groups for norm "group"; None otherwise
    norm_subbands: int | None     # frequency bands for norm "subspectral"; None otherwise
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # front end and classifier
    # from the front end to the Transformer
    freq_reduce: Literal["mean", "flatten"]
    # mean: average the remaining frequency bands (one value per channel); flatten: keep every
    # channel of every remaining band. Either way a linear layer maps the result to d_model.
    # Transformer encoder
    d_model: int                  # width of the Transformer
    n_heads: int                  # attention heads; must divide d_model
    n_layers: int                 # encoder layers
    ff_dim: int                   # width of each layer's feed-forward block
    transformer_dropout: float    # dropout on attention weights, the feed-forward block and each residual branch
    transformer_activation: Literal["relu", "gelu"]   # inside the feed-forward block
    norm_first: bool              # True: layer norm before attention and feed-forward (pre-LN, with a final layer norm); False: after (the original Transformer)
    positional: Literal["sinusoidal", "none"]
    # sinusoidal: the fixed sine and cosine position code of Vaswani et al., added to the inputs;
    # none: no position code (the convolutions already see local order)
    # summary and classifier, as in the CNN
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]
    classifier_hidden: int | None  # width of one hidden layer before the output layer, or None
    dropout: float                # dropout before the classifier's layers


def check(settings: CNNTransformerSettings, n_mels: int) -> None:
    check_front_end(settings, n_mels)
    check_head(settings)
    if settings.d_model < 1 or settings.n_heads < 1 or settings.d_model % settings.n_heads:
        raise ValueError(f"model: n_heads {settings.n_heads} must divide d_model {settings.d_model}")
    if settings.n_layers < 1 or settings.ff_dim < 1:
        raise ValueError("model: n_layers and ff_dim must be at least 1")
    if not 0.0 <= settings.transformer_dropout < 1.0:
        raise ValueError("model.transformer_dropout must be in [0, 1)")
    if settings.positional == "sinusoidal" and settings.d_model % 2:
        raise ValueError("model: the sinusoidal position code needs an even d_model")


def bands_after(settings, n_mels: int) -> int:
    bands = n_mels
    for _ in settings.channels:
        bands //= settings.pool_freq
    return bands


def sinusoidal(length: int, width: int, device, dtype):
    """[length, width]: sin at even columns and cos at odd, frequencies 1 / 10000^(2i / width)."""
    import torch

    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    rate = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / width))
    code = torch.zeros(length, width, device=device, dtype=torch.float32)
    code[:, 0::2] = torch.sin(position * rate)
    code[:, 1::2] = torch.cos(position * rate)
    return code.to(dtype)


def _module_class():
    import torch
    from torch import nn

    class CNNTransformer(nn.Module):
        def __init__(self, settings: CNNTransformerSettings, n_mels: int, n_classes: int):
            super().__init__()
            check(settings, n_mels)
            self.front = conv_front_end(settings, n_mels)
            self.n_blocks = len(settings.channels)
            self.pool_time = settings.pool_time
            self.freq_reduce = settings.freq_reduce
            width_in = settings.channels[-1]
            if settings.freq_reduce == "flatten":
                width_in *= bands_after(settings, n_mels)
            self.project = nn.Linear(width_in, settings.d_model)
            layer = nn.TransformerEncoderLayer(
                settings.d_model, settings.n_heads, dim_feedforward=settings.ff_dim,
                dropout=settings.transformer_dropout, activation=settings.transformer_activation,
                batch_first=True, norm_first=settings.norm_first)
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=settings.n_layers,
                norm=nn.LayerNorm(settings.d_model) if settings.norm_first else None,
                enable_nested_tensor=False)
            self.positional = settings.positional
            self.pooling = settings.pooling
            self.attention = nn.Linear(settings.d_model, 1) if settings.pooling == "attention" else None
            width = pooled_width(settings.d_model, settings.pooling)
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, width, n_classes)

        def frames_after_blocks(self, valid_frames):
            return frames_after(valid_frames, self.n_blocks, self.pool_time)

        def embed(self, x, valid_frames):
            """x: [batch, n_mels, frames]; valid_frames: [batch]. The pooled summary [batch, embedding_dim]."""
            h = self.front(x.unsqueeze(1))                        # [batch, channels, bands, steps]
            if self.freq_reduce == "mean":
                h = h.float().mean(dim=2).transpose(1, 2)        # [batch, steps, channels]
            else:
                b, c, f, t = h.shape
                h = h.float().permute(0, 3, 1, 2).reshape(b, t, c * f)
            h = self.project(h)                                   # [batch, steps, d_model]
            steps = h.shape[1]
            keep = self.frames_after_blocks(valid_frames).clamp_max(steps)
            mask = torch.arange(steps, device=h.device)[None, :] < keep[:, None]
            if self.positional == "sinusoidal":
                h = h + sinusoidal(steps, h.shape[-1], h.device, h.dtype)
            h = self.encoder(h, src_key_padding_mask=~mask)
            return masked_pool(h.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            return self.classify(self.embed(x, valid_frames))

    return CNNTransformer


@register_model("cnn_transformer", settings=CNNTransformerSettings)
def build_cnn_transformer(settings: CNNTransformerSettings, n_mels: int, n_classes: int):
    return _module_class()(settings, n_mels, n_classes)

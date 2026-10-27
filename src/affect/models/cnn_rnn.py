"""CNN + recurrent network over the log-mel image (a from-scratch family, added 2026-10-24; Harrison asked on
2026-10-23 for other combinations trained from scratch).

The convolutional front end is the CNN's (affect.models.cnn.conv_front_end). Its output is turned into one vector
per remaining time step as in the CNN + Transformer (the frequency axis averaged, freq_reduce "mean", or the channels
of every remaining band concatenated, "flatten"), and a recurrent network reads those steps in order: a GRU (Cho et
al. 2014) or an LSTM (Hochreiter and Schmidhuber 1997), one to four layers, in one direction or both. Each utterance
is packed to its own real steps (torch.nn.utils.rnn.pack_padded_sequence), so padding never enters the recurrence in
either direction. The recurrent layers run in float32 outside autocast. The summary is the CNN's pooling over the
real steps (mean, max, both, mean and SD, or learned attention) or "last": the last layer's final state of each
direction (the forward direction at the utterance's last real step, the backward one at its first). The classifier
is the CNN's. Every size is a setting.
"""
import dataclasses
from typing import Literal

from affect.models.cnn import (check_front_end, check_head, classifier_head, conv_front_end, frames_after,
                               masked_pool, pooled_width)
from affect.models.cnn_transformer import bands_after
from affect.registry import register_model


@dataclasses.dataclass(frozen=True)
class CNNRNNSettings:
    # convolutional front end, as in the CNN (affect.models.cnn.CNNSettings)
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + normalisation + activation layers per block
    kernel_size: int              # square kernel, odd
    pool_freq: int                # max-pool factor along frequency after each block
    pool_time: int                # max-pool factor along time after each block (frames / pool_time ** blocks remain)
    norm: Literal["batch", "subspectral", "group", "none"]   # as in the CNN
    norm_groups: int | None       # groups for norm "group"; None otherwise
    norm_subbands: int | None     # frequency bands for norm "subspectral"; None otherwise
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # front end and classifier
    # from the front end to the recurrent network, as in the CNN + Transformer
    freq_reduce: Literal["mean", "flatten"]
    # recurrent network
    rnn: Literal["gru", "lstm"]
    rnn_hidden: int               # units per direction in each layer
    rnn_layers: int               # stacked layers, 1 to 4
    bidirectional: bool           # True: a second direction reads each utterance backwards (outputs concatenated)
    rnn_dropout: float            # dropout between stacked layers (PyTorch's dropout argument); 0 with one layer
    # summary and classifier
    pooling: Literal["mean", "max", "meanmax", "stats", "attention", "last"]
    classifier_hidden: int | None  # width of one hidden layer before the output layer, or None
    dropout: float                # dropout before the classifier's layers


def check(settings: CNNRNNSettings, n_mels: int) -> None:
    check_front_end(settings, n_mels)
    check_head(settings)
    if settings.rnn_hidden < 1:
        raise ValueError("model.rnn_hidden must be at least 1")
    if not 1 <= settings.rnn_layers <= 4:
        raise ValueError("model.rnn_layers must be 1 to 4")
    if not 0.0 <= settings.rnn_dropout < 1.0:
        raise ValueError("model.rnn_dropout must be in [0, 1)")
    if settings.rnn_layers == 1 and settings.rnn_dropout > 0:
        raise ValueError("model.rnn_dropout acts between stacked layers, so it must be 0 with one layer")


def rnn_width(settings: CNNRNNSettings) -> int:
    """Width of the recurrent network's output at each step (both directions concatenated)."""
    return settings.rnn_hidden * (2 if settings.bidirectional else 1)


def _module_class():
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class CNNRNN(nn.Module):
        def __init__(self, settings: CNNRNNSettings, n_mels: int, n_classes: int):
            super().__init__()
            check(settings, n_mels)
            self.front = conv_front_end(settings, n_mels)
            self.n_blocks = len(settings.channels)
            self.pool_time = settings.pool_time
            self.freq_reduce = settings.freq_reduce
            width_in = settings.channels[-1]
            if settings.freq_reduce == "flatten":
                width_in *= bands_after(settings, n_mels)
            recurrent = nn.GRU if settings.rnn == "gru" else nn.LSTM
            self.rnn = recurrent(width_in, settings.rnn_hidden, num_layers=settings.rnn_layers, batch_first=True,
                                 dropout=settings.rnn_dropout, bidirectional=settings.bidirectional)
            self.rnn_kind, self.rnn_layers = settings.rnn, settings.rnn_layers
            self.directions = 2 if settings.bidirectional else 1
            self.rnn_hidden = settings.rnn_hidden
            width = rnn_width(settings)
            self.pooling = settings.pooling
            self.attention = nn.Linear(width, 1) if settings.pooling == "attention" else None
            width = width if settings.pooling == "last" else pooled_width(width, settings.pooling)
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, width, n_classes)

        def frames_after_blocks(self, valid_frames):
            return frames_after(valid_frames, self.n_blocks, self.pool_time)

        def sequence(self, h, keep):
            """h: [batch, steps, width] float32, the front end's steps; keep: [batch] real steps (1 to steps). The
            pooled summary [batch, embedding_dim] of the recurrent network's outputs over each row's real steps."""
            batch, steps = h.shape[0], h.shape[1]
            with torch.autocast(device_type=h.device.type, enabled=False):
                packed = pack_padded_sequence(h.float(), keep.detach().cpu().to(torch.int64), batch_first=True,
                                              enforce_sorted=False)
                out, state = self.rnn(packed)
                if self.pooling == "last":
                    final = state[0] if self.rnn_kind == "lstm" else state   # [layers * directions, batch, hidden]
                    final = final.view(self.rnn_layers, self.directions, batch, self.rnn_hidden)[-1]
                    return final.transpose(0, 1).reshape(batch, self.directions * self.rnn_hidden)
                out, _ = pad_packed_sequence(out, batch_first=True, total_length=steps)   # [batch, steps, width]
            mask = torch.arange(steps, device=h.device)[None, :] < keep[:, None]
            return masked_pool(out.float().transpose(1, 2), mask, self.pooling, self.attention)

        def embed(self, x, valid_frames):
            """x: [batch, n_mels, frames]; valid_frames: [batch]. The pooled summary [batch, embedding_dim]."""
            h = self.front(x.unsqueeze(1))                        # [batch, channels, bands, steps]
            if self.freq_reduce == "mean":
                h = h.float().mean(dim=2).transpose(1, 2)        # [batch, steps, channels]
            else:
                b, c, f, t = h.shape
                h = h.float().permute(0, 3, 1, 2).reshape(b, t, c * f)
            keep = self.frames_after_blocks(valid_frames).clamp_max(h.shape[1])
            return self.sequence(h, keep)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            return self.classify(self.embed(x, valid_frames))

    return CNNRNN


@register_model("cnn_rnn", settings=CNNRNNSettings)
def build_cnn_rnn(settings: CNNRNNSettings, n_mels: int, n_classes: int):
    return _module_class()(settings, n_mels, n_classes)

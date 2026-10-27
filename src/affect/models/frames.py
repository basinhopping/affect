"""Heads on a frozen pretrained encoder's frames (added 2026-10-26; features "pretrained_frames", affect.features.frames).

Dr. Luitel, 2026-10-21: "try my own lif/transformer on top of the pretrained, well lets do a sweep and see what is
best". Each head reads one hidden layer of the encoder, every frame (WavLM Large: 1,024 values, 50 per second), and
is trained with the training loop like every other model; the encoder itself never changes. All four share the input
stage: the frames [batch, width, frames] as the loop delivers them (normalized, time-masked), then consecutive frames
averaged in groups of time_pool (real frames only), dropout (input_dropout), and an optional linear projection to
proj_dim with the activation. Then:

    frames_pool         the projected frames summarized over the utterance's own frames (mean, max, both, mean and SD,
                        or learned attention: attentive pooling) and the CNN's classifier: the simplest head, the trained
                        counterpart of the SVM on each layer's mean frame
    frames_transformer  a linear layer to d_model, an optional sinusoidal position code, a Transformer encoder (as in the
                        CNN + Transformer, padded frames masked out of the attention), the pooling and the classifier
    frames_lif          spiking layers on the projected frames, as in the CNN + LIF (affect.models.lif: linear synapses,
                        recurrence when asked, LIF neurons, a leaky readout over the utterance's own steps, the activity
                        penalty); the readout feeds dropout and a linear layer
    frames_rnn          a GRU or an LSTM packed to each utterance's own frames (as in model cnn_rnn), every pooling or
                        the last state, and the classifier

Each model is built with (settings, n_inputs, n_classes), n_inputs the frame width (runs.spec.input_width).
"""
import dataclasses
import math
from typing import Literal

from affect.models.cnn import check_head, classifier_head, masked_pool, pooled_width
from affect.models.cnn_transformer import sinusoidal
from affect.models.lif import check_neuron
from affect.registry import register_model


@dataclasses.dataclass(frozen=True)
class FramesPoolSettings:
    time_pool: int                 # average consecutive frames in groups of this size first (1: none; 2: 25 per second)
    input_dropout: float           # dropout on the frames, before the projection
    proj_dim: int | None           # a linear layer to this width with the activation, or None (the frames as they are)
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # projection and classifier
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]   # attention: attentive pooling
    classifier_hidden: int | None  # width of one hidden layer before the output layer, or None
    dropout: float                 # dropout before the classifier's layers


@dataclasses.dataclass(frozen=True)
class FramesTransformerSettings:
    time_pool: int
    input_dropout: float
    d_model: int                   # the linear layer from the frames to the Transformer's width
    n_heads: int                   # attention heads; must divide d_model
    n_layers: int                  # encoder layers
    ff_dim: int                    # width of each layer's feed-forward block
    transformer_dropout: float     # dropout on attention weights, the feed-forward block and each residual branch
    transformer_activation: Literal["relu", "gelu"]
    norm_first: bool               # pre-LN (with a final layer norm) or post-LN
    positional: Literal["sinusoidal", "none"]
    pooling: Literal["mean", "max", "meanmax", "stats", "attention"]
    classifier_hidden: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # the classifier's hidden layer
    dropout: float


@dataclasses.dataclass(frozen=True)
class FramesLIFSettings:
    time_pool: int
    input_dropout: float
    proj_dim: int | None           # a linear layer with the activation before the spiking layers, or None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]
    # spiking layers, as in the CNN + LIF (affect.models.lif)
    lif_hidden: tuple[int, ...]    # width of each spiking layer, 1 to 3 layers
    recurrent: bool
    lif_norm: Literal["none", "batch", "layer"]
    lif_dropout: float
    beta: float
    learn_beta: Literal["none", "layer", "neuron"]
    threshold: float
    reset: Literal["subtract", "zero"]
    detach_reset: bool
    surrogate: Literal["atan", "fast_sigmoid", "sigmoid", "boxcar"]
    surrogate_slope: float
    readout_beta: float
    rate_weight: float
    rate_target: float
    dropout: float                 # dropout before the output layer


@dataclasses.dataclass(frozen=True)
class FramesRNNSettings:
    time_pool: int
    input_dropout: float
    proj_dim: int | None
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]
    rnn: Literal["gru", "lstm"]
    rnn_hidden: int
    rnn_layers: int                # 1 to 4
    bidirectional: bool
    rnn_dropout: float             # between stacked layers; 0 with one layer
    pooling: Literal["mean", "max", "meanmax", "stats", "attention", "last"]
    classifier_hidden: int | None
    dropout: float


def check(settings, n_inputs: int) -> None:
    """The checks of any of the four heads (dispatched on the settings' class); no torch needed."""
    if n_inputs < 1:
        raise ValueError("model: the frames need a width of at least 1")
    if settings.time_pool < 1:
        raise ValueError("model.time_pool must be at least 1")
    if not 0.0 <= settings.input_dropout < 1.0:
        raise ValueError("model.input_dropout must be in [0, 1)")
    if getattr(settings, "proj_dim", None) is not None and settings.proj_dim < 1:
        raise ValueError("model.proj_dim must be positive or None")
    if isinstance(settings, FramesPoolSettings):
        check_head(settings)
    elif isinstance(settings, FramesTransformerSettings):
        check_head(settings)
        if settings.d_model < 1 or settings.n_heads < 1 or settings.d_model % settings.n_heads:
            raise ValueError(f"model: n_heads {settings.n_heads} must divide d_model {settings.d_model}")
        if settings.n_layers < 1 or settings.ff_dim < 1:
            raise ValueError("model: n_layers and ff_dim must be at least 1")
        if not 0.0 <= settings.transformer_dropout < 1.0:
            raise ValueError("model.transformer_dropout must be in [0, 1)")
        if settings.positional == "sinusoidal" and settings.d_model % 2:
            raise ValueError("model: the sinusoidal position code needs an even d_model")
    elif isinstance(settings, FramesLIFSettings):
        check_neuron(settings)
        if not 1 <= len(settings.lif_hidden) <= 3:
            raise ValueError("model.lif_hidden must list 1 to 3 spiking layers")
    elif isinstance(settings, FramesRNNSettings):
        check_head(settings)
        if settings.rnn_hidden < 1:
            raise ValueError("model.rnn_hidden must be at least 1")
        if not 1 <= settings.rnn_layers <= 4:
            raise ValueError("model.rnn_layers must be 1 to 4")
        if not 0.0 <= settings.rnn_dropout < 1.0:
            raise ValueError("model.rnn_dropout must be in [0, 1)")
        if settings.rnn_layers == 1 and settings.rnn_dropout > 0:
            raise ValueError("model.rnn_dropout acts between stacked layers, so it must be 0 with one layer")
    else:
        raise ValueError(f"no frames head for settings {type(settings).__name__}")


def _classes():
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    from affect.models import lif
    from affect.models.cnn import activation_layer

    _, _, SpikingLinear = lif.classes()

    class FramesInput(nn.Module):
        """[batch, width, frames] and the real frames -> ([batch, steps, out], mask [batch, steps]): time pooling over
        real frames, dropout, and the optional projection with its activation."""

        def __init__(self, settings, n_inputs: int, proj_dim):
            super().__init__()
            self.time_pool = settings.time_pool
            self.dropout = nn.Dropout(settings.input_dropout)
            self.project = nn.Linear(n_inputs, proj_dim) if proj_dim is not None else None
            self.act = activation_layer(settings.activation) if proj_dim is not None else None
            self.width = proj_dim if proj_dim is not None else n_inputs
            self.n_inputs = n_inputs

        def forward(self, x, valid_frames):
            h = x.float().transpose(1, 2)                                   # [batch, frames, width]
            steps = h.shape[1]
            mask = lif.step_mask(valid_frames.clamp(1, steps), steps, h.device)
            h, mask = lif.pool_time(h, mask, self.time_pool)
            h = self.dropout(h)
            if self.project is not None:
                h = self.act(self.project(h))
            return h * mask[..., None].to(h.dtype), mask

    class FramesPool(nn.Module):
        def __init__(self, settings: FramesPoolSettings, n_inputs: int, n_classes: int):
            super().__init__()
            check(settings, n_inputs)
            self.input = FramesInput(settings, n_inputs, settings.proj_dim)
            self.pooling = settings.pooling
            self.attention = nn.Linear(self.input.width, 1) if settings.pooling == "attention" else None
            self.embedding_dim = pooled_width(self.input.width, settings.pooling)
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, self.embedding_dim, n_classes)

        def embed(self, x, valid_frames):
            h, mask = self.input(x, valid_frames)
            return masked_pool(h.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            return self.classify(self.embed(x, valid_frames))

    class FramesTransformer(nn.Module):
        def __init__(self, settings: FramesTransformerSettings, n_inputs: int, n_classes: int):
            super().__init__()
            check(settings, n_inputs)
            self.input = FramesInput(settings, n_inputs, None)
            self.project = nn.Linear(n_inputs, settings.d_model)
            layer = nn.TransformerEncoderLayer(
                settings.d_model, settings.n_heads, dim_feedforward=settings.ff_dim,
                dropout=settings.transformer_dropout, activation=settings.transformer_activation,
                batch_first=True, norm_first=settings.norm_first)
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=settings.n_layers,
                norm=nn.LayerNorm(settings.d_model) if settings.norm_first else None, enable_nested_tensor=False)
            self.positional = settings.positional
            self.pooling = settings.pooling
            self.attention = nn.Linear(settings.d_model, 1) if settings.pooling == "attention" else None
            self.embedding_dim = pooled_width(settings.d_model, settings.pooling)
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, self.embedding_dim, n_classes)

        def embed(self, x, valid_frames):
            h, mask = self.input(x, valid_frames)
            h = self.project(h)
            if self.positional == "sinusoidal":
                h = h + sinusoidal(h.shape[1], h.shape[-1], h.device, h.dtype)
            h = self.encoder(h, src_key_padding_mask=~mask)
            return masked_pool(h.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            return self.classify(self.embed(x, valid_frames))

    class FramesLIF(nn.Module):
        def __init__(self, settings: FramesLIFSettings, n_inputs: int, n_classes: int):
            super().__init__()
            check(settings, n_inputs)
            self.recorder = lif.Recorder()
            self.input = FramesInput(settings, n_inputs, settings.proj_dim)
            width, layers = self.input.width, []
            for k, hidden in enumerate(settings.lif_hidden):
                layers.append(SpikingLinear(width, hidden, settings, f"lif{k + 1}", self.recorder, real_input=k == 0))
                width = hidden
            self.spiking = nn.ModuleList(layers)
            self.readout_beta = settings.readout_beta
            self.rate_weight = settings.rate_weight
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(width, n_classes))

        def embed(self, x, valid_frames):
            h, mask = self.input(x, valid_frames)
            if self.recorder.on:
                self.recorder.utterances += x.shape[0]
                if self.input.project is not None:
                    self.recorder.add("proj.mac", mask.float().sum().item() * self.input.n_inputs * self.input.width)
            for layer in self.spiking:
                h = layer(h, mask)
            return lif.readout(h, mask, self.readout_beta)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            return self.classify(self.embed(x, valid_frames))

        def activity_penalty(self):
            return lif.penalty_of(self, self.rate_weight)

        def record(self, on: bool) -> None:
            if on:
                self.recorder.reset()
            self.recorder.on = on

        def spike_report(self) -> dict:
            return lif.report(self.recorder)

    class FramesRNN(nn.Module):
        def __init__(self, settings: FramesRNNSettings, n_inputs: int, n_classes: int):
            super().__init__()
            check(settings, n_inputs)
            self.input = FramesInput(settings, n_inputs, settings.proj_dim)
            recurrent = nn.GRU if settings.rnn == "gru" else nn.LSTM
            self.rnn = recurrent(self.input.width, settings.rnn_hidden, num_layers=settings.rnn_layers,
                                 batch_first=True, dropout=settings.rnn_dropout, bidirectional=settings.bidirectional)
            self.rnn_kind, self.rnn_layers, self.rnn_hidden = settings.rnn, settings.rnn_layers, settings.rnn_hidden
            self.directions = 2 if settings.bidirectional else 1
            width = settings.rnn_hidden * self.directions
            self.pooling = settings.pooling
            self.attention = nn.Linear(width, 1) if settings.pooling == "attention" else None
            self.embedding_dim = width if settings.pooling == "last" else pooled_width(width, settings.pooling)
            self.head_dropout = settings.dropout
            self.head = classifier_head(settings, self.embedding_dim, n_classes)

        def embed(self, x, valid_frames):
            h, mask = self.input(x, valid_frames)
            batch, steps = h.shape[0], h.shape[1]
            keep = mask.sum(dim=1)
            with torch.autocast(device_type=h.device.type, enabled=False):
                packed = pack_padded_sequence(h.float(), keep.detach().cpu().to(torch.int64), batch_first=True,
                                              enforce_sorted=False)
                out, state = self.rnn(packed)
                if self.pooling == "last":
                    final = state[0] if self.rnn_kind == "lstm" else state
                    final = final.view(self.rnn_layers, self.directions, batch, self.rnn_hidden)[-1]
                    return final.transpose(0, 1).reshape(batch, self.directions * self.rnn_hidden)
                out, _ = pad_packed_sequence(out, batch_first=True, total_length=steps)
            return masked_pool(out.float().transpose(1, 2), mask, self.pooling, self.attention)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            return self.classify(self.embed(x, valid_frames))

    return FramesPool, FramesTransformer, FramesLIF, FramesRNN


_BUILT = None


def classes() -> tuple:
    global _BUILT
    if _BUILT is None:
        _BUILT = _classes()
    return _BUILT


@register_model("frames_pool", settings=FramesPoolSettings)
def build_frames_pool(settings: FramesPoolSettings, n_inputs: int, n_classes: int):
    return classes()[0](settings, n_inputs, n_classes)


@register_model("frames_transformer", settings=FramesTransformerSettings)
def build_frames_transformer(settings: FramesTransformerSettings, n_inputs: int, n_classes: int):
    return classes()[1](settings, n_inputs, n_classes)


@register_model("frames_lif", settings=FramesLIFSettings)
def build_frames_lif(settings: FramesLIFSettings, n_inputs: int, n_classes: int):
    return classes()[2](settings, n_inputs, n_classes)


@register_model("frames_rnn", settings=FramesRNNSettings)
def build_frames_rnn(settings: FramesRNNSettings, n_inputs: int, n_classes: int):
    return classes()[3](settings, n_inputs, n_classes)


def steps_after(valid_frames, time_pool: int):
    """The steps a head sees for each utterance: its frames in groups of time_pool, rounded up (no torch needed for
    ints)."""
    return math.ceil(valid_frames / time_pool)

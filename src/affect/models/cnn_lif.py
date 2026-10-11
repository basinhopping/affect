"""CNN + LIF, variant A, the hybrid (plan 4.4; model family 3a, added 2026-09-29).

The convolutional front end is the CNN's (affect.models.cnn.conv_front_end). Its output becomes one
vector per remaining time step: the frequency axis is averaged ("mean") or every remaining band's
channels are concatenated ("flatten"), as in the CNN + Transformer. Consecutive steps can then be
averaged in groups of time_pool (real steps only). Those real-valued vectors are the input current
of the first spiking layer; lif_hidden gives the width of each spiking layer (1 to 3), each linear
synapses (recurrent when asked), an optional normalisation of the current, LIF neurons
(affect.models.lif) and dropout with one mask per sequence. The readout is non-spiking: the last
layer's spikes feed a leaky integrator with leak readout_beta, averaged over the utterance's own
steps (lif_reference.readout_embedding), then dropout and a linear layer give the logits.
readout_beta 0 makes that a linear layer on each neuron's spike count per step.

The training loss adds rate_weight times the activity penalty (affect.models.lif) when rate_weight
is above 0. efficiency.json gets each spiking layer's firing rate on the validation set and the
operations per utterance: the front end's multiply-accumulates, the first spiking layer's
multiply-accumulates (its input is real), and every later layer's accumulates (spikes in).
"""
import dataclasses
from typing import Literal

from affect.models.cnn import check_front_end, conv_front_end, frames_after
from affect.models.cnn_transformer import bands_after
from affect.models.lif import check_neuron
from affect.registry import register_model


@dataclasses.dataclass(frozen=True)
class CNNLIFSettings:
    # convolutional front end, as in the CNN (affect.models.cnn.CNNSettings)
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + normalisation + activation layers per block
    kernel_size: int              # square kernel, odd
    pool_freq: int                # max-pool factor along frequency after each block
    pool_time: int                # max-pool factor along time after each block
    norm: Literal["batch", "subspectral", "group", "none"]   # front end, as in the CNN
    norm_groups: int | None       # groups for norm "group"; None otherwise
    norm_subbands: int | None     # frequency bands for norm "subspectral"; None otherwise
    activation: Literal["relu", "leaky_relu", "elu", "gelu", "silu"]   # front end only
    # from the front end to the spiking layers
    freq_reduce: Literal["mean", "flatten"]   # as in the CNN + Transformer
    time_pool: int                # average consecutive steps in groups of this size before the spiking layers (1: none)
    # spiking layers (affect.models.lif)
    lif_hidden: tuple[int, ...]   # width of each spiking layer, 1 to 3 layers
    recurrent: bool               # each spiking layer also gets its own spikes of the previous step
    lif_norm: Literal["none", "batch", "layer"]   # normalisation of each layer's input current (batch: over real steps)
    lif_dropout: float            # dropout on each spiking layer's spikes, one mask per sequence
    beta: float                   # leak per step, in [0, 1)
    learn_beta: Literal["none", "layer", "neuron"]   # fixed leak, one learned per layer, or one per neuron (starting at beta)
    threshold: float              # firing threshold
    reset: Literal["subtract", "zero"]   # subtract the threshold after a spike, or set the potential to zero
    detach_reset: bool            # no gradient through the reset
    surrogate: Literal["atan", "fast_sigmoid", "sigmoid", "boxcar"]   # surrogate gradient of the spike
    surrogate_slope: float        # its sharpness (lif_reference gives each formula)
    # readout and training
    readout_beta: float           # leak of the readout's integrator, in [0, 1); 0 reads spike counts
    rate_weight: float            # weight of the activity penalty in the loss (0: none)
    rate_target: float            # the firing rate the penalty pulls each neuron towards, in [0, 1]
    dropout: float                # dropout before the output layer


def check(settings: CNNLIFSettings, n_mels: int) -> None:
    check_front_end(settings, n_mels)
    check_neuron(settings)
    if settings.time_pool < 1:
        raise ValueError("model.time_pool must be at least 1")
    if not 1 <= len(settings.lif_hidden) <= 3:
        raise ValueError("model.lif_hidden must list 1 to 3 spiking layers")


def _module_class():
    import torch
    from torch import nn

    from affect.models import lif

    Neuron, _, SpikingLinear = lif.classes()

    class CNNLIF(nn.Module):
        def __init__(self, settings: CNNLIFSettings, n_mels: int, n_classes: int):
            super().__init__()
            check(settings, n_mels)
            self.recorder = lif.Recorder()
            self.front = conv_front_end(settings, n_mels)
            self.n_blocks, self.pool_time_factor = len(settings.channels), settings.pool_time
            self.freq_reduce, self.time_pool = settings.freq_reduce, settings.time_pool
            self.n_mels, self.channels = n_mels, settings.channels
            self.kernel_size, self.convs_per_block, self.pool_freq = (settings.kernel_size, settings.convs_per_block,
                                                                      settings.pool_freq)
            width = settings.channels[-1] * (bands_after(settings, n_mels) if settings.freq_reduce == "flatten" else 1)
            layers = []
            for k, hidden in enumerate(settings.lif_hidden):
                layers.append(SpikingLinear(width, hidden, settings, f"lif{k + 1}", self.recorder, real_input=k == 0))
                width = hidden
            self.spiking = nn.ModuleList(layers)
            self.readout_beta = settings.readout_beta
            self.rate_weight = settings.rate_weight
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(width, n_classes))

        def frames_after_blocks(self, valid_frames):
            return frames_after(valid_frames, self.n_blocks, self.pool_time_factor)

        def _front_macs(self, valid_frames):
            """Multiply-accumulates of the front end's convolutions over each utterance's own frames."""
            total, c_in, frames = 0.0, 1, valid_frames.float()
            bands = self.n_mels
            for c_out in self.channels:
                for _ in range(self.convs_per_block):
                    total += float((frames * bands).sum().item()) * c_out * c_in * self.kernel_size ** 2
                    c_in = c_out
                frames = torch.div(frames, self.pool_time_factor, rounding_mode="floor").clamp_min(1)
                bands //= self.pool_freq
            return total

        def embed(self, x, valid_frames):
            """x: [batch, n_mels, frames]; valid_frames: [batch]. The readout [batch, embedding_dim]."""
            if self.recorder.on:
                self.recorder.utterances += x.shape[0]
                self.recorder.add("front.mac", self._front_macs(valid_frames))
            h = self.front(x.unsqueeze(1))                        # [batch, channels, bands, steps]
            if self.freq_reduce == "mean":
                h = h.float().mean(dim=2).transpose(1, 2)        # [batch, steps, channels]
            else:
                b, c, f, t = h.shape
                h = h.float().permute(0, 3, 1, 2).reshape(b, t, c * f)
            steps = h.shape[1]
            mask = lif.step_mask(self.frames_after_blocks(valid_frames).clamp_max(steps), steps, h.device)
            h, mask = lif.pool_time(h, mask, self.time_pool)
            for layer in self.spiking:
                h = layer(h, mask)
            return lif.readout(h, mask, self.readout_beta)

        def classify(self, embedding):
            return self.head(embedding)

        def forward(self, x, valid_frames):
            return self.classify(self.embed(x, valid_frames))

        def activity_penalty(self):
            """Added to the loss by train.loop.Step; None when rate_weight is 0."""
            return lif.penalty_of(self, self.rate_weight)

        def record(self, on: bool) -> None:
            if on:
                self.recorder.reset()
            self.recorder.on = on

        def spike_report(self) -> dict:
            return lif.report(self.recorder)

    return CNNLIF


@register_model("cnn_lif", settings=CNNLIFSettings)
def build_cnn_lif(settings: CNNLIFSettings, n_mels: int, n_classes: int):
    return _module_class()(settings, n_mels, n_classes)

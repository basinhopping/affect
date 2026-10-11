"""CNN + LIF, variant B, fully spiking (plan 4.4; model family 3b, added 2026-09-29).

Convolution blocks like the CNN's, with LIF neurons (affect.models.lif) after every convolution in
place of the activation, and max pooling of the spikes (a spike survives if any in the window
fired). Two ways for the audio to enter (encoding):

  "frames"  each log-mel frame is one time step. The first convolution reads the normalised
            log-mel values (real numbers as input current, as in Bittar and Garner 2022); its
            neurons, one per channel and frequency row, integrate along the frames. Later layers
            read spikes. Pooling along time shortens every later layer's sequence.
  "repeat"  the whole log-mel image is shown static_steps times (the vision approach): each neuron
            is one channel at one frequency row and frame, and integrates over the repeats.

After the last block, "frames" gives one vector of spikes per remaining frame (channels times
remaining frequency rows); "repeat" gives, per repeat, each channel and row's mean over the
utterance's own frames. Then up to two spiking linear layers (lif_hidden, recurrent when asked)
and the readout of the CNN + LIF hybrid: a leaky integrator averaged over the steps
(lif_reference.readout_embedding), dropout and a linear layer.

conv_norm "batch" puts batch norm on each convolution's current (batch, frequency and frames of the
batch, as in the CNN); "none" gives the convolutions a bias instead. As in the CNN, a pooling
window at the end of an utterance can see one padded frame. The activity penalty and the
efficiency report work as in the hybrid; here the first convolution is the only layer with real
input (multiply-accumulates; with "repeat" it is computed once, not per repeat), and every other
layer receives spikes (accumulates).
"""
import dataclasses
from typing import Literal

from affect.models.cnn import frames_after
from affect.models.lif import check_neuron
from affect.registry import register_model


@dataclasses.dataclass(frozen=True)
class SNNSettings:
    encoding: Literal["frames", "repeat"]   # see the module docstring
    static_steps: int | None      # repeats of the image for "repeat"; None for "frames"
    # spiking convolution blocks
    channels: tuple[int, ...]     # output channels of each block; the number of entries is the number of blocks
    convs_per_block: int          # convolution + LIF layers per block
    kernel_size: int              # square kernel over frequency and frames, odd
    pool_freq: int                # max-pool factor along frequency after each block
    pool_time: int                # max-pool factor along frames after each block
    conv_norm: Literal["batch", "none"]   # batch norm on each convolution's current, or none (then a bias)
    # spiking linear layers after the convolutions
    lif_hidden: tuple[int, ...]   # width of each, 0 to 2 layers
    recurrent: bool               # the linear layers also get their own spikes of the previous step
    lif_norm: Literal["none", "batch", "layer"]   # normalisation of the linear layers' input current
    lif_dropout: float            # dropout on the spikes of every spiking layer, one mask per sequence
    # neurons, everywhere (affect.models.lif)
    beta: float                   # leak per step, in [0, 1)
    learn_beta: Literal["none", "layer", "neuron"]   # fixed leak, one learned per layer, or one per neuron (starting at beta)
    threshold: float              # firing threshold
    reset: Literal["subtract", "zero"]
    detach_reset: bool            # no gradient through the reset
    surrogate: Literal["atan", "fast_sigmoid", "sigmoid", "boxcar"]
    surrogate_slope: float
    # readout and training
    readout_beta: float           # leak of the readout's integrator, in [0, 1); 0 reads spike counts
    rate_weight: float            # weight of the activity penalty in the loss (0: none)
    rate_target: float            # the firing rate the penalty pulls each channel or unit towards
    dropout: float                # dropout before the output layer


def check(settings: SNNSettings, n_mels: int) -> None:
    if settings.encoding == "repeat":
        if settings.static_steps is None or settings.static_steps < 1:
            raise ValueError("model: encoding 'repeat' needs static_steps of at least 1")
    elif settings.static_steps is not None:
        raise ValueError("model: encoding 'frames' takes static_steps None")
    if not settings.channels or any(c < 1 for c in settings.channels):
        raise ValueError("model.channels must be a non-empty list of positive numbers")
    if settings.convs_per_block < 1:
        raise ValueError("model.convs_per_block must be at least 1")
    if settings.kernel_size < 1 or settings.kernel_size % 2 == 0:
        raise ValueError("model.kernel_size must be odd")
    if settings.pool_freq < 1 or settings.pool_time < 1:
        raise ValueError("model.pool_freq and pool_time must be at least 1")
    if n_mels // settings.pool_freq ** len(settings.channels) < 1:
        raise ValueError(f"model: {len(settings.channels)} blocks pooling frequency by {settings.pool_freq} "
                         f"leave no bands of {n_mels}")
    if len(settings.lif_hidden) > 2:
        raise ValueError("model.lif_hidden lists at most 2 spiking linear layers")
    check_neuron(settings)


def _module_class():
    import torch
    from torch import nn

    from affect.models import lif

    Neuron, _, SpikingLinear = lif.classes()

    class SpikingConv(nn.Module):
        """Convolution, optional batch norm of its current, and LIF neurons."""

        def __init__(self, c_in: int, c_out: int, bands: int, settings, name: str, recorder, real_input: bool):
            super().__init__()
            k = settings.kernel_size
            self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=settings.conv_norm == "none")
            self.norm = nn.BatchNorm2d(c_out) if settings.conv_norm == "batch" else None
            beta_shape = (c_out, bands) if settings.encoding == "frames" else (c_out, bands, 1)
            self.neuron = Neuron(beta_shape, settings, name, recorder)
            self.fan_out = c_out * k * k
            self.dropout = settings.lif_dropout
            self.real_input, self.name, self.recorder = real_input, name, recorder

        def current(self, h, frame_mask):
            """h [n, c_in, bands, frames] -> the masked current [n, c_out, bands, frames]."""
            cur = self.conv(h)
            if self.norm is not None:
                cur = self.norm(cur)
            return cur * frame_mask[:, None, None, :].to(cur.dtype)

        def count_input(self, h, frame_mask, repeats: int = 1):
            if not self.recorder.on:
                return
            m = frame_mask[:, None, None, :].float()
            if self.real_input:
                self.recorder.add(f"{self.name}.mac", (m.expand_as(h).sum().item()) * self.fan_out * repeats)
            else:
                self.recorder.add(f"{self.name}.ac", (h.float() * m).sum().item() * self.fan_out)

    class SNN(nn.Module):
        def __init__(self, settings: SNNSettings, n_mels: int, n_classes: int):
            super().__init__()
            check(settings, n_mels)
            self.recorder = lif.Recorder()
            self.encoding, self.static_steps = settings.encoding, settings.static_steps
            self.n_blocks, self.pool_time = len(settings.channels), settings.pool_time
            self.pool = nn.MaxPool2d((settings.pool_freq, settings.pool_time))
            convs, c_in, bands = [], 1, n_mels
            for b, c_out in enumerate(settings.channels):
                block = []
                for k in range(settings.convs_per_block):
                    block.append(SpikingConv(c_in, c_out, bands, settings, f"conv{b + 1}.{k + 1}", self.recorder,
                                             real_input=not convs and not block))
                    c_in = c_out
                convs.append(nn.ModuleList(block))
                bands //= settings.pool_freq
            self.blocks = nn.ModuleList(convs)
            width = settings.channels[-1] * bands
            hidden = []
            for k, size in enumerate(settings.lif_hidden):
                real = k == 0 and settings.encoding == "repeat"      # per-repeat rates, not spikes
                hidden.append(SpikingLinear(width, size, settings, f"lif{k + 1}", self.recorder, real_input=real))
                width = size
            self.spiking = nn.ModuleList(hidden)
            self.dropout_p = settings.lif_dropout
            self.readout_beta, self.rate_weight = settings.readout_beta, settings.rate_weight
            self.embedding_dim = width
            self.head_dropout = settings.dropout
            self.head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(width, n_classes))

        def frames_after_blocks(self, valid_frames):
            return frames_after(valid_frames, self.n_blocks, self.pool_time)

        def _frames(self, x, valid_frames):
            """encoding "frames": one step per frame. Returns ([batch, steps, width], step mask)."""
            h = x.unsqueeze(1)                                            # [batch, 1, mels, frames]
            valid = valid_frames
            for b, block in enumerate(self.blocks):
                frames = h.shape[-1]
                mask = lif.step_mask(valid.clamp_max(frames), frames, h.device)
                for layer in block:
                    layer.count_input(h, mask)
                    cur = layer.current(h, mask).permute(0, 3, 1, 2)      # [batch, frames, channels, bands]
                    s = layer.neuron(cur, mask[:, :, None, None])
                    s = lif.time_dropout(s, self.dropout_p, self.training)
                    h = s.permute(0, 2, 3, 1)                             # [batch, channels, bands, frames]
                h = self.pool(h)
                valid = torch.div(valid, self.pool_time, rounding_mode="floor").clamp_min(1)
            b_, c, f, t = h.shape
            steps = h.permute(0, 3, 1, 2).reshape(b_, t, c * f)
            return steps, lif.step_mask(valid.clamp_max(t), t, h.device)

        def _repeat(self, x, valid_frames):
            """encoding "repeat": the image static_steps times. Returns ([batch, repeats, width], mask)."""
            n, reps = x.shape[0], self.static_steps
            h = x.unsqueeze(1)                                            # [batch, 1, mels, frames]
            valid = valid_frames
            first = True
            for block in self.blocks:
                frames = h.shape[-1]
                mask = lif.step_mask(valid.clamp_max(frames), frames, h.device)
                for layer in block:
                    if first:                                             # the same current every repeat
                        layer.count_input(h, mask)
                        cur = layer.current(h, mask)[:, None].expand(-1, reps, -1, -1, -1)
                        first = False
                    else:                                                 # h: [batch, repeats, c, bands, frames]
                        flat = h.reshape((n * reps,) + tuple(h.shape[2:]))
                        layer.count_input(flat, mask.repeat_interleave(reps, dim=0))
                        cur = layer.current(flat, mask.repeat_interleave(reps, dim=0))
                        cur = cur.reshape((n, reps) + tuple(cur.shape[1:]))
                    s = layer.neuron(cur, mask[:, None, None, None, :])
                    h = lif.time_dropout(s, self.dropout_p, self.training)
                flat = self.pool(h.reshape((n * reps,) + tuple(h.shape[2:])))
                h = flat.reshape((n, reps) + tuple(flat.shape[1:]))
                valid = torch.div(valid, self.pool_time, rounding_mode="floor").clamp_min(1)
            frames = h.shape[-1]
            mask = lif.step_mask(valid.clamp_max(frames), frames, h.device).float()
            rates = (h * mask[:, None, None, None, :]).sum(dim=-1) / mask.sum(dim=1)[:, None, None, None]
            steps = rates.reshape(n, reps, -1)                            # [batch, repeats, channels * bands]
            return steps, torch.ones(n, reps, dtype=torch.bool, device=x.device)

        def embed(self, x, valid_frames):
            """x: [batch, n_mels, frames]; valid_frames: [batch]. The readout [batch, embedding_dim]."""
            if self.recorder.on:
                self.recorder.utterances += x.shape[0]
            if self.encoding == "frames":
                h, mask = self._frames(x.float(), valid_frames)
            else:
                h, mask = self._repeat(x.float(), valid_frames)
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

    return SNN


@register_model("snn", settings=SNNSettings)
def build_snn(settings: SNNSettings, n_mels: int, n_classes: int):
    return _module_class()(settings, n_mels, n_classes)

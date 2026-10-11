"""Spiking layers shared by the CNN + LIF models (plan 4.4), added 2026-09-29.

The neuron is the one written out in affect.models.lif_reference, which a node test checks this
module against, forward and backward, on fixed inputs:

    subtract reset:  U[t] = beta * U[t-1] + I[t] - theta * S[t-1]
    zero reset:      U[t] = beta * (1 - S[t-1]) * U[t-1] + I[t]
    S[t] = 1 if U[t] >= theta else 0, with a surrogate gradient going back

Every neuron argument is a setting of the model (no library defaults): beta, learn_beta ("none";
"layer", one learned leak per layer; "neuron", one per neuron; a learned leak starts at beta and is
clamped to [0, 1] when used, as snnTorch does), threshold, reset, detach_reset, surrogate and
surrogate_slope. The state is computed in float32 even under bf16 autocast. Time runs as a plain
Python loop over steps: fine for the tens to a few hundred steps these models see; a fused kernel
(SpikingJelly) would be the fix if a node timing shows the loop is the bottleneck.

Also here: the time pooling, dropout with one mask per sequence, batch norm over the real steps
only, the readout (see lif_reference), the activity penalty (plan 4.4, "activity regularization":
rate_weight times the mean over neurons of (firing rate - rate_target) squared, each layer's rates
over the batch's real steps), and the spike recorder behind efficiency.json.

Operations per utterance (efficiency.json, estimates, plan 4.4): a synaptic layer fed with real
numbers does multiply-accumulates (MAC): its input elements times its fan-out; one fed with spikes
does accumulates (AC): its input spikes times its fan-out (Linear: out features; Conv2d: out
channels times kernel area, ignoring the border). Energy at 4.6 pJ per MAC and 0.9 pJ per AC
(Horowitz 2014, 45 nm, as used by Spikformer), which ignores memory access. Counted over the
utterance's own frames only.
"""
import math

from affect.models.lif_reference import RESETS, SURROGATES

LEARN_BETA = ("none", "layer", "neuron")
LIF_NORMS = ("none", "batch", "layer")


def check_neuron(settings) -> None:
    """The neuron fields every spiking model has; no torch needed."""
    if not 0.0 <= settings.beta < 1.0:
        raise ValueError("model.beta must be in [0, 1)")
    if settings.learn_beta not in LEARN_BETA:
        raise ValueError(f"model.learn_beta must be one of {LEARN_BETA}")
    if settings.threshold <= 0.0:
        raise ValueError("model.threshold must be positive")
    if settings.reset not in RESETS or settings.surrogate not in SURROGATES:
        raise ValueError(f"model: reset must be one of {RESETS} and surrogate one of {SURROGATES}")
    if settings.surrogate_slope <= 0.0:
        raise ValueError("model.surrogate_slope must be positive")
    if not 0.0 <= settings.lif_dropout < 1.0:
        raise ValueError("model.lif_dropout must be in [0, 1)")
    if settings.lif_norm not in LIF_NORMS:
        raise ValueError(f"model.lif_norm must be one of {LIF_NORMS}")
    if any(h < 1 for h in settings.lif_hidden):
        raise ValueError("model.lif_hidden sizes must be positive")
    if not 0.0 <= settings.readout_beta < 1.0:
        raise ValueError("model.readout_beta must be in [0, 1)")
    if settings.rate_weight < 0.0 or not 0.0 <= settings.rate_target <= 1.0:
        raise ValueError("model: rate_weight must be at least 0 and rate_target in [0, 1]")
    if not 0.0 <= settings.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")


def _spike_class():
    import torch

    class Spike(torch.autograd.Function):
        """Heaviside forward (1 where x >= 0); the surrogate derivative going back
        (lif_reference.surrogate_derivative, the same four formulas)."""

        @staticmethod
        def forward(ctx, x, kind, slope):
            ctx.save_for_backward(x)
            ctx.kind, ctx.slope = kind, slope
            return (x >= 0).to(x.dtype)

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            k, a = ctx.kind, ctx.slope
            if k == "atan":
                d = (a / 2.0) / (1.0 + (math.pi / 2.0 * a * x) ** 2)
            elif k == "fast_sigmoid":
                d = 1.0 / (1.0 + a * x.abs()) ** 2
            elif k == "sigmoid":
                s = torch.sigmoid(a * x)
                d = a * s * (1.0 - s)
            elif k == "boxcar":
                d = (x.abs() < 0.5 / a).to(x.dtype) * (0.5 * a)
            else:
                raise ValueError(k)
            return grad * d, None, None

    return Spike


_SPIKE = None


def spike(x, kind: str, slope: float):
    global _SPIKE
    if _SPIKE is None:
        _SPIKE = _spike_class()
    return _SPIKE.apply(x, kind, slope)


def step_mask(valid_steps, steps: int, device):
    """[batch, steps] bool, True on each sequence's own steps."""
    import torch

    return torch.arange(steps, device=device)[None, :] < valid_steps[:, None]


def pool_time(h, mask, factor: int):
    """Mean over consecutive groups of `factor` steps of h [batch, steps, width], counting real
    steps only. Returns (pooled h, new mask); a group with any real step is real."""
    import torch

    if factor == 1:
        return h, mask
    b, t, d = h.shape
    groups = -(-t // factor)
    pad = groups * factor - t
    m = mask.to(h.dtype)
    if pad:
        h = torch.cat([h, h.new_zeros(b, pad, d)], dim=1)
        m = torch.cat([m, m.new_zeros(b, pad)], dim=1)
    h = h.reshape(b, groups, factor, d)
    m = m.reshape(b, groups, factor)
    count = m.sum(dim=2)
    pooled = (h * m[..., None]).sum(dim=2) / count.clamp_min(1.0)[..., None]
    return pooled, count > 0


def readout(spikes, mask, readout_beta: float):
    """spikes: [batch, steps, width]; mask: [batch, steps]. (1 - readout_beta) times the mean over
    real steps of Y[t] = readout_beta * Y[t-1] + S[t] (lif_reference.readout_embedding)."""
    import torch

    s = spikes.float() * mask[..., None].float()
    if readout_beta > 0.0:
        y = torch.zeros_like(s[:, 0])
        trace = []
        for t in range(s.shape[1]):
            y = readout_beta * y + s[:, t]
            trace.append(y)
        s = torch.stack(trace, dim=1) * mask[..., None].float()
    count = mask.float().sum(dim=1).clamp_min(1.0)
    return (1.0 - readout_beta) * s.sum(dim=1) / count[:, None]


def time_dropout(x, p: float, training: bool):
    """Dropout with one mask per sequence, the same at every step: x [batch, steps, ...]."""
    import torch

    if not training or p == 0.0:
        return x
    keep = torch.empty((x.shape[0], 1) + tuple(x.shape[2:]), device=x.device).bernoulli_(1.0 - p) / (1.0 - p)
    return x * keep.to(x.dtype)


class Recorder:
    """Sums, per named quantity, over every batch while on; efficiency.json reads them."""

    def __init__(self):
        self.on = False
        self.sums = {}
        self.utterances = 0

    def add(self, key: str, value) -> None:
        if self.on:
            self.sums[key] = self.sums.get(key, 0.0) + float(value)

    def reset(self) -> None:
        self.sums, self.utterances = {}, 0


def _module_classes():
    import torch
    from torch import nn

    class Neuron(nn.Module):
        """A layer of LIF neurons. forward(current [batch, steps, *per-step shape], real, recurrent=None)
        returns the spikes, float32, with the current's shape. real is 1 on real positions and 0 on
        padding, broadcasting to the spikes; it decides nothing in the dynamics (padding comes after
        each utterance's own steps), only what the activity penalty and the recorder count. beta_shape
        is the shape of one leak per neuron (learn_beta "neuron"); it broadcasts over each step."""

        def __init__(self, beta_shape, settings, name: str, recorder: Recorder):
            super().__init__()
            self.threshold, self.reset = settings.threshold, settings.reset
            self.detach_reset = settings.detach_reset
            self.surrogate, self.slope = settings.surrogate, settings.surrogate_slope
            self.rate_weight, self.rate_target = settings.rate_weight, settings.rate_target
            self.name, self.recorder = name, recorder
            if settings.learn_beta == "none":
                self.register_buffer("beta_value", torch.tensor(settings.beta, dtype=torch.float32))
            elif settings.learn_beta == "layer":
                self.beta_value = nn.Parameter(torch.tensor(settings.beta, dtype=torch.float32))
            else:
                self.beta_value = nn.Parameter(torch.full(tuple(beta_shape), settings.beta, dtype=torch.float32))
            self.penalty = None

        def beta(self):
            return self.beta_value.clamp(0.0, 1.0)

        def forward(self, current, real, recurrent=None):
            beta = self.beta()
            cur = current if current.dtype == torch.float64 else current.float()   # float64 only in tests
            u = cur.new_zeros((cur.shape[0],) + tuple(cur.shape[2:]))
            s = torch.zeros_like(u)
            out = []
            for t in range(cur.shape[1]):
                i_t = cur[:, t]
                if recurrent is not None:
                    i_t = i_t + recurrent(s).to(i_t.dtype)
                r = s.detach() if self.detach_reset else s
                if self.reset == "subtract":
                    u = beta * u + i_t - self.threshold * r
                else:
                    u = beta * (1.0 - r) * u + i_t
                s = spike(u - self.threshold, self.surrogate, self.slope)
                out.append(s)
            spikes = torch.stack(out, dim=1)
            real = real.float().expand_as(spikes)
            self.penalty = None
            if self.training and self.rate_weight > 0.0:
                # one rate per unit of the first per-step axis (a linear unit, or a convolution channel)
                axes = tuple(a for a in range(spikes.dim()) if a != 2)
                rates = (spikes * real).sum(dim=axes) / real.sum(dim=axes).clamp_min(1.0)
                self.penalty = ((rates - self.rate_target) ** 2).mean()
            if self.recorder.on:
                self.recorder.add(f"{self.name}.spikes", (spikes * real).sum().item())
                self.recorder.add(f"{self.name}.neuron_steps", real.sum().item())
            return spikes

    class StepBatchNorm(nn.Module):
        """Batch norm of [batch, steps, width] over the real steps only (padded steps pass as 0)."""

        def __init__(self, width: int):
            super().__init__()
            self.norm = nn.BatchNorm1d(width)

        def forward(self, x, mask):
            out = torch.zeros_like(x)
            out[mask] = self.norm(x[mask]).to(x.dtype)
            return out

    class SpikingLinear(nn.Module):
        """Linear synapses (plus a recurrent weight when asked), an optional normalisation of the
        input current, LIF neurons, and dropout with one mask per sequence."""

        def __init__(self, n_in: int, n_out: int, settings, name: str, recorder: Recorder, real_input: bool):
            super().__init__()
            self.fc = nn.Linear(n_in, n_out, bias=settings.lif_norm == "none")
            self.norm = (StepBatchNorm(n_out) if settings.lif_norm == "batch"
                         else nn.LayerNorm(n_out) if settings.lif_norm == "layer" else None)
            self.recurrent = nn.Linear(n_out, n_out, bias=False) if settings.recurrent else None
            self.neuron = Neuron((n_out,), settings, name, recorder)
            self.dropout = settings.lif_dropout
            self.n_in, self.n_out, self.real_input = n_in, n_out, real_input
            self.name, self.recorder = name, recorder

        def forward(self, x, mask):
            if self.recorder.on:
                steps = mask.float().sum().item()
                if self.real_input:
                    self.recorder.add(f"{self.name}.mac", steps * self.n_in * self.n_out)
                else:
                    self.recorder.add(f"{self.name}.ac", (x.float() * mask[..., None].float()).sum().item() * self.n_out)
            current = self.fc(x)
            if isinstance(self.norm, StepBatchNorm):
                current = self.norm(current, mask)
            elif self.norm is not None:
                current = self.norm(current)
            current = current * mask[..., None].to(current.dtype)
            spikes = self.neuron(current, mask[..., None], self.recurrent)
            if self.recorder.on and self.recurrent is not None:
                self.recorder.add(f"{self.name}.ac", (spikes * mask[..., None].float()).sum().item() * self.n_out)
            return time_dropout(spikes, self.dropout, self.training)

    return Neuron, StepBatchNorm, SpikingLinear


_CLASSES = None


def classes():
    global _CLASSES
    if _CLASSES is None:
        _CLASSES = _module_classes()
    return _CLASSES


def penalty_of(model, weight: float):
    """rate_weight times the mean over the model's neuron layers of their last penalty, or None."""
    if weight == 0.0:
        return None
    Neuron = classes()[0]
    terms = [m.penalty for m in model.modules() if isinstance(m, Neuron) and m.penalty is not None]
    if not terms:
        return None
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return weight * total / len(terms)


def report(recorder: Recorder) -> dict:
    """efficiency.json's spike section from a recorder's sums: each neuron layer's firing rate over
    real steps, and operations and energy per sequence (estimates, see the module docstring). A
    sequence is one utterance, or one segment when the job evaluates on segments."""
    pj_per_mac, pj_per_ac = 4.6, 0.9      # Horowitz 2014, 45 nm, as used by Spikformer: fixed facts, not settings
    sums, n = recorder.sums, max(1, recorder.utterances)
    layers = sorted({k.rsplit(".", 1)[0] for k in sums if k.endswith(".spikes")})
    rates = {name: (sums[f"{name}.spikes"] / sums[f"{name}.neuron_steps"]
                    if sums.get(f"{name}.neuron_steps") else None) for name in layers}
    mac = sum(v for k, v in sums.items() if k.endswith(".mac")) / n
    ac = sum(v for k, v in sums.items() if k.endswith(".ac")) / n
    silent = [name for name, r in rates.items() if r == 0.0]
    return {"sequences": recorder.utterances, "firing_rate": rates, "silent_layers": silent,
            "mac_per_sequence": mac, "ac_per_sequence": ac,
            "energy_uj_per_sequence_estimate": (mac * pj_per_mac + ac * pj_per_ac) / 1e6,
            "energy_note": "4.6 pJ per MAC and 0.9 pJ per AC (Horowitz 2014, 45 nm); memory access ignored"}

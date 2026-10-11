"""A plain reference of the leaky integrate-and-fire neuron, in numpy (plan 4.4 and 6.6).

The torch layers in affect.models.lif must match this on fixed inputs, forward and backward (a
node test does both); this module needs no torch, so its own tests also run in the Cowork
container. Added 2026-09-29.

The neuron, with input current I[t] for t = 0 .. T-1, leak beta in [0, 1) and threshold theta > 0,
is the formulation of Eshraghian et al. 2023 (their equation for U[t]) and snnTorch's Leaky neuron:
the reset at step t uses the spike of step t - 1.

    U[-1] = 0, S[-1] = 0
    reset "subtract":  U[t] = beta * U[t-1] + I[t] - theta * S[t-1]
    reset "zero":      U[t] = beta * (1 - S[t-1]) * U[t-1] + I[t]
    spike:             S[t] = 1 if U[t] >= theta, else 0

A recurrent layer adds R S[t-1] to I[t]. A spike is a step function, so training uses a surrogate
gradient: going back, dS[t]/dU[t] is taken as sg(U[t] - theta), with one of

    "atan"          (slope / 2) / (1 + (pi / 2 * slope * x) ** 2)      snnTorch's default, slope 2
    "fast_sigmoid"  1 / (1 + slope * |x|) ** 2                          Zenke and Ganguli 2018
    "sigmoid"       slope * sigmoid(slope * x) * (1 - sigmoid(slope * x))
    "boxcar"        0.5 * slope where |x| < 0.5 / slope, else 0          Bittar and Garner 2022 use slope 1

With detach_reset, the reset term passes no gradient (snnTorch always detaches it).

The readout: the last spiking layer's spikes feed a leaky integrator without threshold,
Y[t] = readout_beta * Y[t-1] + S[t], averaged over the utterance's own steps and scaled by
(1 - readout_beta), so that a steady firing rate r gives about r whatever the leak; a linear layer
turns that into logits. The linear layer commutes with the average, so this equals the average
potential of a non-spiking output layer, with its bias added once. readout_beta 0 gives the
spike count of each neuron divided by the number of steps.
"""
import math

import numpy as np

SURROGATES = ("atan", "fast_sigmoid", "sigmoid", "boxcar")
RESETS = ("subtract", "zero")


def surrogate_derivative(x, kind: str, slope: float):
    x = np.asarray(x, dtype=np.float64)
    if kind == "atan":
        return (slope / 2.0) / (1.0 + (math.pi / 2.0 * slope * x) ** 2)
    if kind == "fast_sigmoid":
        return 1.0 / (1.0 + slope * np.abs(x)) ** 2
    if kind == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-slope * x))
        return slope * s * (1.0 - s)
    if kind == "boxcar":
        return np.where(np.abs(x) < 0.5 / slope, 0.5 * slope, 0.0)
    raise ValueError(f"unknown surrogate {kind!r}")


def lif_forward(current, beta, threshold: float, reset: str, recurrent=None, fire=None):
    """current: [T, ...]. beta: a number or an array broadcasting to one step. recurrent: None, or
    an [N, N] matrix R for a layer of N neurons (current [T, N]), adding R @ S[t-1] to I[t].
    fire: None for the step function; the tests pass a smooth function instead, to check
    lif_backward against finite differences. Returns (spikes, potentials), both [T, ...], the
    potentials as computed at each step (the reset shows in the next step's U)."""
    current = np.asarray(current, dtype=np.float64)
    if reset not in RESETS:
        raise ValueError(f"unknown reset {reset!r}")
    u = np.zeros(current.shape[1:])
    s = np.zeros(current.shape[1:])
    spikes, potentials = np.zeros_like(current), np.zeros_like(current)
    for t in range(current.shape[0]):
        i_t = current[t] + (recurrent @ s if recurrent is not None else 0.0)
        if reset == "subtract":
            u = beta * u + i_t - threshold * s
        else:
            u = beta * (1.0 - s) * u + i_t
        s = (u >= threshold).astype(np.float64) if fire is None else fire(u - threshold)
        spikes[t], potentials[t] = s, u
    return spikes, potentials


def lif_backward(current, beta: float, threshold: float, reset: str, surrogate: str, slope: float,
                 detach_reset: bool, grad_spikes, fire=None):
    """Backpropagation through time for one feedforward layer with a scalar leak, written out step
    by step without any autograd, to check the torch layer's gradients. grad_spikes is dL/dS[t]
    from outside the layer, [T, ...]. Returns (dL/dI, dL/dbeta).

    Going back from the last step, with dU and dS the total derivatives of L:
      subtract:  dS[t] = g[t] - theta * dU[t+1]           (the reset term; 0 with detach_reset)
                 dU[t] = dS[t] * sg[t] + beta * dU[t+1]
                 dL/dbeta += dU[t] * U[t-1]
      zero:      dS[t] = g[t] - beta * U[t] * dU[t+1]     (0 with detach_reset)
                 dU[t] = dS[t] * sg[t] + beta * (1 - S[t]) * dU[t+1]
                 dL/dbeta += dU[t] * (1 - S[t-1]) * U[t-1]
      dI[t] = dU[t]
    """
    current = np.asarray(current, dtype=np.float64)
    grad_spikes = np.asarray(grad_spikes, dtype=np.float64)
    spikes, u = lif_forward(current, beta, threshold, reset, fire=fire)
    sg = surrogate_derivative(u - threshold, surrogate, slope)
    zero = np.zeros(current.shape[1:])
    d_u_next = zero
    grad_current = np.zeros_like(current)
    grad_beta = 0.0
    for t in range(current.shape[0] - 1, -1, -1):
        if reset == "subtract":
            through_reset = zero if detach_reset else -threshold * d_u_next
            carried = beta * d_u_next
        else:
            through_reset = zero if detach_reset else -beta * u[t] * d_u_next
            carried = beta * (1.0 - spikes[t]) * d_u_next
        d_u = (grad_spikes[t] + through_reset) * sg[t] + carried
        grad_current[t] = d_u
        u_prev = u[t - 1] if t > 0 else zero
        s_prev = spikes[t - 1] if t > 0 else zero
        grad_beta += float(np.sum(d_u * (u_prev if reset == "subtract" else (1.0 - s_prev) * u_prev)))
        d_u_next = d_u
    return grad_current, grad_beta


def readout_embedding(spikes, valid_steps: int, readout_beta: float):
    """spikes: [T, N]; the embedding [N]: (1 - readout_beta) times the mean over the first
    valid_steps steps of Y[t] = readout_beta * Y[t-1] + S[t]."""
    y = np.zeros(spikes.shape[1:])
    total = np.zeros(spikes.shape[1:])
    for t in range(valid_steps):
        y = readout_beta * y + spikes[t]
        total += y
    return (1.0 - readout_beta) * total / valid_steps

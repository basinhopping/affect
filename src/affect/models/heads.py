"""Extra output heads on a model's pooled summary (model.embed), added 2026-09-26.

The gender head (train.aux_gender): dropout (the model's own classifier dropout) and one linear
layer from the model's embedding to two logits, female and male. "multitask" trains it like the
emotion output, so the shared layers also learn gender. "adversarial" puts a gradient reversal
between the embedding and the head (Ganin and Lempitsky 2015): the head still learns to predict
gender, while the gradient that reaches the shared layers is negated, pushing them to drop gender.
The weight of the gender loss (train.aux_gender_weight) scales both.

The valence and activation head (train.aux_va, added 2026-10-24) is the multitask form of the same class with two
outputs, valence and activation (affect.train.loop.make_va_head).

The projection head (train.contrastive, added 2026-10-15): a linear layer as wide as the embedding,
ReLU, and a linear layer to train.contrastive_dim outputs, the projection network of Khosla et al.
(2020, section 3.1, whose hidden layer is also as wide as the encoder's output). The supervised
contrastive loss (affect.train.loop.supervised_contrastive_loss) L2-normalises its outputs. It is
used in training only: predictions come from the model alone.
"""
AUX_KINDS = ("multitask", "adversarial")


def _module_class():
    import torch
    from torch import nn

    class _Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return grad.neg()

    class AuxHead(nn.Module):
        def __init__(self, width: int, n_out: int, dropout: float, reverse: bool):
            super().__init__()
            self.reverse = reverse
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(width, n_out)

        def forward(self, embedding):
            h = _Reverse.apply(embedding) if self.reverse else embedding
            return self.linear(self.dropout(h))

    return AuxHead


def build_aux_head(kind: str, width: int, n_out: int, dropout: float):
    if kind not in AUX_KINDS:
        raise ValueError(f"aux head kind {kind!r} is not one of {AUX_KINDS}")
    return _module_class()(width, n_out, dropout, reverse=kind == "adversarial")


def _projection_class():
    from torch import nn

    class ProjectionHead(nn.Module):
        def __init__(self, width: int, dim: int):
            super().__init__()
            self.hidden = nn.Linear(width, width)
            self.relu = nn.ReLU()
            self.out = nn.Linear(width, dim)

        def forward(self, embedding):
            return self.out(self.relu(self.hidden(embedding)))

    return ProjectionHead


def build_projection_head(width: int, dim: int):
    if width < 1 or dim < 1:
        raise ValueError(f"a projection head needs width and dim of at least 1, got {width} and {dim}")
    return _projection_class()(width, dim)

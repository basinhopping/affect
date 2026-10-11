"""Extra output heads on a model's pooled summary (model.embed), added 2026-09-26.

The gender head (train.aux_gender): dropout (the model's own classifier dropout) and one linear
layer from the model's embedding to two logits, female and male. "multitask" trains it like the
emotion output, so the shared layers also learn gender. "adversarial" puts a gradient reversal
between the embedding and the head (Ganin and Lempitsky 2015): the head still learns to predict
gender, while the gradient that reaches the shared layers is negated, pushing them to drop gender.
The weight of the gender loss (train.aux_gender_weight) scales both.
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

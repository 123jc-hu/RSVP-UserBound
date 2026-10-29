import torch
from torch import nn


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * float(ctx.lambd), None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReverseFn.apply(x, float(lambd))


class DomainDiscriminator(nn.Module):
    """Small MLP domain classifier for source-only adversarial alignment."""

    def __init__(self, num_domains: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_domains = int(num_domains)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.num_domains),
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        x = grad_reverse(x, lambd=float(grl_lambda))
        return self.net(x)

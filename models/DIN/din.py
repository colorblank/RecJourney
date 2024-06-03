import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FeatArgs:
    profile_dim: int
    seq_len: int
    candidate_dim: int
    context_dim: int


@dataclass
class AUArgs:
    dim_hidden: int
    dim_out: int = 1


@dataclass
class HeadArgs:
    dim_hidden: int
    dim_out: int


class ActivationUnit(nn.Module):
    def __init__(
        self, dim_in: int, dim_hidden: int, dim_out: int = 1, activation: str = "prelu"
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.PReLU() if activation == "prelu" else nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x_hist: torch.Tensor, x_cand: torch.Tensor) -> torch.Tensor:
        """_summary_

        Arguments:
            x_hist -- a list item features of history items.
                each item feature is a tensor.
                shape: [batch_size, seq_len, dim]
            x_cand -- candicate item feature
                shape: [batch_size, dim]

        Returns:
            torch.Tensor, size: [batch_size, seq_len, 1]
        """
        seq_len = x_hist.shape[1]
        x_cand = x_cand.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x_hist, x_cand, x_hist - x_cand, x_cand * x_hist], dim=-1)
        return self.fc(x)


class DIN(nn.Module):
    def __init__(
        self,
        feat_args: FeatArgs,
        au_args: AUArgs,
        head_args: HeadArgs,
        activation: str = "prelu",
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.au = ActivationUnit(
            dim_in=feat_args.candidate_dim * 4,
            dim_hidden=au_args.dim_hidden,
            dim_out=au_args.dim_out,
            activation=activation,
        )
        dim_in = feat_args.profile_dim + feat_args.candidate_dim * 2
        self.pred = nn.Sequential(
            nn.Linear(dim_in, head_args.dim_hidden, bias=bias),
            nn.PReLU() if activation == "prelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_args.dim_hidden, head_args.dim_out, bias=bias),
        )

    def forward(
        self,
        x_profile: torch.Tensor,
        x_hist: torch.Tensor,
        x_cand: torch.Tensor,
        x_context: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Arguments:
            x_profile -- _description_
            x_hist -- size [batch_size, seq_len, dim]
            x_cand -- _description_
            x_context -- _description_

        Returns:
            _description_
        """        
        w = self.au(x_hist, x_cand) # [batch_size, seq_len, 1]

        x = (x_hist * w).sum(dim=1)
        x = torch.cat([x_profile, x, x_context], dim=-1)
        y = self.pred(x)
        return y


if __name__ == "__main__":
    feat_args = FeatArgs(
        profile_dim=10,
        seq_len=5,
        candidate_dim=10,
        context_dim=10,
    )
    au_args = AUArgs(
        dim_hidden=10,
        dim_out=1,
    )
    head_args = HeadArgs(
        dim_hidden=10,
        dim_out=1,
    )
    din = DIN(feat_args, au_args, head_args)
    batch_size = 2
    x_profile = torch.randn(batch_size, feat_args.profile_dim)
    x_context = torch.randn(batch_size, feat_args.context_dim)
    x_hist = torch.randn(batch_size, feat_args.seq_len, feat_args.candidate_dim)
    x_candicate = torch.randn(batch_size, feat_args.candidate_dim)

    y = din(x_profile, x_hist, x_candicate, x_context)

    print(y.shape)
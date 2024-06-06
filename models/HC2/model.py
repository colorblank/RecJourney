import torch
import torch.nn as nn
import torch.nn.functional as F


class IndividualContrastiveLoss(nn.Module):
    def __init__(self, tau: float, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon

    def forward(
        self,
        f_org: torch.Tensor,
        f_aug: torch.Tensor,
        f_neg_other: torch.Tensor,
        f_neg_cross: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Arguments:
            f_org -- original feature, size = (batch_size, dim_feat)
            f_aug -- augmented feature, size = (batch_size, dim_feat)
            f_neg_other -- negtive sample feature from other scene,
                            size = (batch_size, num_neg, dim_feat)
            f_neg_cross -- negtive sample feature from cross scene
                            size = (batch_size, num_neg, dim_feat)
            e_pos -- positive sample embedding, size = (batch_size, dim_embedding)
            e_neg -- negtive sample embedding, size = (batch_size, num_neg, dim_embedding)
        Returns:
            _description_
        """

        # normalize
        f_org = F.normalize(f_org, dim=-1)
        f_aug = F.normalize(f_aug, dim=-1)
        f_neg_cross = F.normalize(f_neg_cross, dim=-1)
        f_neg_other = F.normalize(f_neg_other, dim=-1)

        h_neg_other = torch.exp(
            torch.einsum(
                "bnd,bnd->bn", f_org.unsqueeze(1).expand_as(f_neg_other), f_neg_other
            )
            / self.tau
        )
        h_neg_cross = torch.exp(
            torch.einsum(
                "bnd,bnd->bn", f_aug.unsqueeze(1).expand_as(f_neg_cross), f_neg_cross
            )
            / self.tau
        )
        h_pos = torch.exp(torch.einsum("bd,bd->b", f_org, f_aug) / self.tau)

        loss = -h_pos / (h_pos + (h_neg_other + h_neg_cross).sum(1) + self.epsilon)
        return loss.mean()


class GeneralContrastiveLoss(nn.Module):
    def __init__(self, tau: float, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon

    def forward(
        self,
        f_org: torch.Tensor,
        f_aug: torch.Tensor,
        f_neg: torch.Tensor,
        e_org: torch.Tensor,
        e_pos: torch.Tensor,
        e_neg: torch.Tensor,
    ) -> torch.Tensor:
        sim_neg = torch.einsum(
            "bnd,bnd->bn", e_pos.unsqueeze(1).expand_as(e_neg), e_neg
        )
        sim_pos = torch.einsum("bd,bd->b", e_org, e_pos)

        # normalize
        f_org = F.normalize(f_org, dim=-1)
        f_aug = F.normalize(f_aug, dim=-1)
        f_neg = F.normalize(f_neg, dim=-1)

        h_neg = torch.exp(
            torch.einsum("bnd,bnd->bn", f_org.unsqueeze(1).expand_as(f_neg), f_neg)
            / self.tau
        )
        h_pos = sim_pos * torch.exp(torch.einsum("bd,bd->b", f_org, f_aug) / self.tau)

        loss = -h_pos / (h_pos + (sim_neg + h_neg).sum(1) + self.epsilon)
        return loss.mean()


if __name__ == "__main__":
    f_org = torch.randn(32, 64)
    f_aug = torch.randn(32, 64)
    f_neg_other = torch.randn(32, 10, 64)
    f_neg_cross = torch.randn(32, 10, 64)
    loss_s = IndividualContrastiveLoss(tau=0.1)
    y = loss_s(f_org, f_aug, f_neg_other, f_neg_cross)
    print(y)
    e_org = torch.randn(32, 64)
    e_pos = torch.randn(32, 64)
    e_neg = torch.randn(32, 10, 64)
    loss_g = GeneralContrastiveLoss(tau=0.1)
    y = loss_g(f_org, f_aug, f_neg_other, e_org, e_pos, e_neg)
    print(y)

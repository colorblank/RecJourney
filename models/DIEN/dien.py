from typing import List

import torch
import torch.nn as nn


class LinearAct(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, bias: bool = True, act: str = "relu"
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias)
        if act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif act.lower() == "prelu":
            self.act = nn.PReLU()
        else:
            raise NotImplementedError(f"{act} is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class GRU(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = True) -> None:
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(dim_in + dim_hidden, dim_hidden, bias=bias), nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(dim_in + dim_hidden, dim_hidden, bias=bias), nn.Sigmoid()
        )
        self.candidate_gate = nn.Sequential(
            nn.Linear(dim_in + dim_hidden, dim_hidden, bias=bias), nn.Tanh()
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        if h is None:
            h = torch.zeros_like(x).to(x.device)
        u = self.update_gate(torch.cat([x, h], dim=-1))
        r = self.reset_gate(torch.cat([x, h], dim=-1))
        h_hat = self.candidate_gate(torch.cat([x, r * h], dim=-1))
        h = u * h + (1 - u) * h_hat
        return h


class AUGRUCell(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = True):
        super(AUGRUCell, self).__init__()

        in_dim = dim_in + dim_hidden
        self.reset_gate = nn.Sequential(
            nn.Linear(in_dim, dim_hidden, bias=bias), nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(in_dim, dim_hidden, bias=bias), nn.Sigmoid()
        )
        self.h_hat_gate = nn.Sequential(
            nn.Linear(in_dim, dim_hidden, bias=bias), nn.Tanh()
        )

    def forward(
        self, X: torch.Tensor, h_prev: torch.Tensor, attention_score: torch.Tensor
    ) -> torch.Tensor:
        """_summary_

        Arguments:
            X -- current feature, size = (batch_size, dim_in)
            h_prev -- previous hidden state, size = (batch_size, dim_hidden)
            attention_score -- _description_

        Returns:
            _description_
        """
        temp_input = torch.cat([h_prev, X], dim=-1)
        r = self.reset_gate(temp_input)
        u = self.update_gate(temp_input)

        h_hat = self.h_hat_gate(torch.cat([h_prev * r, X], dim=-1))

        u = attention_score.unsqueeze(1) * u
        h_cur = (1.0 - u) * h_prev + u * h_hat

        return h_cur


class DynamicGRU(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, bias=True):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.rnn_cell = AUGRUCell(dim_in, dim_hidden, bias=bias)

    def forward(
        self, X: torch.Tensor, attenion_scores: torch.Tensor, h0: torch.Tensor = None
    ) -> torch.Tensor:
        """_summary_

        Arguments:
            X -- history visited item feature, size = (batch_size, seq_len, dim_in)
            attenion_scores -- _description_

        Keyword Arguments:
            h0 -- _description_ (default: {None})

        Returns:
            _description_
        """
        B, T, _ = X.shape
        H = self.hidden_dim

        output = torch.zeros(B, T, H).type(X.type()).to(X.device)
        h_prev = torch.zeros(B, H).type(X.type()).to(X.device) if h0 is None else h0
        for t in range(T):
            h_prev = output[:, t, :] = self.rnn_cell(
                X[:, t, :], h_prev, attenion_scores[:, t]
            )
        return output


class ActivationUnit(nn.Module):
    """from DIN

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dims: List[int],
        dim_out: int = 1,
        act: str = "prelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        dims = [dim_in] + hidden_dims + [dim_out]
        self.fc = nn.Sequential(
            *[
                LinearAct(dim_in, dim_out, bias, act)
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
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


class AttentionLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dims: List[int],
        return_score: bool = False,
        bias: bool = True,
        act: str = "sigmoid",
        mask_value: float = -float("inf"),
    ) -> None:
        super().__init__()
        self.return_score = return_score
        self.mask_value = mask_value
        self.local_attn = ActivationUnit(emb_dim * 4, hidden_dims, bias=bias, act=act)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """_summary_

        Args:
            query (torch.Tensor): size = (batch_size, dim)
            keys (torch.Tensor): size = (batch_size, seq_len, dim)
            mask (torch.Tensor, optional): size = (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: size = (batch_size, seq_len)
        """
        attention_score = self.local_attn(keys, query)  # [batch_size, seq_len, 1]

        if mask is not None:
            mask = mask.unsqueeze(-1) == 1  # [batch_size, seq_len, 1]
            attention_score = attention_score.masked_fill(~mask, self.mask_value)

        attn_weights = torch.softmax(attention_score, dim=1)
        attn_weights = attn_weights.squeeze(-1)  # [batch_size, seq_len]
        if self.return_score:
            return attn_weights
        else:
            return torch.einsum("bt,btd->bd", attn_weights, keys)


class DIEN(nn.Module):
    def __init__(
        self, emb_dim: int, hidden_dims: List[int] = [80, 40], act: str = "relu"
    ):
        super().__init__()

        self.gru_based_layer = nn.GRU(emb_dim * 2, emb_dim * 2, batch_first=True)
        self.attention_layer = AttentionLayer(emb_dim, hidden_dims=hidden_dims, act=act)
        self.gru_customized_layer = DynamicGRU(emb_dim * 2, emb_dim * 2)

    def forward(
        self,
        user_embedding: torch.Tensor,
        item_historical_embedding: torch.Tensor,
        item_embedding: torch.Tensor,
        mask: torch.Tensor,
        sequential_length: torch.Tensor,
    ):
        """_summary_

        Arguments:
            user_embedding -- user profile embedding
                size: (batch_size, emb_dim)
            item_historical_embedding -- user visited items embedding
                size = (batch_size, seq_len, emb_dim)
            item_embedding -- target item embedding
                size = (batch_size, emb_dim)
            mask -- mask of item_historical_embedding
                size = (batch_size, seq_len)
            sequential_length -- length of each user's historical items
                size = (batch_size, )

        Keyword Arguments:
            neg_sample -- _description_ (default: {False})

        Returns:
            _description_
        """
        item_historical_embedding_sum = torch.matmul(
            mask.unsqueeze(dim=1), item_historical_embedding
        ).squeeze() / sequential_length.unsqueeze(dim=1)

        output_based_gru, _ = self.gru_based_layer(item_historical_embedding)
        attention_scores = self.attention_layer(
            item_embedding, output_based_gru, mask, return_scores=True
        )
        output_customized_gru = self.gru_customized_layer(
            output_based_gru, attention_scores
        )

        attention_feature = output_customized_gru[
            range(len(sequential_length)), sequential_length - 1
        ]

        combination = torch.cat(
            [
                user_embedding,
                item_embedding,
                item_historical_embedding_sum,
                item_embedding * item_historical_embedding_sum,
                attention_feature,
            ],
            dim=1,
        )

        scores = self.output_layer(combination)

        return scores.squeeze()


if __name__ == "__main__":
    emb_dim = 10
    hidden_dimss = [20, 30]
    attn_model = AttentionLayer(emb_dim, hidden_dimss, return_score=True)
    batch_size = 2
    seq_length = 5
    emb_dim = 10
    query = torch.randn(batch_size, emb_dim)
    fact = torch.randn(batch_size, seq_length, emb_dim)
    mask = torch.randn(batch_size, seq_length) > 0

    # 运行 forward 方法
    weighted_facts = attn_model.forward(query, fact, mask)
    print(weighted_facts.shape)

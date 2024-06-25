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


class AttentionLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dimss: List[int],
        bias: bool = True,
        act: str = "sigmoid",
    ):
        super().__init__()

        layers_dimensions = [emb_dim * 4] + hidden_dimss + [1]
        self.layers = nn.Sequential(
            *[
                LinearAct(dim_in, dim_out, bias, act)
                for dim_in, dim_out in zip(
                    layers_dimensions[:-1], layers_dimensions[1:]
                )
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        fact: torch.Tensor,
        mask: torch.Tensor,
        return_scores: bool = False,
    ) -> torch.Tensor:
        """
        Perform forward pass to compute attention-weighted facts or directly return attention scores.

        Parameters:
        - query (torch.Tensor): Query tensor, shape (B, D) where B is batch size, D is embedding dimension.
        - fact (torch.Tensor): Fact tensor, shape (B, T, D) where T is sequence length.
        - mask (torch.Tensor): Mask tensor, shape (B, T) indicating valid inputs.
        - return_scores (bool, optional): If True, returns attention scores. Defaults to False.

        Returns:
        - torch.Tensor: Attention-weighted facts if return_scores is False, otherwise attention scores.
        """
        B, T, D = fact.size()

        query_broadcasted = query.unsqueeze(1).expand(B, T, D)
        combined = torch.cat(
            [
                fact,  # Original facts (B, T, D)
                query_broadcasted,  # Broadcasted query (B, T, D)
                fact * query_broadcasted,  # Element-wise product (B, T, D)
                query_broadcasted - fact,  # Element-wise difference (B, T, D)
            ],
            dim=2,
        )  # Resulting in (B, T, 4D)

        raw_scores = torch.squeeze(self.layers(combined), dim=-1)
        masked_scores = torch.where(mask == 1, raw_scores, float("-inf"))
        attn_weights = torch.softmax(masked_scores, dim=1).masked_fill(mask == 0, 0)
        if return_scores:
            return attn_weights.squeeze(1)
        else:
            return torch.einsum("bt,btd->bd", attn_weights, fact)


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
        sequential_length,
    ):
        """_summary_

        Arguments:
            user_embedding -- _description_
            item_historical_embedding -- _description_
            item_embedding -- _description_
            mask -- _description_
            sequential_length -- _description_

        Keyword Arguments:
            neg_sample -- _description_ (default: {False})

        Returns:
            _description_
        """
        # history item embedding; [ batch_size, max_length, emb_dim * 2 ]
        # target item embedding; [ batch_size, emb_dim * 2 ]
        # mask; [ batch_size, max_length ]
        # user embedding; [ batch_size, emb_dim ]
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
    attn_model = AttentionLayer(emb_dim, hidden_dimss)
    batch_size = 2
    seq_length = 5
    emb_dim = 10
    query = torch.randn(batch_size, emb_dim)
    fact = torch.randn(batch_size, seq_length, emb_dim)
    mask = torch.ones(batch_size, seq_length)

    # 运行 forward 方法
    weighted_facts = attn_model.forward(query, fact, mask, return_scores=True)
    print(weighted_facts.shape)

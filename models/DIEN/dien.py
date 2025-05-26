import torch
import torch.nn as nn
from torch import Tensor


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

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, h: Tensor = None):
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

    def forward(self, X: Tensor, h_prev: Tensor, attention_score: Tensor) -> Tensor:
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

    def forward(self, X: Tensor, attenion_scores: Tensor, h0: Tensor = None) -> Tensor:
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
        H = self.dim_hidden

        output = torch.zeros(B, T, H).type(X.type()).to(X.device)
        h_prev = torch.zeros(B, H).type(X.type()).to(X.device) if h0 is None else h0
        for t in range(T):
            h_prev = output[:, t, :] = self.rnn_cell(
                X[:, t, :], h_prev, attenion_scores[:, t]
            )
        return output


class ActivationUnit(nn.Module):
    """from DIN

    Parameters:
    - dim_in (int): _description_
    - hidden_dims (List[int]): _description_
    - dim_out (int, optional): _description_. Defaults to 1.
    - act (str, optional): _description_. Defaults to "prelu".
    - bias (bool, optional): _description_. Defaults to True.

    Input:
    - x_hist (Tensor): size = (batch_size, seq_len, dim)
    - x_cand (Tensor): size = (batch_size, dim)

    Returns:
        Tensor: size = (batch_size, seq_len, 1)
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dims: list[int],
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

    def forward(self, x_hist: Tensor, x_cand: Tensor) -> Tensor:
        """_summary_

        Arguments:
            x_hist -- a list item features of history items.
                each item feature is a tensor.
                shape: [batch_size, seq_len, dim]
            x_cand -- candicate item feature
                shape: [batch_size, dim]

        Returns:
            Tensor, size: [batch_size, seq_len, 1]
        """
        seq_len = x_hist.shape[1]
        x_cand = x_cand.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x_hist, x_cand, x_hist - x_cand, x_cand * x_hist], dim=-1)
        return self.fc(x)


class AttentionLayer(nn.Module):
    """_summary_

    Parameters:
    - emb_dim (int): _description_
    - hidden_dims (List[int]): _description_
    - return_score (bool, optional): _description_. Defaults to False.
    - bias (bool, optional): _description_. Defaults to True.
    - act (str, optional): _description_. Defaults to "sigmoid".
    - mask_value (float, optional): _description_. Defaults to -float("inf").

    Input:
    - query (Tensor): size = (batch_size, dim)
    - keys (Tensor): size = (batch_size, seq_len, dim)
    - mask (Tensor, optional): size = (batch_size, seq_len). Defaults to None.

    Returns:
        Tensor: size = (batch_size, seq_len)
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dims: list[int],
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
        query: Tensor,
        keys: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """_summary_

        Args:
            query (Tensor): size = (batch_size, dim)
            keys (Tensor): size = (batch_size, seq_len, dim)
            mask (Tensor, optional): size = (batch_size, seq_len). Defaults to None.

        Returns:
            Tensor: size = (batch_size, seq_len)
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


class PredictHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        hidden_dims: list[int],
        out_dim: int = 1,
        act: str = "sigmoid",
        bias: bool = True,
    ) -> None:
        super().__init__()
        predict_head_dims = [dim_in] + hidden_dims + [out_dim]
        fcs = []
        for i, dims in enumerate(zip(predict_head_dims[:-1], predict_head_dims[1:])):
            din, dout = dims
            activation = "sigmoid" if i != len(predict_head_dims) - 1 else act
            fc = LinearAct(din, dout, bias, activation)
            fcs.append(fc)
        self.fc = nn.Sequential(*fcs)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class DIEN(nn.Module):
    def __init__(
        self,
        user_emb_dim: int,
        emb_dim: int,
        hidden_dims: list[int],
        predict_head_dims: list[int],
        num_classes: int = 1,
        act: str = "relu",
        bias: bool = True,
    ):
        super().__init__()

        self.gru_based_layer = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.attention_layer = AttentionLayer(emb_dim, hidden_dims=hidden_dims, act=act)
        self.gru_customized_layer = DynamicGRU(emb_dim, emb_dim)

        self.predict_head = PredictHead(
            dim_in=emb_dim * 4 + user_emb_dim,
            hidden_dims=predict_head_dims,
            out_dim=num_classes,
            bias=bias,
            act=act,
        )

    def forward(
        self,
        user_embedding: Tensor,
        item_historical_embedding: Tensor,
        item_embedding: Tensor,
        mask: Tensor,
        sequential_length: Tensor,
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
        item_historical_embedding_masked = (
            item_historical_embedding * mask.unsqueeze(-1).float()
        )
        item_historical_embedding_sum = item_historical_embedding_masked.sum(1)
        item_historical_embedding_sum = (
            item_historical_embedding_sum / sequential_length.unsqueeze(-1)
        )

        output_based_gru, _ = self.gru_based_layer(item_historical_embedding)
        attention_scores = self.attention_layer(item_embedding, output_based_gru, mask)
        output_customized_gru = self.gru_customized_layer(
            output_based_gru, attention_scores
        )

        attention_feature = output_customized_gru[
            range(len(sequential_length)), sequential_length - 1
        ]

        combination = torch.cat(
            [
                user_embedding,  # [batch_size, user_emb_dim]
                item_embedding,  # [batch_size, item_emb_dim]
                item_historical_embedding_sum,  # [batch_size, item_emb_dim]
                item_embedding
                * item_historical_embedding_sum,  # [batch_size, item_emb_dim]
                attention_feature,  # [batch_size, item_emb_dim]
            ],
            dim=1,
        )

        scores = self.predict_head(combination)

        return scores.squeeze()


if __name__ == "__main__":
    emb_dim = 10
    hidden_dims = [20, 30]
    attn_model = AttentionLayer(emb_dim, hidden_dims, return_score=True)
    batch_size = 2
    seq_length = 5
    emb_dim = 10
    query = torch.randn(batch_size, emb_dim)
    fact = torch.randn(batch_size, seq_length, emb_dim)
    mask = torch.randn(batch_size, seq_length) > 0
    user_emb_dim = emb_dim * 5
    model = DIEN(
        user_emb_dim=user_emb_dim,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        predict_head_dims=hidden_dims,
        act="relu",
    )

    user_embedding: Tensor = torch.randn(batch_size, user_emb_dim)
    item_historical_embedding: Tensor = torch.randn(batch_size, seq_length, emb_dim)
    item_embedding: Tensor = torch.randn(batch_size, emb_dim)
    mask: Tensor = torch.randint(0, 2, (batch_size, seq_length))
    sequential_length: Tensor = torch.randint(1, seq_length, (batch_size,))
    y = model(
        user_embedding,
        item_historical_embedding,
        item_embedding,
        mask,
        sequential_length,
    )
    print(y.shape)

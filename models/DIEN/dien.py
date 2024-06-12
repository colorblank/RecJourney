import torch
import torch.nn as nn


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
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(AUGRUCell, self).__init__()

        in_dim = input_dim + hidden_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias), nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias), nn.Sigmoid()
        )
        self.h_hat_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias), nn.Tanh()
        )

    def forward(self, X, h_prev, attention_score):
        temp_input = torch.cat([h_prev, X], dim=-1)
        r = self.reset_gate(temp_input)
        u = self.update_gate(temp_input)

        h_hat = self.h_hat_gate(torch.cat([h_prev * r, X], dim=-1))

        u = attention_score.unsqueeze(1) * u
        h_cur = (1.0 - u) * h_prev + u * h_hat

        return h_cur


class DynamicGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = AUGRUCell(input_dim, hidden_dim, bias=True)

    def forward(self, X, attenion_scores, h0=None):
        B, T, D = X.shape
        H = self.hidden_dim

        output = torch.zeros(B, T, H).type(X.type())
        h_prev = torch.zeros(B, H).type(X.type()) if h0 is None else h0
        for t in range(T):
            h_prev = output[:, t, :] = self.rnn_cell(
                X[:, t, :], h_prev, attenion_scores[:, t]
            )
        return output


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_size, activation_layer="sigmoid"):
        super().__init__()

        Activation = nn.Sigmoid
        if activation_layer == "Dice":
            pass

        def _dense(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), Activation())

        dimension_pair = [embedding_dim * 8] + hidden_size
        layers = [
            _dense(dimension_pair[i], dimension_pair[i + 1])
            for i in range(len(hidden_size))
        ]
        layers.append(nn.Linear(hidden_size[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, query, fact, mask, return_scores=False):
        B, T, D = fact.shape

        query = torch.ones((B, T, 1)).type(query.type()) * query.view((B, 1, D))
        combination = torch.cat([fact, query, fact * query, query - fact], dim=2)

        scores = self.model(combination).squeeze()
        scores = torch.where(mask == 1, scores, torch.ones_like(scores) * (-(2**31)))

        scores = (scores.softmax(dim=-1) * mask).view((B, 1, T))

        if return_scores:
            return scores.squeeze()
        return torch.matmul(scores, fact).squeeze()


class DIEN(nn.Module):
    def __init__(self, n_uid, n_mid, n_cid, embedding_dim):
        super().__init__()

        self.gru_based_layer = nn.GRU(
            embedding_dim * 2, embedding_dim * 2, batch_first=True
        )
        self.attention_layer = AttentionLayer(
            embedding_dim, hidden_size=[80, 40], activation_layer="sigmoid"
        )
        self.gru_customized_layer = DynamicGRU(embedding_dim * 2, embedding_dim * 2)

        # self.output_layer = MLP( embedding_dim * 9, [ 200, 80], 1, 'ReLU')

    def forward(
        self,
        user_embedding,
        item_historical_embedding,
        item_embedding,
        mask,
        sequential_length,
        neg_sample=False,
    ):
        # history item embedding; [ batch_size, max_length, embedding_dim * 2 ]
        # target item embedding; [ batch_size, embedding_dim * 2 ]
        # mask; [ batch_size, max_length ]
        # user embedding; [ batch_size, embedding_dim ]
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


class DIEN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # gru
        self.gru = nn.GRU(
            input_size=self.cfg.emb_dim,
            hidden_size=self.cfg.emb_dim,
            num_layers=1,
            batch_first=True,
        )
        # attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.cfg.emb_dim, self.cfg.emb_dim),
            nn.Tanh(),
            nn.Linear(self.cfg.emb_dim, 1),
            nn.Softmax(dim=1),
        )
        # weighted gru
        self.wgru = nn.GRU(
            input_size=self.cfg.emb_dim,
            hidden_size=self.cfg.emb_dim,
            num_layers=1,
            batch_first=True,
        )

        # predict layer
        self.predict = nn.Sequential(
            nn.Linear(self.cfg.emb_dim * 2, self.cfg.emb_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.emb_dim, 1),
            nn.Sigmoid(),
        )

import torch
from torch import nn


class FusionMLP(nn.Module):
    def __init__(self, input_dimz, hidden_dim, out_dim, graph_num, first, **model_args):
        super().__init__()
        self.input_dimz = input_dimz
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.graph_num = graph_num
        self.first = first

        self.fusion_num_layer = model_args["fusion_num_layer"]
        self.fusion_dim = model_args["fusion_dim"]
        self.fusion_dropout = model_args["fusion_dropout"]

        self.if_forward = model_args["if_forward"]
        self.if_backward = model_args["if_backward"]
        self.node_dim = model_args["node_dim"]
        self.nhead = model_args["nhead"]
        self.if_feedback = model_args["if_feedback"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        if self.graph_num > 1:
            self.fusion_graph_model = nn.Sequential(
                *[MultiLayerPerceptron(input_dimz=self.input_dimz,
                                       hidden_dim=self.hidden_dim,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
            )
            if self.if_forward:
                self.fusion_forward_linear = nn.Linear(in_features=self.hidden_dim, out_features=self.fusion_dim,
                                                       bias=True)

            if self.if_backward:
                self.fusion_backward_linear = nn.Linear(in_features=self.hidden_dim, out_features=self.fusion_dim,
                                                        bias=True)

            self.fusion_model = nn.Sequential(
                *[MultiLayerPerceptron(input_dimz=self.graph_num * self.fusion_dim,
                                       hidden_dim=self.graph_num * self.fusion_dim,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
                nn.Linear(in_features=self.graph_num * self.fusion_dim, out_features=self.out_dim, bias=True),
            )
        else:
            self.fusion_model = nn.Sequential(
                *[MultiLayerPerceptron(input_dimz=self.input_dimz,
                                       hidden_dim=self.hidden_dim,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
                nn.Linear(in_features=self.hidden_dim, out_features=self.node_dim, bias=True)
            )
            self.fusion_linear = nn.Linear(in_features=self.node_dim, out_features=self.out_dim, bias=True)

        if not self.first:
            if self.if_feedback:
                if self.if_forward:
                    self.forward_att = nn.MultiheadAttention(embed_dim=self.fusion_dim,
                                                             num_heads=self.nhead,
                                                             batch_first=True)
                    self.forward_fc = nn.Sequential(
                        nn.Linear(in_features=self.fusion_dim, out_features=self.node_dim, bias=True),
                        nn.Sigmoid()
                    )
                if self.if_backward:
                    self.backward_att = nn.MultiheadAttention(embed_dim=self.fusion_dim,
                                                              num_heads=self.nhead,
                                                              batch_first=True)
                    self.backward_fc = nn.Sequential(
                        nn.Linear(in_features=self.fusion_dim, out_features=self.node_dim, bias=True),
                        nn.Sigmoid()
                    )

    def forward(self, history_data, time_series_emb, predict_emb,
                node_forward_emb, node_backward_emb, hidden_forward_emb, hidden_backward_emb):
        tem_emb = []
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1] * self.time_of_day_size
            tem_emb.append(self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)])
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2] * self.day_of_week_size
            tem_emb.append(self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)])

        if not self.first:
            if self.if_feedback:
                if self.if_forward:
                    node_forward_emb = node_forward_emb[0]
                    hidden_forward_emb = hidden_forward_emb[0]
                    hidden_forward_emb = \
                        self.forward_att(hidden_forward_emb, hidden_forward_emb, hidden_forward_emb)[0]
                    hidden_forward_emb = self.forward_fc(hidden_forward_emb)
                    node_forward_emb = [node_forward_emb * hidden_forward_emb]
                if self.if_backward:
                    node_backward_emb = node_backward_emb[0]
                    hidden_backward_emb = hidden_backward_emb[0]
                    hidden_backward_emb = \
                        self.backward_att(hidden_backward_emb, hidden_backward_emb, hidden_backward_emb)[0]
                    hidden_backward_emb = self.backward_fc(hidden_backward_emb)
                    node_backward_emb = [node_backward_emb * hidden_backward_emb]

        if self.graph_num > 1:
            hidden_forward = []
            hidden_backward = []
            if self.if_forward:
                forward_emb = torch.cat(time_series_emb + predict_emb + node_forward_emb + tem_emb, dim=2)
                hidden_forward = self.fusion_graph_model(forward_emb)
                hidden_forward = [self.fusion_forward_linear(hidden_forward)]
            if self.if_backward:
                backward_emb = torch.cat(time_series_emb + predict_emb + node_backward_emb + tem_emb, dim=2)
                hidden_backward = self.fusion_graph_model(backward_emb)
                hidden_backward = [self.fusion_backward_linear(hidden_backward)]

            hidden = torch.cat(hidden_forward + hidden_backward, dim=2)
            predict = self.fusion_model(hidden)
            return predict, hidden_forward, hidden_backward, node_forward_emb, node_backward_emb
        else:
            hidden = torch.cat(
                time_series_emb + predict_emb + node_forward_emb + node_backward_emb + tem_emb, dim=2)
            hidden = self.fusion_model(hidden)
            predict = self.fusion_linear(hidden)
        return predict, [hidden], [hidden], [hidden], node_forward_emb, node_backward_emb


class GraphMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x + self.fc2(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dimz, hidden_dim, dropout=0.2) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dimz, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_data)
        hidden = hidden + input_data
        return hidden

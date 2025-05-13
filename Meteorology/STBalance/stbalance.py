import torch
from torch import nn
import numpy as np
from .mlp import MultiLayerPerceptron, GraphMLP, FusionMLP
from .transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

device = 'cuda'


class STBalance(nn.Module):

    def __init__(self, configs):
        super().__init__()
        # attributes
        self.num_nodes = configs.num_nodes
        self.input_len = configs.seq_len
        self.output_len = configs.pred_len

        self.his_len = configs.his_len
        self.if_enhance = configs.if_enhance
        self.enhance_dim = configs.enhance_dim
        self.if_en = configs.if_en
        self.if_de = configs.if_de

        self.fusion_num_step = configs.fusion_num_step
        self.fusion_num_layer = configs.fusion_num_layer
        self.fusion_dim = configs.fusion_dim
        self.fusion_out_dim = configs.fusion_out_dim
        self.fusion_dropout = configs.fusion_dropout

        self.if_forward = configs.if_forward
        self.if_backward = configs.if_backward
        self.adj_mx = [torch.from_numpy(np.load('./adj_geo.npy')), torch.from_numpy(np.load('./adj_ang.npy'))]
        self.node_dim = configs.node_dim
        self.nhead = configs.nhead

        self.if_time_in_day = configs.if_time_in_day
        self.if_day_in_week = configs.if_day_in_week
        self.if_day_in_month = configs.if_day_in_month
        self.if_day_in_year = configs.if_day_in_year
        self.temp_dim = configs.temp_dim

        self.graph_num = 1 * self.if_forward + 1 * self.if_backward

        self.st_dim = (self.graph_num > 0) * self.node_dim + \
                      self.if_time_in_day * self.temp_dim + \
                      self.if_day_in_week * self.temp_dim + \
                      self.if_day_in_month * self.temp_dim + \
                      self.if_day_in_year * self.temp_dim

        self.output_dim = self.fusion_num_step * self.fusion_out_dim

        if self.if_forward:
            self.adj_mx_forward_encoder = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
            )
        if self.if_backward:
            self.adj_mx_backward_encoder = nn.Sequential(
                GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
            )

        self.fusion_layers = nn.ModuleList([
            FusionMLP(
                input_dim=self.st_dim + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                hidden_dim=self.st_dim + self.input_len + self.if_de * self.input_len + self.if_enhance * self.enhance_dim,
                out_dim=self.fusion_out_dim,
                graph_num=self.graph_num,
                first=True, configs=configs)
        ])
        for _ in range(self.fusion_num_step - 1):
            self.fusion_layers.append(
                FusionMLP(input_dim=self.st_dim + self.fusion_out_dim,
                          hidden_dim=self.st_dim + self.fusion_out_dim,
                          out_dim=self.fusion_out_dim,
                          graph_num=self.graph_num,
                          first=False, configs=configs)
            )
        if self.fusion_num_step > 1:
            self.regression_layer = nn.Sequential(
                *[MultiLayerPerceptron(input_dim=self.output_dim,
                                       hidden_dim=self.output_dim,
                                       dropout=self.fusion_dropout)
                  for _ in range(self.fusion_num_layer)],
                nn.Linear(in_features=self.output_dim, out_features=self.output_len, bias=True),
            )

        if self.if_enhance:
            self.long_linear = nn.Sequential(
                nn.Linear(in_features=self.his_len, out_features=self.enhance_dim, bias=True),
            )

        if self.if_en:
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(d_model=self.input_len, nhead=self.nhead, dim_feedforward=4 * self.input_len,
                                        batch_first=True), num_layers=self.nhead)
        if self.if_de:
            self.decoder = TransformerDecoder(
                TransformerDecoderLayer(d_model=self.input_len, nhead=self.nhead, dim_feedforward=4 * self.input_len,
                                        batch_first=True), num_layers=self.nhead)

    # def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor,
    #             batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_his,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        long_input_data_emb = []
        if self.if_enhance:
            long_input_data = x_his.transpose(1, 2)
            long_input_data = self.long_linear(long_input_data)
            long_input_data_emb.append(long_input_data)

        input_data = x_enc.transpose(1, 2)
        batch_size, num_nodes, _ = input_data.shape
        input_data_en = []
        input_data_de = []
        if self.if_en:
            input_data_en.append(self.encoder(input_data))
        else:
            input_data_en.append(input_data)
        if self.if_de:
            input_data_de.append(self.decoder(input_data, input_data_en[0]))

        time_series_emb = [torch.cat(long_input_data_emb + input_data_en + input_data_de, dim=2)]

        node_forward_emb = []
        node_backward_emb = []
        if self.if_forward:
            node_forward = self.adj_mx[0].to(device)
            node_forward = self.adj_mx_forward_encoder(node_forward.float().unsqueeze(0)).expand(batch_size, -1, -1)
            node_forward_emb.append(node_forward)

        if self.if_backward:
            node_backward = self.adj_mx[1].to(device)
            node_backward = self.adj_mx_backward_encoder(node_backward.unsqueeze(0)).expand(batch_size, -1, -1)
            node_backward_emb.append(node_backward)

        predicts = []
        predict_emb = []
        hidden_forward_emb = []
        hidden_backward_emb = []
        for index, layer in enumerate(self.fusion_layers):
            predict, \
            hidden_forward, hidden_backward, \
            node_forward_emb_out, node_backward_emb_out = layer(x_enc, x_mark_enc, time_series_emb, predict_emb,
                                                                node_forward_emb, node_backward_emb,
                                                                hidden_forward_emb, hidden_backward_emb, )
            predicts.append(predict)
            predict_emb = [predict]
            time_series_emb = []
            hidden_forward_emb = hidden_forward
            hidden_backward_emb = hidden_backward

            node_forward_emb = node_forward_emb_out
            node_backward_emb = node_backward_emb_out

        predicts = torch.cat(predicts, dim=2)
        if self.fusion_num_step > 1:
            predicts = self.regression_layer(predicts)
        return predicts.transpose(1, 2)

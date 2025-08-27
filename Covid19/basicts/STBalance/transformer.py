import copy
from torch import Tensor
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: Tensor) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, memory)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=False) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout,
                                               batch_first=batch_first)
        self.linear1 = nn.Linear(in_features=d_model, out_features=dim_feedforward, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=dim_feedforward, out_features=d_model, bias=True)
        self.activation = nn.ReLU()

    def forward(self, src: Tensor) -> Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + src2
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=False) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout,
                                               batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout,
                                                    batch_first=batch_first)
        self.linear1 = nn.Linear(in_features=d_model, out_features=dim_feedforward, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=dim_feedforward, out_features=d_model, bias=True)
        self.activation = nn.ReLU()

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + tgt2
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + tgt2
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + tgt2
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

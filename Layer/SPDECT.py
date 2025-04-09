from Model.ffn import *
from Model.GCN import *
from Layer.SPDECT_layer import SPDECTLayer
from torch_geometric.nn import GCNConv


class SPDECT(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(SPDECT, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN(in_channels, hidden_channels)
        self.SPDECTLayers = nn.ModuleList()
        for _ in range(0, layers):
            self.SPDECTLayers.append(
                SPDECTLayer(n_head, hidden_channels, dropout=dropout2))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.attn=[]

    def forward(self, x: torch.Tensor, calibration_mask=None, need_attn=False):

        x = self.attribute_encoder(x)
        for i in range(0, self.layers):
            x = self.SPDECTLayers[i](x, calibration_mask, need_attn)

        x = self.dropout(x)
        x = self.classifier(x)
        return x

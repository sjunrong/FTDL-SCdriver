import torch.nn as nn
from torch_geometric.nn import ChebConv


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c_1, hid_c_2, out_c, dropout, K):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels=in_c, out_channels=hid_c_1, K=K)
        self.conv2 = ChebConv(in_channels=hid_c_1, out_channels=hid_c_2, K=K)
        self.conv3 = ChebConv(in_channels=hid_c_2, out_channels=out_c, K=K)
        self.linear_1 = nn.Linear(in_features=64, out_features=300)
        self.linear_2 = nn.Linear(in_features=64, out_features=100)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, data, device):
        data = data.to(device)
        graph = data.edge_index
        gene_x = data.x
        output_conv_1 = self.conv1(gene_x, graph)
        output_linear_1 = self.linear_1(gene_x)
        output_1 = self.drop(self.act(output_conv_1) + output_linear_1)
        output_conv_2 = self.conv2(output_1, graph)

        output_2 = self.drop(self.act(output_conv_2))
        output_linear_2 = self.linear_2(gene_x)
        output_3 = output_2 + self.act(output_linear_2)
        output_conv_4 = self.conv3(output_3, graph)
        return output_conv_4

import torch
import torch.nn as nn
from math import sqrt
from torch_geometric.nn import ChebConv


class MultiHeadSelfAttention(nn.Module):
    """
    MultiHeadSelfAttention module
    """
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads  # 2
        dk = self.dim_k // nh  # dim_k of each head 1
        dv = self.dim_v // nh  # dim_v of each head 1

        q = self.linear_q(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk) 5.reshape(16,5,2)
        k = self.linear_k(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        v = self.linear_v(x.reshape(batch, dim_in)).reshape(batch, nh, dv)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, self.dim_v)  # batch, n, dim_v

        return att


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c_1, hid_c_2, out_c, K, dropout_rate):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels=in_c, out_channels=hid_c_1, K=K)
        self.conv2 = ChebConv(in_channels=hid_c_1, out_channels=hid_c_2, K=K)
        self.conv3 = ChebConv(in_channels=hid_c_2, out_channels=out_c, K=K)
        self.linear_1 = nn.Linear(in_features=64, out_features=300)
        self.linear_2 = nn.Linear(in_features=64, out_features=100)
        self.drop = nn.Dropout(dropout_rate)
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

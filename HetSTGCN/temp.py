import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        # GLU
        self.casual_conv = nn.Conv2d(in_channels, 2 * out_channels, (1, kernel_size))
        self.out_channels = out_channels
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.permute(0, 3, 1, 2) # (n组数据, 通道数, 节点数，时间长度？)

        # RELU
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # out = F.relu(temp + self.conv3(X))

        # GLU
        x_causal_conv = self.casual_conv(X)
        temp = self.conv1(X) # TODO: 残差连接应该怎么搞？？
        x_p = x_causal_conv[:, :self.out_channels, :, :]
        x_q = x_causal_conv[:, -self.out_channels:, :, :]
        out = torch.mul((x_p + temp), self.sigmoid(x_q))

        out = out.permute(0, 2, 3, 1)
        return out


class GraphConvolution(nn.Module):
    def __init__(self, A_wave, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.A_wave = A_wave
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, X, h0, layer):
        beta = math.log(0.5 / layer + 1) # lambda = 0.5
        # hi = torch.einsum("ij,jklm->kilm", [self.A_wave, X.permute(1, 0, 2, 3)])
        hi = torch.spmm(self.A_wave, X)
        support = (1 - 0.1) * hi + 0.1 * h0 # alpha = 0.1
        output = beta * torch.mm(support, self.weight) + (1 - beta) * support
        return output


class HetGraphConvolution(nn.Module):
    def __init__(self, A00, A01, A10, in_channels, out_channels):
        super(HetGraphConvolution, self).__init__()
        self.A00 = A00
        self.A01 = A01
        self.A10 = A10
        self.theta1 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.theta2 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.theta3 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta1.shape[1])
        self.theta1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.theta2.shape[1])
        self.theta2.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.theta3.shape[1])
        self.theta3.data.uniform_(-stdv, stdv)

    def forward(self, X0l, X1l, X00):
        u1 = torch.einsum("ij,jklm->kilm", [self.A00, X0l.permute(1, 0, 2, 3)])
        u1 = F.relu(torch.matmul(u1, self.theta1))

        u2 = torch.einsum("ij,jklm->kilm", [torch.mm(self.A01, self.A10), X0l.permute(1, 0, 2, 3)])
        u3 = torch.einsum("ij,jklm->kilm", [self.A01, X1l.permute(1, 0, 2, 3)])
        u2 = torch.sigmoid(torch.matmul(u2, self.theta2) + torch.matmul(u3, self.theta3))

        u1 = F.relu((u1 + u2) / 2)

        X0l1 = F.relu((1 - 0.1) * u1 + 0.1 * X00)
        return X0l1



class HetSTGCNBlock(nn.Module):
    def __init__(self, A_wave, As, in_channels, spatial_channels, out_channels, num_nodes):
        super(HetSTGCNBlock, self).__init__()
        self.A_wave = A_wave
        self.As = As
        self.A_01 = torch.tensor(self.As[2], dtype=torch.float32).cuda()
        self.A_10 = torch.tensor(self.As[3], dtype=torch.float32).cuda()
        self.A_00 = torch.tensor(self.As[6], dtype=torch.float32).cuda()
        self.A_11 = torch.tensor(self.As[7], dtype=torch.float32).cuda()

        self.temporal1 = TemporalConv(in_channels=in_channels, out_channels=out_channels)
        self.temporal2 = TemporalConv(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes) # TODO: what is it ?

        self.layers = 6
        self.convs1 = nn.ModuleList()
        for _ in range(self.layers): # 6层
            self.convs1.append(HetGraphConvolution(self.A_00, self.A_01, self.A_10, out_channels, spatial_channels))
        self.convs2 = nn.ModuleList()
        for _ in range(self.layers):
            self.convs2.append(HetGraphConvolution(self.A_11, self.A_10, self.A_01, out_channels, spatial_channels))

        self.act_fn = nn.ReLU()

    def forward(self, X):
        X = X.type(torch.FloatTensor).cuda()
        t1 = self.temporal1(X)

        X0 = t1[:, :81, :, :]
        X1 = t1[:, 81:, :, :]
        u0 = X0
        u1 = X1

        for _ in range(self.layers):
            u0 = self.convs1(u0, u1, X0)
            u1 = self.convs2(u1, u0, X1)
        
        tk = torch.cat((u0, u1), 1)
        tk = self.temporal2(tk)
        return self.batch_norm(tk)



class HetSTGCN(nn.Module):
    def __init__(self, num_nodes, A_wave, As, num_features, num_timesteps_input,
                num_timesteps_output):
        super(HetSTGCN, self).__init__()
        self.block1 = HetSTGCNBlock(A_wave=A_wave, As=As, in_channels=num_features, out_channels=64,
                                    spatial_channels=16, num_nodes=num_nodes)
        self.block2 = HetSTGCNBlock(A_wave=A_wave, As=As, in_channels=64, out_channels=64,
                                    spatial_channels=16, num_nodes=num_nodes)
        # 这里为什么需要last_temporal
        self.last_temporal = TemporalConv(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64, num_timesteps_output)
    
    def forward(self, X):
        out1 = self.block1(X)
        out2 = self.block2(out1)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4
    
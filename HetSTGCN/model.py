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


class HetSTGCNBlock(nn.Module):
    def __init__(self, A_wave, in_channels, spatial_channels, out_channels, num_nodes):
        super(HetSTGCNBlock, self).__init__()
        self.A_wave = A_wave
        self.temporal1 = TemporalConv(in_channels=in_channels, out_channels=out_channels)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TemporalConv(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes) # TODO: what is it ?

        # 2层卷积层
        self.Theta2 = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.Theta3 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))

        self.reset_parameters() # TODO: what is it?

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.Theta1.shape[1])
        # self.Theta1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.Theta2.shape[1])
        self.Theta2.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.Theta3.shape[1])
        self.Theta3.data.uniform_(-stdv, stdv)

    def forward(self, X):
        X = X.type(torch.FloatTensor).cuda()
        t = self.temporal1(X)

        A_hat = torch.tensor(self.A_wave, dtype=torch.float32).cuda()

        # 1层卷积层
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)]) # TODO: what is it?   
        # t2 = F.relu(torch.matmul(lfs, self.Theta1)) # TODO: what is it?

        # t3 = self.temporal2(t2)
        # return self.batch_norm(t3) # TODO: Why?

        # 2层卷积层
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)]) # TODO: what is it?   
        t2 = F.relu(torch.matmul(lfs, self.Theta2)) # TODO: what is it?

        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t2.permute(1, 0, 2, 3)]) # TODO: what is it?   
        t3 = F.relu(torch.matmul(lfs, self.Theta3)) # TODO: what is it?

        t4 = self.temporal2(t3)
        return self.batch_norm(t4) # TODO: Why?


class HetSTGCN(nn.Module):
    def __init__(self, num_nodes, A_wave, num_features, num_timesteps_input,
                num_timesteps_output):
        super(HetSTGCN, self).__init__()
        self.block1 = HetSTGCNBlock(A_wave=A_wave, in_channels=num_features, out_channels=64,
                                    spatial_channels=16, num_nodes=num_nodes)
        self.block2 = HetSTGCNBlock(A_wave=A_wave, in_channels=64, out_channels=64,
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
    
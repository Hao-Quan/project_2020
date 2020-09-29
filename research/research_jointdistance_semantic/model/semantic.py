# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        #self.bn = nn.BatchNorm1d(dim* 25)
        self.bn = nn.BatchNorm1d(dim * 25 * 2)

    def forward(self, x):
        bs, c, num_joints, step, h = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

# class Model(nn.Module):
#     def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
#         super(Model, self).__init__()
#
#         if graph is None:
#             raise ValueError()
#         else:
#             Graph = import_class(graph)
#             self.graph = Graph(**graph_args)
#
#         A = self.graph.A
#         self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
#
#         self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
#         self.l2 = TCN_GCN_unit(64, 64, A)
#         self.l3 = TCN_GCN_unit(64, 64, A)
#         self.l4 = TCN_GCN_unit(64, 64, A)
#         self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
#         self.l6 = TCN_GCN_unit(128, 128, A)
#         self.l7 = TCN_GCN_unit(128, 128, A)
#         self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
#         self.l9 = TCN_GCN_unit(256, 256, A)
#         self.l10 = TCN_GCN_unit(256, 256, A)
#
#         self.fc = nn.Linear(256, num_class)
#         nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
#         bn_init(self.data_bn, 1)
#
#     def forward(self, x):
#         N, C, T, V, M = x.size()
#
#         x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
#         x = self.data_bn(x)
#         x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
#
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         x = self.l4(x)
#         x = self.l5(x)
#         x = self.l6(x)
#         x = self.l7(x)
#         x = self.l8(x)
#         x = self.l9(x)
#         x = self.l10(x)
#
#         # N*M,C,T,V
#         c_new = x.size(1)
#         x = x.view(N, M, c_new, -1)
#         x = x.mean(3).mean(1)
#
#         return self.fc(x)


class Model(nn.Module):
    def __init__(self, num_class, num_point, num_person, graph, graph_args, bias=True):
        super(Model, self).__init__()

        self.dim1 = 256
        # self.dim1 = 128
        #self.dataset = dataset
        self.seg = 1
        num_joint = 25
        #N = batch_size
        # N = 3
        # M = 2
        # T = 300
        in_channels = 3

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        #if args.train:
        # self.spa = self.one_hot(N*M, num_joint, self.seg*T)
        # self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        # self.tem = self.one_hot(N*M, self.seg, num_joint)
        # self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        # self.spa = self.one_hot(M, num_joint, self.seg * T)
        # self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        # self.tem = self.one_hot(M, self.seg, num_joint)
        # self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        # else:
        #     self.spa = self.one_hot(32 * 5, num_joint, self.seg)
        #     self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        #     self.tem = self.one_hot(32 * 5, self.seg, num_joint)
        #     self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=False, bias=bias)
        self.dif_embed = embed(3, 64, norm=False, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        # Dynamic Representation
        # bs, step, dim = input.size()
        # bs, dim, step, num_joints, h = input.size()
        N, C, T, V, M = input.size()

        self.spa = self.one_hot(N * M, V, self.seg * T)
        self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(N * M, self.seg, V)
        self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        input = input.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        input = self.data_bn(input)
        #num_joints = dim // 3
        #input = input.view((bs, step, num_joints, dim, h))
        input = input.view(N*M, T, V, C)
        #input = input.permute(0, 3, 2, 1, 4).contiguous()
        input = input.permute(0, 3, 2, 1).contiguous()

        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(N*M, dif.size(1), V, 1).zero_(), dif], dim=-1)
        #should be: dif.new(bs, dif.size(1), num_joints, step-1, h).zero_()

        # # joint - spatial embed
        # pos = self.joint_embed(input)
        # # joint - temporal embed
        # dif = self.dif_embed(dif)
        # # joint distance - spatial embed
        # tem1 = self.tem_embed(self.tem)
        # # joint distance - temporal embed
        # dif = self.dif_embed(dif)

        # original version
        pos = self.joint_embed(input)
        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)

        dy = pos + dif
        # Joint-level Module
        # print("pos: {}".format(pos.size()))
        # print("dif: {}".format(dif.size()))
        # print("dy: {}".format(dy.size()))
        # print("tem1: {}".format(tem1.size()))
        # print("spa1: {}".format(spa1.size()))
        # input = torch.cat([dy, spa1], 1)

        #input = dy
        input = torch.cat([dy, spa1], 1)
        g = self.compute_g1(input)
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)
        # Frame-level Module
        input = input + tem1
        input = self.cnn(input)
        # Classification
        output = self.maxpool(input)    # output (2, 512, 1, 1)  input (2, 512, 1, 20)
        output = torch.flatten(output, 1) # output (2, 512)   input (2, 512, 1, 1)
        output = output.view(N, output.size(1), M)
        output = output.mean(2)
        output = self.fc(output) # output (2, 60)   input (2, 512)

        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot
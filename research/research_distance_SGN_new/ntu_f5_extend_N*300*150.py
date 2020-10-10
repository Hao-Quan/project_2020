import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataGenNet(nn.Module):

    def __init__(self):
        super(DataGenNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(300, 75, kernel_size=1)
        self.bn = nn.BatchNorm2d(75)
        self.rl = nn.ReLU(75)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.relu(self.conv1(x))
        x = self.conv1(x)
        x = self.bn(x)
        x = self.rl(x)

        return x


dgnet = DataGenNet()
print(dgnet)

params = list(dgnet.parameters())
print(len(params))
print(params[0].size())


#input = torch.randn(100, 300, 300, 2)

# path = '../ntu_light/h5/xsub/joint_distance/'
path = '/data/ntu/h5/xsub/joint_distance/'

sets = {
    'x_joint_distance_flatten', 'valid_x_joint_distance_flatten', 'test_x_joint_distance_flatten'
}

for data in sets:

    tmp = np.load(path + '{}.npy'.format(data), mmap_mode='r')
    
    tmp = torch.tensor(tmp)
    dgnet = dgnet.double()
    out = dgnet(tmp)
    out = out.view(out.size(0), 150, 300)
    out = out.permute(0, 2, 1)
    out = out.detach().numpy()
    
    with open(path + '{}_N*300*150.npy'.format(data), 'wb') as fs:
        np.save(fs, out)

    print("Write: " + data + " N*300*150")
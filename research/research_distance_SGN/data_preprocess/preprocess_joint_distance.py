import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Local

train_X = np.load('../data/ntu_light/xsub/train_data_joint.npy', mmap_mode='r')

# val_X = np.load('/data/ntu/xsub/joint_distance/val_data_joint_distance.npy', mmap_mode='r')

# train_X = np.load('/data/ntu/xsub/joint_distance/train_data_joint_distance.npy', mmap_mode='r')

# val_X = np.load('./data/ntu_light/xsub/joint_distance/val_data_joint_distance.npy', mmap_mode='r')

# Westworld
# train_X = np.load('/data/ntu/xsub/train_data_joint.npy', mmap_mode='r')
# with open('/data/ntu/xsub/train_label.pkl', 'rb') as f:
#     train_Y = pkl.load(f)
# val_X = np.load('/data/ntu/xsub/val_data_joint.npy', mmap_mode='r')
# with open('/data/ntu/xsub/val_label.pkl', 'rb') as f:
#     val_Y = pkl.load(f)

num_array = train_X.shape[0]
filled = np.empty([num_array, 300, 25, 25, 2])

for n in tqdm(range(num_array)):
    for i in range(0, 300):
        for j in range(25):
            for k in range(j, 25):
                if k == j:
                    filled[n, i, k, j] = 0
                else:
                    filled[n, i, k, j] = np.sqrt(np.sum((train_X[n,:,i,k,:] - train_X[n,:,i,j,:])**2, 0))
                    filled[n, i, j, k] = filled[n, i, k, j]


# with open('/data/ntu/xsub/joint_distance/train_data_joint_distance.npy', 'wb') as f:
#     np.save(f, filled)
# with open('./data/ntu_light/xsub/joint_distance/val_data_joint_distance.npy', 'wb') as f:
#     np.save(f, filled)

# Westworld
# with open('/data/ntu/xsub/joint_distance/val_data_joint_distance.npy', 'wb') as f:
#     np.save(f, filled)


# num_array = train_X.shape[0]
#
# filled_flatten = np.empty([num_array, 300, 300, 2])
#
# for n in tqdm(range(num_array)):
#     for i in range(0, 300):
#         for j in range(25):
#             for k in range(j + 1, 25):
#                 filled_flatten[n, i, j] = train_X[n, i, j, k]
#
# with open('/data/ntu/xsub/joint_distance/train_data_joint_distance_flatten.npy', 'wb') as f:
#     np.save(f, filled_flatten)
#
# print("")
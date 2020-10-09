import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
#from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# model = torch.load("results/NTU/SGN/1_best.pth")
# model.eval()
#confusion_matrix = confusion_matrix(y_test, predictions).astype(np.float)
print("ciao")

# train_X = np.load('./data/ntu/xsub/train_data_joint.npy', mmap_mode='r')
# val_X = np.load('./data/ntu_light/xsub/val_data_joint.npy', mmap_mode='r')
train_X_distance = np.load('./data/ntu_light/xsub/joint_distance/train_data_joint_distance.npy', mmap_mode='r')

train_X_distance_flatten = np.load('./data/ntu_light/xsub/joint_distance/train_data_joint_distance_flatten.npy', mmap_mode='r')

# Westworld

a = np.zeros([2, 3, 5])
b = np.zeros([2, 3, 5])
c = np.concatenate([a, b], axis=1)


print("")
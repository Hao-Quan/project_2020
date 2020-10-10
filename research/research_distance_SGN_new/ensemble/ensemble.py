import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    r1 = np.load('0_score_joint.npy', mmap_mode='r')
    r2 = np.load('0_score_distance.npy', mmap_mode='r')

    label = np.load('../data/ntu/h5/xsub/test_y.npy', mmap_mode='r')

    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label)-7)):
        # _, l = label[:, i]
        # _, r11 = list(r1.items())[i]
        # _, r22 = list(r2.items())[i]
        # r = r11 + r22 * arg.alpha
        #
        l = label[i]
        r11 =r1[i]
        r22 = r2[i]
        r = r11 + r22
        #r = r11
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

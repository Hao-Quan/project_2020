import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    r1 = np.load('0_score_joint_author.npy', mmap_mode='r')
    # r1 = np.load('0_score_joint.npy', mmap_mode='r')
    r2 = np.load('0_score_joint_and_distance_with_pretrained.npy', mmap_mode='r')

    label = np.load('../data/ntu/h5/xsub/test_y.npy', mmap_mode='r')

    right_num = total_num = right_num_5 = 0


    alpha = 22
    list_pred = []
    for i in range(len(label) - 7):
        l = label[i]
        r11 = r1[i]
        r22 = r2[i]
        r = alpha * r11 + r22
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1

        # append predict label in list for Confusion matrix
        list_pred.append(r)

    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    with open('./xsub/xsub_predict_label.npy', 'wb') as f:
        print("save ./xsub/xsub_predict_label.npy")
        np.save(f, list_pred)


    # best_acc = -1
    # best_alpha = -1
    #
    # for alpha in np.arange(0.5,50,0.5):
    #     for i in range(len(label)-7):
    #         # _, l = label[:, i]
    #         # _, r11 = list(r1.items())[i]
    #         # _, r22 = list(r2.items())[i]
    #         # r = r11 + r22 * arg.alpha
    #         #
    #         l = label[i]
    #         r11 =r1[i]
    #         r22 = r2[i]
    #         r = alpha*r11 + r22
    #         #r = r11
    #         rank_5 = r.argsort()[-5:]
    #         right_num_5 += int(int(l) in rank_5)
    #         r = np.argmax(r)
    #         right_num += int(r == int(l))
    #         total_num += 1
    #     acc = right_num / total_num
    #     acc5 = right_num_5 / total_num
    #
    #     if acc > best_acc:
    #         best_acc = acc
    #         best_alpha = alpha
    #
    #     print('Alpha: {:.4f}%'.format(alpha))
    #     print('Top1 Acc: {:.4f}%'.format(acc * 100))
    #     print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    #
    # print("Best acc: {:.4f} with Alpha {:.2f}".format(best_acc, best_alpha))

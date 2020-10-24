import sys
sys.path.extend(['../'])

import pickle
import argparse
import pandas as pd

from tqdm import tqdm

from data_gen.preprocess import pre_normalization


# https://arxiv.org/pdf/1604.02808.pdf, Section 3.2
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]

max_body_true = 2
max_body_kinect = 4

num_joint = 25
max_frame = 300


import numpy as np
import os

# NTU raw data version
# def read_skeleton_filter(file):
#     with open(file, 'r') as f:
#         # Jump header row
#         next(f)
#         row_data = f.readline()
#
#         skeleton_sequence = {}
#         skeleton_sequence['numFrame'] = int(row_data[0])
#         skeleton_sequence['frameInfo'] = []
#         # num_body = 0
#         for t in range(skeleton_sequence['numFrame']):
#             frame_info = {}
#             frame_info['numBody'] = int(row_data[2])
#             frame_info['bodyInfo'] = []
#
#             for m in range(frame_info['numBody']):
#                 body_info = {}
#                 # body_info_key = [
#                 #     'bodyID', 'clipedEdges', 'handLeftConfidence',
#                 #     'handLeftState', 'handRightConfidence', 'handRightState',
#                 #     'isResticted', 'leanX', 'leanY', 'trackingState'
#                 # ]
#                 # body_info = {
#                 #     k: float(v)
#                 #     for k, v in zip(body_info_key, f.readline().split())
#                 # }
#                 body_info['numJoint'] = 25
#                 body_info['jointInfo'] = []
#                 for v in range(body_info['numJoint']):
#                     joint_info_key = [
#                         'x', 'y', 'z',
#                         'depthX', 'depthY',
#                         'orientationX', 'orientationY', 'orientationZ', 'orientationW',
#                         'trackingState'
#                     ]
#
#                     # joint_info = {
#                     #     k: float(v)
#                     #     for k, v in zip(joint_info_key, f.readline().split())
#                     # }
#                     row_data = [row_data[3], row_data[4], row_data[5],
#                                 row_data[6], row_data[7],
#                                 row_data[8], row_data[9], row_data[10], row_data[11],
#                                 row_data[12]]
#
#
#                     joint_info = {
#                         k: float(v)
#                         for k, v in zip(joint_info_key, row_data.split())
#                     }
#
#                     body_info['jointInfo'].append(joint_info)
#                 frame_info['bodyInfo'].append(body_info)
#             skeleton_sequence['frameInfo'].append(frame_info)
#
#     return skeleton_sequence

# csv version
def read_skeleton_filter(file):
    df = pd.read_csv(file)
    saved_column = df.columns
    skeleton_sequence = {}
    # get #row in each csv
    skeleton_sequence['numFrame'] = len(df.index)
    skeleton_sequence['frameInfo'] = []

    # how many people in single frame
    if (df.iloc[0][1] == df.iloc[1][1]):
        numBody = 1
        step_frame = 1
    else:
        numBody = 2
        step_frame = 2

    for t in range(0, skeleton_sequence['numFrame'], step_frame):
        frame_info = {}

        frame_info['numBody'] = numBody

        frame_info['bodyInfo'] = []

        for m in range(frame_info['numBody']):
            body_info = {}
            body_info_key = [
                'bodyID'
            ]
            body_info = {
                k: int(v)
                for k, v in zip(body_info_key, [df.iloc[t + m][1]])
            }

            body_info['numJoint'] = 25
            body_info['jointInfo'] = []

            for v in range(body_info['numJoint']):
                joint_info_key = [
                    'x', 'y', 'z',
                    'depthX', 'depthY',
                    'orientationX', 'orientationY', 'orientationZ', 'orientationW',
                    'trackingState'
                ]
                row_data = [df.iloc[t + m][3 + v * 10], df.iloc[t + m][4 + v * 10], df.iloc[t + m][5 + v * 10],
                            df.iloc[t + m][6 + v * 10], df.iloc[t + m][7 + v * 10],
                            df.iloc[t + m][8 + v * 10], df.iloc[t + m][9 + v * 10], df.iloc[t + m][10 + v * 10], df.iloc[t + m][11 + v * 10],
                            df.iloc[t + m][12 + v * 10]]
                joint_info = {
                    k: float(v)
                    for k, v in zip(joint_info_key, row_data)
                }
                body_info['jointInfo'].append(joint_info)
            frame_info['bodyInfo'].append(body_info)
        skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence

def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in sorted(os.listdir(data_path)):
        if filename in ignored_samples:
            continue

        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # Fill in the data tensor `fp` one training example a time
    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

    # s = 'A047_P001_G001_C001.csv'
    # i = 0

    data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
    fp[i, :, 0:data.shape[1], :, :] = data
    np.save('{}/{}_data_joint_no_normalized.npy'.format(out_path, part), fp)

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

    print("end one sample")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETRI Data Converter.')
    parser.add_argument('--data_path', default='../etri_raw/')
    parser.add_argument('--ignored_sample_path',
                        default=None)
    parser.add_argument('--out_folder', default='../data/etri/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)

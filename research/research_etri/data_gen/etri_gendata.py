# start - rotation.py
import math
import numpy as np
import multiprocessing

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)
# end - rotation.py

# start - preprocess.py
import sys
sys.path.extend(['../'])

from tqdm import tqdm

# from data_gen.rotation import *


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

# end - preprocess.py


import sys
sys.path.extend(['../'])

import pickle
import argparse
import pandas as pd

from tqdm import tqdm

#from data_gen.preprocess import pre_normalization

# NTU cross-subject, cross-view setting
# https://arxiv.org/pdf/1604.02808.pdf, Section 3.2
# training_subjects = [
#     1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
# ]
# training_cameras = [2, 3]

# ETRI cross-subject
training_subjects = [
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100
]
# training_subjects = [
#     1, 2
# ]

# ETRI cross-viwe setting
training_cameras = [1, 3, 4, 6, 7]

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

    if len(df.index) > 300:
        skeleton_sequence['numFrame'] = 300
    else:
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
    # np.save('{}/{}_data_joint_no_normalized.npy'.format(out_path, part), fp)

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

    #print("end one sample")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETRI Data Converter.')
    #parser.add_argument('--data_path', default='../etri_raw/')
    # parser.add_argument('--data_path', default='../data/etri/etri_raw_data/')
    # parser.add_argument('--data_path', default='../data/etri/etri_raw_data_light/')
    # parser.add_argument('--out_folder', default='../data/etri/etri_data')
    # parser.add_argument('--out_folder', default='../data/etri/etri_data_light')

    # Westworld
    parser.add_argument('--data_path', default='/data/etri/etri_raw_data/')
    parser.add_argument('--out_folder', default='/data/etri/etri_data/')

    # Local
    # parser.add_argument('--data_path', default='../data/etri/etri_raw_data/')
    # parser.add_argument('--out_folder', default='../data/etri/etri_data/')

    # Local light
    # parser.add_argument('--data_path', default='../data/etri/etri_raw_data_light/')
    # parser.add_argument('--out_folder', default='../data/etri/etri_data_light/')

    parser.add_argument('--ignored_sample_path',
                        default=None)

    # benchmark = ['xsub', 'xview']
    # part = ['train', 'val']
    benchmark = ['xsub']
    part = ['train', 'val']
    arg = parser.parse_args()

    # Single CPU version
    # for b in benchmark:
    #     for p in part:
    #         out_path = os.path.join(arg.out_folder, b)
    #         if not os.path.exists(out_path):
    #             os.makedirs(out_path)
    #         print(b, p)
    #         gendata(
    #             arg.data_path,
    #             out_path,
    #             arg.ignored_sample_path,
    #             benchmark=b,
    #             part=p)

    # Multi processing version
    processes = []
    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            # Multi-processing
            p = multiprocessing.Process(
                target=gendata,
                args=(
                    arg.data_path,
                    out_path,
                    arg.ignored_sample_path,
                    b,
                    p,
                )
            )
            processes.append(p)
            p.start()

    for process in processes:
        process.join()


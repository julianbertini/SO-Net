import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py
import json
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .augmentation import *


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

def load_file_names(root, name):
    """Opens file containing file names of data and returns them as
       a numpy array to be used in train.py for k-fold cross-validation

        root - root folder housing data
        name - name of the file
    """
    filenames = []
    with open(os.path.join(root, name), 'r') as f:
        for line in f:
            filenames.append(line[:-1])

    return np.asarray(filenames)


def make_dataset_shapenet_normal(root, mode):
    """ Gets the shuffled names of the files that
    will be considered as part of the training or testing split
    """
    if mode == 'train':
        f = open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r')
        file_name_list = json.load(f)
        f.close()
    elif mode == 'test':
        f = open(os.path.join(root, 'train_test_split', 'shuffled_test_file_list.json'), 'r')
        file_name_list = json.load(f)
        f.close()
    else:
        raise Exception('Mode should be train/test.')

    return file_name_list


class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 3

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''

        :param x: numpy array of Nxd
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class ShapeNetLoader(data.Dataset):
    def __init__(self, file_names, mode, opt):
        super(ShapeNetLoader, self).__init__()
        self.root = opt.dataroot
        self.opt = opt
        self.mode = mode

        self.node_num = opt.node_num
        self.rows = round(math.sqrt(self.node_num))
        self.cols = self.rows

        # self.dataset is a list if file names corresponding to the data files
        self.dataset = file_names 
        # ensure there is no batch-1 batch
        if np.size(self.dataset) % self.opt.batch_size == 1:
            current_size = np.size(self.dataset)
            print('Reducing dataset so there is no batch of 1...')
            print('Reducing from %d to %d' % (current_size, current_size-1))
            self.dataset = self.dataset[:-1]

        # load the folder-category txt
        self.categories = ['Vessel', 'Aneurysm']

        # kNN search on SOM nodes
        self.knn_builder = KNNBuilder(self.opt.som_k)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        data = np.load(os.path.join(self.root, file))
        pc_np = data['pc']
        sn_np = data['sn']
        # ground-truth label (e.g. Aneurysm (1) or Vessel (0))
        seg_np = data['part_label']
        # som node points
        som_node_np = data['som_node']
        # label is an integer representing the type of image (e.g. airplane, or car) for ShapeNet
        # For aneurysm, we only have 1 type of image: vessel segments with aneurysms.
        # So label will always be 0, denoting this.
        label = 0
        assert(label >= 0)


        if self.opt.input_pc_num < pc_np.shape[0]:
            chosen_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
            pc_np = pc_np[chosen_idx, :]
            sn_np = sn_np[chosen_idx, :]
            seg_np = seg_np[chosen_idx]
        else:
            chosen_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num-pc_np.shape[0], replace=True)
            pc_np_redundent = pc_np[chosen_idx, :]
            sn_np_redundent = sn_np[chosen_idx, :]
            seg_np_redundent = seg_np[chosen_idx]
            pc_np = np.concatenate((pc_np, pc_np_redundent), axis=0)
            sn_np = np.concatenate((sn_np, sn_np_redundent), axis=0)
            seg_np = np.concatenate((seg_np, seg_np_redundent), axis=0)

        # augmentation
        if self.mode == 'train':
            # rotate by random degree over model z (point coordinate y) axis
            # pc_np = rotate_point_cloud(pc_np)
            # som_node_np = rotate_point_cloud(som_node_np)

            # rotate by 0/90/180/270 degree over model z (point coordinate y) axis
            # pc_np = rotate_point_cloud_90(pc_np)
            # som_node_np = rotate_point_cloud_90(som_node_np)

            # random jittering
            pc_np = jitter_point_cloud(pc_np)
            sn_np = jitter_point_cloud(sn_np)
            som_node_np = jitter_point_cloud(som_node_np, sigma=0.04, clip=0.1)

            # random scale
            scale = np.random.uniform(low=0.8, high=1.2)
            pc_np = pc_np * scale
            sn_np = sn_np * scale
            som_node_np = som_node_np * scale

            # random shift
            # shift = np.random.uniform(-0.1, 0.1, (1,3))
            # pc_np += shift
            # som_node_np += shift

        # convert to tensor
        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 3xN
        seg = torch.from_numpy(seg_np.astype(np.int64))  # N

        # som
        som_node = torch.from_numpy(som_node_np.transpose().astype(np.float32))  # 3xnode_num

        # kNN search: som -> som
        if self.opt.som_k >= 2:
            D, I = self.knn_builder.self_build_search(som_node_np)
            som_knn_I = torch.from_numpy(I.astype(np.int64))  # node_num x som_k
        else:
            som_knn_I = torch.from_numpy(np.arange(start=0, stop=self.opt.node_num, dtype=np.int64).reshape(
                (self.opt.node_num, 1)))  # node_num x 1

        return pc, sn, label, seg, som_node, som_knn_I



if __name__=="__main__":
    # dataset = make_dataset_modelnet40('/ssd/dataset/modelnet40_ply_hdf5_2048/', True)
    # print(len(dataset))
    # print(dataset[0])


    class VirtualOpt():
        def __init__(self):
            self.load_all_data = False
            self.root = '/content/drive/MyDrive/data/IntrA/annotated/ad'
            self.input_pc_num = 8000
            self.batch_size = 20
            self.node_num = 49
            self.som_k = 9
    opt = VirtualOpt()
    filenames = load_file_names(opt.root, 'file_name_list.txt')
    trainset = ShapeNetLoader(filenames, 'train', opt)
    print(len(trainset))
    pc, sn, label, seg, som_node, som_knn_I = trainset[0]

    print(seg)

    x_np = pc.numpy().transpose()
    node_np = som_node.numpy().transpose()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_np[:, 0].tolist(), x_np[:, 1].tolist(), x_np[:, 2].tolist(), s=1)
    ax.scatter(node_np[:, 0].tolist(), node_np[:, 1].tolist(), node_np[:, 2].tolist(), s=6, c='r')
    plt.show()





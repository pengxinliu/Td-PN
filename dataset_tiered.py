# -------------------------------------
# Project: Transductive Prototypical Network for Few-shot Classification
# Date: 2020.1.11
# Author: Pengxin Liu
# All Rights Reserved
# -------------------------------------

# coding: utf-8
from __future__ import print_function
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
import cv2


class dataset_tiered(object):
    def __init__(self, n_examples, n_episodes, split, args):
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.n_examples = n_examples
        self.n_episodes = n_episodes
        self.split = split
        self.seed = args['seed']
        self.root_dir = './data/tiered-imagenet'

        self.iamge_data = []
        self.dict_index_label = []

    def load_data_pkl(self):
        """
            load the pkl processed tieredImagenet into label
            maintain label data dictionary for indexes
        """
        labels_name = '{}/{}_labels.pkl'.format(self.root_dir, self.split)
        images_name = '{}/{}_images.npz'.format(self.root_dir, self.split)
        print('labels:', labels_name)
        print('images:', images_name)

        # decompress images if npz not exits
        if not os.path.exists(images_name):
            png_pkl = images_name[:-4] + '_png.pkl'
            if os.path.exists(png_pkl):
                decompress(images_name, png_pkl)
            else:
                raise ValueError('path png_pkl not exits')

        if os.path.exists(images_name) and os.path.exists(labels_name):
            try:
                with open(labels_name, 'rb') as f:
                    data = pkl.load(f, encoding='bytes')
                    label_specific = data["label_specific"]
            except:
                with open(labels_name, 'rb') as f:
                    data = pkl.load(f, encoding='bytes')
                    label_specific = data[b'label_specific']
            print('read label data:{}'.format(len(label_specific)))
        labels = label_specific

        with np.load(images_name, mmap_mode="r", encoding='latin1') as data:
            image_data = data["images"]
            print('read image data:{}'.format(image_data.shape))

        n_classes = np.max(labels) + 1

        print('n_classes:{}'.format(n_classes))
        dict_index_label = {}  # key:label, value:idxs

        for cls in range(n_classes):
            idxs = np.where(labels == cls)[0]
            nums = idxs.shape[0]
            np.random.RandomState(self.seed).shuffle(idxs)
            n_label = int(nums)

            dict_index_label[cls] = idxs[0:n_label]

        self.image_data = image_data
        self.dict_index_label = dict_index_label
        self.n_classes = n_classes
        print(dict_index_label[0])

    def next_data(self, n_way, n_shot, n_query):
        """
            get support, query data from n_way
        """
        support = np.zeros([n_way, n_shot, self.im_height, self.im_width, self.channels], dtype=np.float32)
        query = np.zeros([n_way, n_query, self.im_height, self.im_width, self.channels], dtype=np.float32)

        selected_classes = np.random.permutation(self.n_classes)[:n_way]
        for i, cls in enumerate(selected_classes[0:n_way]):  # train way
            idx = self.dict_index_label[cls]
            np.random.RandomState().shuffle(idx)
            idx1 = idx[0:n_shot + n_query]
            support[i] = self.image_data[idx1[:n_shot]]
            query[i] = self.image_data[idx1[n_shot:]]

        support_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        return support, support_labels, query, query_labels


def decompress(path, output):
    with open(output, 'rb') as f:
        array = pkl.load(f, encoding='bytes')
    images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
    for ii, item in tqdm(enumerate(array), desc='decompress'):
        im = cv2.imdecode(item, 1)
        images[ii] = im
    np.savez(path, images=images)

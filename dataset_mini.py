# -------------------------------------
# Project: Transductive Prototypical Network for Few-shot Classification
# Date: 2020.1.11
# Author: Pengxin Liu
# All Rights Reserved
# -------------------------------------

# coding: utf-8
from __future__ import print_function
import numpy as np
from PIL import Image
import pickle as pkl
import os
import glob
import csv


class dataset_mini(object):
    def __init__(self, n_examples, n_episodes, split, args):
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.n_examples = n_examples
        self.n_episodes = n_episodes
        self.split = split
        self.seed = args['seed']
        self.root_dir = './data/mini-imagenet'
        self.dataset_l = []

        self.args = args

    def load_data(self):
        """
            Load data into memory
        """
        print('Loading {} dataset'.format(self.split))
        data_split_path = os.path.join(self.root_dir, 'splits', '{}.csv'.format(self.split))
        with open(data_split_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data_classes = {}
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                data_classes[row[1]] = 1
            data_classes = data_classes.keys()
        print(data_classes)

        n_classes = len(data_classes)
        print('n_classes:{}'.format(n_classes))
        dataset_l = np.zeros([n_classes, self.n_examples, self.im_height, self.im_width, self.channels], dtype=np.float32)

        for i, cls in enumerate(data_classes):
            im_dir = os.path.join(self.root_dir, 'data/{}/'.format(self.split), cls)
            im_files = sorted(glob.glob(os.path.join(im_dir, '*.jpg')))
            np.random.RandomState(self.seed).shuffle(im_files)
            for j, im_file in enumerate(im_files):
                im = np.array(Image.open(im_file).resize((self.im_width, self.im_height)),
                              np.float32, copy=False)
                if j < self.n_examples:
                    dataset_l[i, j] = im
        print('labeled data:', np.shape(dataset_l))

        self.dataset_l = dataset_l
        self.n_classes = n_classes

    def load_data_pkl(self):
        """
            load the pkl processed mini-imagenet into label
        """
        pkl_name = '{}/mini-imagenet-cache-{}.pkl'.format(self.root_dir, self.split)
        print('Loading pkl dataset: {} '.format(pkl_name))

        try:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f, encoding='bytes')
                image_data = data[b'image_data']
                class_dict = data[b'class_dict']
        except:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f)
                image_data = data['image_data']
                class_dict = data['class_dict']

        print(data.keys(), image_data.shape, class_dict.keys())
        data_classes = sorted(class_dict.keys())  # sorted to keep the order

        n_classes = len(data_classes)
        print('n_classes:{}'.format(n_classes))
        dataset_l = np.zeros([n_classes, self.n_examples, self.im_height, self.im_width, self.channels], dtype=np.float32)

        for i, cls in enumerate(data_classes):
            idxs = class_dict[cls]
            np.random.RandomState(self.seed).shuffle(idxs)
            dataset_l[i] = image_data[idxs[0:self.n_examples]]
        print('labeled data:', np.shape(dataset_l))

        self.dataset_l = dataset_l
        self.n_classes = n_classes

        del image_data

    def next_data(self, n_way, n_shot, n_query):
        """
            get support, query data from n_way
        """
        support = np.zeros([n_way, n_shot, self.im_height, self.im_width, self.channels], dtype=np.float32)
        query = np.zeros([n_way, n_query, self.im_height, self.im_width, self.channels], dtype=np.float32)

        selected_classes = np.random.permutation(self.n_classes)[:n_way]
        for i, cls in enumerate(selected_classes[0:n_way]):  # train way
            idx1 = np.random.permutation(self.n_examples)[:n_shot + n_query]
            support[i] = self.dataset_l[cls, idx1[:n_shot]]
            query[i] = self.dataset_l[cls, idx1[n_shot:]]

        support_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)

        return support, support_labels, query, query_labels

# -------------------------------------
# Project: Transductive Prototypical Network for Few-shot Classification
# Date: 2020.1.11
# Author: Pengxin Liu
# All Rights Reserved
# -------------------------------------

# coding: utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf


class models(object):
    def __init__(self, args):
        # parameters
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']
        self.args = args

        # placeholders for data and label
        self.x = tf.placeholder(tf.float32, [None, None, self.im_height, self.im_width, self.channels])
        self.ys = tf.placeholder(tf.int64, [None, None])
        self.q = tf.placeholder(tf.float32, [None, None, self.im_height, self.im_width, self.channels])
        self.y = tf.placeholder(tf.int64, [None, None])
        self.phase = tf.placeholder(tf.bool, name='phase')

        self.alpha = args['alpha']

    def conv_block(self, inputs, out_channels, pool_pad='VALID', name='conv'):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding="same")
            conv = tf.contrib.layers.batch_norm(conv, is_training=self.phase, decay=0.999, epsilon=1e-3, scale=True,
                                                center=True)
            conv = tf.nn.relu(conv)
            conv = tf.contrib.layers.max_pool2d(conv, 2, padding=pool_pad)

            return conv

    def encoder(self, x, h_dim, z_dim, reuse=False):
        # Feature embedding network
        with tf.variable_scope('encoder', reuse=reuse):
            net = self.conv_block(x, h_dim, name='conv_1')
            net = self.conv_block(net, h_dim, name='conv_2')
            net = self.conv_block(net, h_dim, name='conv_3')
            net = self.conv_block(net, z_dim, name='conv_4')

            net = tf.contrib.layers.flatten(net)

            return net

    def topk(self, W, k):
        # choose top k query examples
        values, indices = tf.nn.top_k(W, k, sorted=False)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, k])
        full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], axis=2)
        full_indices = tf.reshape(full_indices, [-1, 2])

        topk_W = tf.sparse_to_dense(full_indices, tf.shape(W), sparse_values=1., default_value=0.,
                                    validate_indices=False)

        return topk_W

    def proto_refine(self, num_way, num_shot, emb_x, emb_q):
        # initialize and refine prototypes
        d = tf.shape(emb_q)[1]

        emb_p = tf.reduce_mean(tf.reshape(emb_x, [num_way, num_shot, -1]), 1)
        e_dists = tf.reduce_mean(tf.square(tf.expand_dims(emb_q, 1) - tf.expand_dims(emb_p, 0)), 2)
        assign = tf.nn.softmax(-e_dists)
        ind = tf.transpose(assign)
        # ind = yq > 0.5
        # ind = tf.cast(ind, dtype=tf.float32)
        ind = self.topk(ind, self.args['k'])
        Z = tf.reduce_sum(ind, axis=1)
        new_p = tf.matmul(ind, emb_q)
        num_shot = tf.cast(num_shot, tf.float32)
        opt_p = tf.tile(tf.expand_dims(1 / (num_shot + Z), 1), [1, d]) * (num_shot * emb_p + new_p)

        return opt_p, assign

    def loss_comp(self, num_way, num_query, opt_p, emb_q, assign):
        # compute loss and acc
        epsilon = np.finfo(float).eps

        # query labels prediction
        e_dists = tf.reduce_mean(tf.square(tf.expand_dims(emb_q, axis=1) - tf.expand_dims(opt_p, axis=0)), axis=2)
        yq = tf.nn.softmax(-e_dists)
        label = tf.argmax(yq, 1)

        # ground-truth computation
        gt = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_way), 1), [1, tf.cast(num_query, tf.int32)]), [-1])
        y_one_hot = tf.reshape(tf.one_hot(gt, depth=num_way), [num_way * num_query, -1])

        # loss computation
        ce_loss = self.alpha * y_one_hot * tf.log(assign + epsilon) + (1 - self.alpha) * (1 - y_one_hot) * tf.log(1 - assign + epsilon)
        ce_loss = tf.negative(ce_loss)
        ce_loss = tf.reduce_mean(tf.reduce_sum(ce_loss, 1))

        # only consider query examples acc
        acc = tf.reduce_mean(tf.to_float(tf.equal(label, tf.cast(gt, tf.int64))))

        return ce_loss, acc

    def construct(self):
        # construct the model
        num_way, num_shot, num_query = tf.shape(self.x)[0], tf.shape(self.x)[1], tf.shape(self.q)[1]
        x = tf.reshape(self.x, [-1, self.im_height, self.im_width, self.channels])
        q = tf.reshape(self.q, [-1, self.im_height, self.im_width, self.channels])
        emb_x = self.encoder(x, self.h_dim, self.z_dim)
        emb_q = self.encoder(q, self.h_dim, self.z_dim, reuse=True)

        opt_p, assign = self.proto_refine(num_way, num_shot, emb_x, emb_q)

        ce_loss, acc = self.loss_comp(num_way, num_query, opt_p, emb_q, assign)

        return ce_loss, acc

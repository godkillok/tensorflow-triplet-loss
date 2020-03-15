"""Test for the triplet loss computation."""

import numpy as np
import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model.triplet_loss import _pairwise_distances
from model.triplet_loss import _get_triplet_mask
from model.triplet_loss import _get_anchor_positive_triplet_mask
from model.triplet_loss import _get_anchor_negative_triplet_mask

def anchor_negative_triplet_mask():
    """Test the triplet loss with batch all triplet mining in a simple case.

    There is just one class in this super simple edge case, and we want to make sure that
    the loss is 0.
    """
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 1

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)
    tags=np.random.rand(5, feat_dim).astype(np.float32)
    loss_np = 0.0
    input_data = tf.constant([[1.0, 2, 3], [4.0, 5, 6], [7.0, 8, 9]])
    loss_tf = tf.nn.l2_normalize(input_data, dim = 0)
    a = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
    a = np.asarray(a)
    idx1 = tf.Variable([0, 2, 3, 1], tf.int32)
    #out1 = tf.nn.embedding_lookup(a, idx1)
    a_t = tf.transpose(a,[1,0])
    out1=tf.matmul(a,a_t)
    init = tf.global_variables_initializer()
    batch_size=a.shape[0]
    labels = tf.range(0,batch_size,1)
    # loss_tf, fraction = batch_all_triplet_loss(labels, embeddings, margin, squared=squared)
    with tf.Session() as sess:
        sess.run(init)
        #tf.initialize_all_variables().run()

        loss_tf_val = sess.run(labels)
        print(loss_tf_val)
    # assert np.allclose(loss_np, loss_tf_val)
    # assert np.allclose(fraction_val, 0.0)


# anchor_negative_triplet_mask()


def random_label(labels_lists,zero_pos3):
    rows = tf.shape(labels_lists)[0]
    # Number of zeros on each row
    zero_mask = tf.cast(tf.greater(labels_lists, 0), tf.int32)
    num_zeros = tf.reduce_sum(zero_mask, axis=1)
    # Random values
    r = tf.random_uniform([rows], 0, 1, dtype=tf.float32)
    # Multiply by the number of zeros to decide which of the zeros you pick
    zero_idx = tf.cast(tf.floor(r * tf.cast(num_zeros, r.dtype)), tf.int32)
    # Find the indices of the smallest values, which should be the zeros
    _, zero_pos = tf.nn.top_k(labels_lists, k=tf.maximum(tf.reduce_max(num_zeros), 1))
    # Select the corresponding position of each row
    labes = tf.gather_nd(zero_pos, tf.stack([tf.range(rows), zero_idx], axis=1))

    return result, labes,tf.stack([tf.range(rows), zero_idx], axis=1)

def dge():
    x = [[1, 3],
         [2, 4]]
    zero_pos=[[[1,2,3],[1,3,3]],
              [[1, 4, 3], [1, 5, 3]]
              ]

    x = np.asarray(x)
    zero_pos=np.asarray(zero_pos)
    #labes=random_label(x)
    rows=2
    result = random_label(x,zero_pos)
    with tf.Session() as sess:
        print(sess.run(result))
        print(sess.run(result))
        print(sess.run(result))
        print(sess.run(result))
        print(sess.run(result))
        print(sess.run(result))
dge()

def gegy():
    tf.gather_nd(zero_pos, tf.stack([tf.range(rows), zero_idx], axis=1))

def triplet_mask():
    """Test function _get_triplet_mask."""
    num_data = 64
    num_classes = 3
    num_data = 4
    num_classes = 3

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                distinct = (i != j and i != k and j != k)
                valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                mask_np[i, j, k] = (distinct and valid)
    labels = tf.range(0, num_classes, 1)
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)
    print(labels)
    mask_tf = _get_triplet_mask(labels)
    labels_equal=_get_anchor_negative_triplet_mask(labels)
    with tf.Session() as sess:
        mask_tf_val = sess.run(labels_equal)
        print(mask_tf_val)
    # assert np.allclose(mask_np, mask_tf_val)
# triplet_mask()
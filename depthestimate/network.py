import tflearn
import tensorflow as tf
from . import tf_nndistance


def build_graph(img_inp, factor=1):
    tflearn.init_graph(seed=1029, num_cores=2, gpu_memory_fraction=0.9, soft_placement=True)
    batch = tf.shape(img_inp)[0]
    distributed_points = factor * 256

    with tf.variable_scope('points'):
        x = img_inp
        # 240 320
        x = tf.image.resize_images(x, (192, 256))
        # 192 256
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x0 = x
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        # 96 128
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x1 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        # 48 64
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        # 24 32
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        # 12 16
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        # 6 8
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        # 3 4
        x_additional = tflearn.layers.core.fully_connected(x, factor*2048, activation='relu', weight_decay=1e-3, regularizer='L2')
        x = tflearn.layers.conv.conv_2d_transpose(x, 256, [5, 5], [6, 8], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 6 8
        x5 = tflearn.layers.conv.conv_2d(x5, 256, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x5, name="add1"), name="relu1")
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 128, [5, 5], [12, 16], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 12 16
        x4 = tflearn.layers.conv.conv_2d(x4, 128, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x4, name="add2"), name="relu2")
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [24, 32], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 24 32
        x3 = tflearn.layers.conv.conv_2d(x3, 64, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x3, name="add3"), name="relu3")
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 32, [5, 5], [48, 64], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 48 64
        x2 = tflearn.layers.conv.conv_2d(x2, 32, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x2, name="add4"), name="relu4")
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 16, [5, 5], [96, 128], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 96 128
        x1 = tflearn.layers.conv.conv_2d(x1, 16, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x1, name="add5"), name="relu5")
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 48 64
        x2 = tflearn.layers.conv.conv_2d(x2, 32, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x2, name="add6"), name="relu6")
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 24 32
        x3 = tflearn.layers.conv.conv_2d(x3, 64, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x3, name="add7"), name="relu7")
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (5, 5), strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 12 16
        x4 = tflearn.layers.conv.conv_2d(x4, 128, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x4, name="add8"), name="relu8")
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 6 8
        x5 = tflearn.layers.conv.conv_2d(x5, 256, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x5, name="add9"), name="relu9")
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2')
        # 3 4
        x_additional = tflearn.layers.core.fully_connected(x_additional, factor*2048, activation='linear', weight_decay=1e-4, regularizer='L2')
        x_additional = tf.nn.relu(tf.add(x_additional, tflearn.layers.core.fully_connected(x, factor*2048, activation='linear', weight_decay=1e-3, regularizer='L2'), name="add10"),
                                  name="relu10")
        x = tflearn.layers.conv.conv_2d_transpose(x, 256, [5, 5], [6, 8], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 6 8
        x5 = tflearn.layers.conv.conv_2d(x5, 256, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x5, name="add11"), name="relu11")
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x5 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 128, [5, 5], [12, 16], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 12 16
        x4 = tflearn.layers.conv.conv_2d(x4, 128, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x4, name="add12"), name="relu12")
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x4 = x
        x = tflearn.layers.conv.conv_2d_transpose(x, 64, [5, 5], [24, 32], strides=2, activation='linear', weight_decay=1e-5, regularizer='L2')
        # 24 32
        x3 = tflearn.layers.conv.conv_2d(x3, 64, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.nn.relu(tf.add(x, x3, name="add13"), name="relu13")
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')

        x_additional = tflearn.layers.core.fully_connected(x_additional, factor * 1024, activation='relu', weight_decay=1e-3, regularizer='L2')
        x_additional = tflearn.layers.core.fully_connected(x_additional, distributed_points * 3, activation='linear', weight_decay=1e-3, regularizer='L2')
        x_additional = tf.reshape(x_additional, (batch, distributed_points, 3))
        x = tflearn.layers.conv.conv_2d(x, 3, (3, 3), strides=1, activation='linear', weight_decay=1e-5, regularizer='L2')
        x = tf.reshape(x, (batch, 768, 3))
        x = tf.concat([x_additional, x], 1)
        x = tf.reshape(x, (batch, distributed_points + 768, 3), name='points')
    return x


def build_graph_training(img_inp, pc_gt, factor=1):
    x = build_graph(img_inp, factor)
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pc_gt, x)
    # dists_forward = dists_forward * tf.norm(pc_gt, axis=2)
    mindist_forward = dists_forward
    mindist_backword = dists_backward
    dists_forward = tf.reduce_mean(dists_forward)
    dists_backward = tf.reduce_mean(dists_backward)
    loss_nodecay = (dists_forward + dists_backward / 2.0)
    tf.summary.scalar('distance_from_gt_to_pred (forward)', dists_forward)
    tf.summary.scalar('distance_from_pred_to_gt (backward)', dists_backward)
    tf.summary.scalar('pc_loss', loss_nodecay)
    return x, mindist_forward, mindist_backword, loss_nodecay, dists_forward, dists_backward

import cv2
import numpy as np
import tensorflow as tf
import sys

nn_distance_module = tf.load_op_library('depthestimate/tf_nndistance_so.so')

BATCH_SIZE = 1
HEIGHT = 192
WIDTH = 256


def loadModel(weightsfile):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(weightsfile + 'train_nn.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(weightsfile))
    graph = tf.get_default_graph()
    img_inp = graph.get_tensor_by_name("img_inp:0")
    print(img_inp)
    x = graph.get_tensor_by_name("result:0")
    print(x)
    return sess, img_inp, x


def run_image(model, img_in, img_mask):
    sess, img_inp, x = model
    img_in = img_in * (1 - img_mask[:, :, None]) + 191 * img_mask[:, :, None]
    img_packed = np.dstack([img_in.astype('float32') / 255, img_mask[:, :, None]])
    assert img_packed.shape == (HEIGHT, WIDTH, 4)

    (ret,), = sess.run([x], feed_dict={img_inp: img_packed[None, :, :, :]})
    return ret


if __name__ == '__main__':
    model = loadModel(sys.argv[3])
    img_in = cv2.imread(sys.argv[1])
    img_mask = cv2.imread(sys.argv[2], 0) != 0
    fout = open(sys.argv[1] + '.txt', 'w')
    ret = run_image(model, img_in, img_mask)
    print(ret)
    for x, y, z in ret:
        print(x, y, z, file=fout)

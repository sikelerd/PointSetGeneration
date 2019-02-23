import tensorflow as tf
import time
import argparse
import os
import cv2
from depthestimate.BatchFetcher import BATCH_SIZE, FETCH_BATCH_SIZE, HEIGHT, WIDTH, POINTCLOUDSIZE, OUTPUTPOINTS, BatchFetcher
from depthestimate import network, show3d

lastbatch = None
lastconsumed = FETCH_BATCH_SIZE


def fetch_batch():
    global lastbatch, lastconsumed
    if lastbatch is None or lastconsumed + BATCH_SIZE > FETCH_BATCH_SIZE:
        lastbatch = fetchworker.fetch()
        lastconsumed = 0
    ret = [i[lastconsumed:lastconsumed + BATCH_SIZE] for i in lastbatch]
    lastconsumed += BATCH_SIZE
    return ret


def stop_fetcher():
    fetchworker.shutdown()


def train_network(resourceid, keyname, dumpdir):
    learning_rate = 3e-5 * BATCH_SIZE / FETCH_BATCH_SIZE
    with tf.device('/gpu:%d' % resourceid):
        img_inp, x, pt_gt, loss, optimizer, mindist, loss_nodecay, dists_forward, dists_backward = \
            network.build_graph_training(HEIGHT, WIDTH, POINTCLOUDSIZE, OUTPUTPOINTS, learning_rate)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess, \
            open('%s/%s.log' % (dumpdir, keyname), 'a') as fout:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(dumpdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        trainloss_accs = [0, 0, 0]
        trainloss_acc0 = 1e-9
        validloss_accs = [0, 0, 0]
        validloss_acc0 = 1e-9
        lastsave = time.time()
        epoch = 0
        fetchworker.bno = 0 // (FETCH_BATCH_SIZE / BATCH_SIZE)
        fetchworker.start()
        data, ptcloud, validating = fetch_batch()
        validating = validating[0] != 0
        while epoch < 1:
            epoch += 1
            t0 = time.time()
            t1 = time.time()
            print('image type', img_inp[0][0, 0, 0], type(data[0][0, 0, 0]))
            print('point type', pt_gt[0][0, 0], type(ptcloud[0][0, 0]))
            print('input image shape', data.shape)
            print('input pt shape', ptcloud.shape)
            if not validating:
                _, pred, total_loss, trainloss, trainloss1, trainloss2, summary = sess.run(
                    [optimizer, x, loss, loss_nodecay, dists_forward, dists_backward, merged],
                    feed_dict={img_inp: data, pt_gt: ptcloud})
                trainloss_accs[0] = trainloss_accs[0] * 0.99 + trainloss
                trainloss_accs[1] = trainloss_accs[1] * 0.99 + trainloss1
                trainloss_accs[2] = trainloss_accs[2] * 0.99 + trainloss2
                trainloss_acc0 = trainloss_acc0 * 0.99 + 1
                writer.add_summary(summary, epoch)
            else:
                pred, total_loss, validloss, validloss1, validloss2 = sess.run([x, loss, loss_nodecay, dists_forward, dists_backward],
                                                                                             feed_dict={img_inp: data, pt_gt: ptcloud})
                validloss_accs[0] = validloss_accs[0] * 0.997 + validloss
                validloss_accs[1] = validloss_accs[1] * 0.997 + validloss1
                validloss_accs[2] = validloss_accs[2] * 0.997 + validloss2
                validloss_acc0 = validloss_acc0 * 0.997 + 1
            t2 = time.time()

            if not validating:
                showloss = trainloss
                showloss1 = trainloss1
                showloss2 = trainloss2
            else:
                showloss = validloss
                showloss1 = validloss1
                showloss2 = validloss2
            print(epoch, trainloss_accs[0] / trainloss_acc0, trainloss_accs[1] / trainloss_acc0, trainloss_accs[2] / trainloss_acc0, showloss, showloss1, showloss2,
                  validloss_accs[0] / validloss_acc0, validloss_accs[1] / validloss_acc0, validloss_accs[2] / validloss_acc0, total_loss - showloss, file=fout)
            if epoch % 100 == 0:
                fout.flush()
            if time.time() - lastsave > 900:
                saver.save(sess, '%s/' % dumpdir + keyname + ".ckpt")
                lastsave = time.time()
            print(epoch, 't', trainloss_accs[0] / trainloss_acc0, trainloss_accs[1] / trainloss_acc0, trainloss_accs[2] / trainloss_acc0, 'v', validloss_accs[0] / validloss_acc0,
                  validloss_accs[1] / validloss_acc0, validloss_accs[2] / validloss_acc0, total_loss - showloss, t1 - t0, t2 - t1, time.time() - t0, fetchworker.queue.qsize())
        saver.save(sess, '%s/' % dumpdir + keyname + ".ckpt")
        writer.close()


def load_model(resourceid, weightsfile):
    with tf.device('/gpu:%d' % resourceid):
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
        return sess, img_inp, x


def run_image(model, img_in):
    sess, img_inp, x = model
    img_in = img_in.astype('float32') / 255

    (ret,), = sess.run([x], feed_dict={img_inp: img_in[None, :, :, :]})
    return ret


def train(args):
    if not os.path.exists(args.dump):
        os.mkdir(args.dump)

    keyname = os.path.basename(__file__).rstrip('.py')
    train_network(args.resourceid, keyname, args.dump)


def predict(args):
    model = load_model(args.resourceid, args.dump + '/')
    img_in = cv2.imread(args.image)
    fout = open(args.image + '.txt', 'w')
    ret = run_image(model, img_in)
    for x, y, z in ret:
        print(x, y, z, file=fout)
    show3d.showpoints(ret)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train or evaluate the network')
    subparser = argparser.add_subparsers()
    argparser.add_argument(
        '-data',
        metavar='data',
        default='data',
        help='The source directory of the data (Default data)')
    argparser.add_argument(
        '-dump',
        metavar='dump',
        default='dump',
        help='The directory to dump files. (Default dump)')
    argparser.add_argument(
        '-id', '--resourceid',
        metavar='id',
        default=0,
        help='The GPU on which training takes place. (Default 0)')

    train_parser = subparser.add_parser('train')
    train_parser.set_defaults(func=train)

    predict_parser = subparser.add_parser('predict')
    predict_parser.add_argument(
        'image',
        metavar='img',
        help='the image filt to predict')
    predict_parser.set_defaults(func=predict)

    argsuments = argparser.parse_args()

    fetchworker = BatchFetcher(argsuments.data)
    try:
        argsuments.func(argsuments)
    finally:
        stop_fetcher()

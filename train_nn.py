import tensorflow as tf
import time
import pickle
import argparse
import os
from depthestimate.BatchFetcher import BATCH_SIZE, FETCH_BATCH_SIZE, HEIGHT, WIDTH, POINTCLOUDSIZE, OUTPUTPOINTS, BatchFetcher
from depthestimate import network

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


def train_network(resourceid, keyname, datadir, dumpdir):
    img_inp, x, pt_gt, loss, optimizer, batchno, batchnoinc, mindist, loss_nodecay, dists_forward, dists_backward, dist0 = \
        network.build_graph(resourceid, HEIGHT, WIDTH, POINTCLOUDSIZE, OUTPUTPOINTS, 3e-5 * BATCH_SIZE / FETCH_BATCH_SIZE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess, \
            open('%s/%s.log' % (dumpdir, keyname), 'a') as fout:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("dump", sess.graph)
        sess.run(tf.global_variables_initializer())
        trainloss_accs = [0, 0, 0]
        trainloss_acc0 = 1e-9
        validloss_accs = [0, 0, 0]
        validloss_acc0 = 1e-9
        lastsave = time.time()
        bno = sess.run(batchno)
        fetchworker.bno = bno // (FETCH_BATCH_SIZE / BATCH_SIZE)
        fetchworker.start()
        while bno < 300000:
            t0 = time.time()
            data, ptcloud, validating = fetch_batch()
            t1 = time.time()
            validating = validating[0] != 0
            if not validating:
                _, pred, total_loss, trainloss, trainloss1, trainloss2, distmap_0, summary = sess.run(
                    [optimizer, x, loss, loss_nodecay, dists_forward, dists_backward, dist0, merged],
                    feed_dict={img_inp: data, pt_gt: ptcloud})
                trainloss_accs[0] = trainloss_accs[0] * 0.99 + trainloss
                trainloss_accs[1] = trainloss_accs[1] * 0.99 + trainloss1
                trainloss_accs[2] = trainloss_accs[2] * 0.99 + trainloss2
                trainloss_acc0 = trainloss_acc0 * 0.99 + 1
                writer.add_summary(summary, bno)
            else:
                _, pred, total_loss, validloss, validloss1, validloss2, distmap_0 = sess.run([batchnoinc, x, loss, loss_nodecay, dists_forward, dists_backward, dist0],
                                                                                             feed_dict={img_inp: data, pt_gt: ptcloud})
                validloss_accs[0] = validloss_accs[0] * 0.997 + validloss
                validloss_accs[1] = validloss_accs[1] * 0.997 + validloss1
                validloss_accs[2] = validloss_accs[2] * 0.997 + validloss2
                validloss_acc0 = validloss_acc0 * 0.997 + 1
            t2 = time.time()

            bno = sess.run(batchno)
            if not validating:
                showloss = trainloss
                showloss1 = trainloss1
                showloss2 = trainloss2
            else:
                showloss = validloss
                showloss1 = validloss1
                showloss2 = validloss2
            print(bno, trainloss_accs[0] / trainloss_acc0, trainloss_accs[1] / trainloss_acc0, trainloss_accs[2] / trainloss_acc0, showloss, showloss1, showloss2,
                  validloss_accs[0] / validloss_acc0, validloss_accs[1] / validloss_acc0, validloss_accs[2] / validloss_acc0, total_loss - showloss, file=fout)
            if bno % 128 == 0:
                fout.flush()
            if time.time() - lastsave > 900:
                saver.save(sess, '%s/' % dumpdir + keyname + ".ckpt")
                lastsave = time.time()
            print(bno, 't', trainloss_accs[0] / trainloss_acc0, trainloss_accs[1] / trainloss_acc0, trainloss_accs[2] / trainloss_acc0, 'v', validloss_accs[0] / validloss_acc0,
                  validloss_accs[1] / validloss_acc0, validloss_accs[2] / validloss_acc0, total_loss - showloss, t1 - t0, t2 - t1, time.time() - t0, fetchworker.queue.qsize())
        saver.save(sess, '%s/' % dumpdir + keyname + ".ckpt")
        writer.close()


def dumppredictions(resourceid, keyname, dumpdir, valnum):
    img_inp, x, pt_gt, loss, optimizer, batchno, batchnoinc, mindist, loss_nodecay, dists_forward, dists_backward, dist0 = \
        network.build_graph(resourceid, HEIGHT, WIDTH, POINTCLOUDSIZE, OUTPUTPOINTS, 3e-5 * BATCH_SIZE / FETCH_BATCH_SIZE)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    saver = tf.train.Saver()
    fout = open("%s/%s.v.pkl" % (dumpdir, keyname), 'wb')
    with tf.Session(config=config) as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "%s/%s.ckpt" % (dumpdir, keyname))
        fetchworker.bno = 0
        fetchworker.start()
        cnt = 0
        for i in range(0, 63000):
            t0 = time.time()
            data, ptcloud, validating = fetch_batch()
            validating = validating[0] != 0
            if not validating:
                continue
            cnt += 1
            pred, distmap = sess.run([x, mindist], feed_dict={img_inp: data, pt_gt: ptcloud})
            pickle.dump((i, data, ptcloud, pred, distmap), fout, protocol=-1)

            i, 'time', time.time() - t0, cnt
            if cnt >= valnum:
                break
    fout.close()


def train(args):
    if not os.path.exists(args.dump):
        os.mkdir(args.dump)

    keyname = os.path.basename(__file__).rstrip('.py')
    train_network(args.resourceid, keyname, args.data, args.dump)


def predict(args):
    keyname = os.path.basename(__file__).rstrip('.py')
    dumppredictions(args.resourceid, keyname, args.dump, args.valnum)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocess data')
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
        '-num', '--valnum',
        metavar='valnum',
        default=3,
        help='the index of the file to use vor validation (Default 3)')
    predict_parser.set_defaults(func=predict)

    argsuments = argparser.parse_args()

    fetchworker = BatchFetcher(argsuments.data)
    try:
        argsuments.func(argsuments)
    finally:
        stop_fetcher()

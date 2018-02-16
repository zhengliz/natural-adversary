import os, sys
import pickle
import numpy as np
import tensorflow as tf
import argparse
from keras.models import load_model

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import tflib.mnist
from mnist_wgan_inv import MnistWganInv
from search import iterative_search, recursive_search


def save_adversary(adversary, filename):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    ax[0].imshow(np.reshape(adversary['x'], (28, 28)),
                 interpolation='none', cmap=plt.get_cmap('gray'))
    ax[0].text(1, 5, str(adversary['y']), color='white', fontsize=50)
    ax[0].axis('off')

    ax[1].imshow(np.reshape(adversary['x_adv'], (28, 28)),
                 interpolation='none', cmap=plt.get_cmap('gray'))
    ax[1].text(1, 5, str(adversary['y_adv']), color='white', fontsize=50)
    ax[1].axis('off')

    fig.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_path', type=str, default='./models/model-47999',
                        help='mnist GAN path')
    parser.add_argument('--rf_path', type=str, default='./models/mnist_rf_9045.sav',
                        help='RF classifier path')
    parser.add_argument('--lenet_path', type=str, default='./models/mnist_lenet_9871.h5',
                        help='LeNet classifier path')
    parser.add_argument('--classifier', type=str, default='rf',
                        help='classifier: rf OR lenet')
    parser.add_argument('--iterative', action='store_true',
                        help='iterative search OR recursive')
    parser.add_argument('--nsamples', type=int, default=5000,
                        help='number of samples in each search iteration')
    parser.add_argument('--step', type=float, default=0.01,
                        help='Delta r for search step size')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_path', type=str, default='./examples/',
                        help='output path')
    args = parser.parse_args()

    if args.classifier == 'rf':
        classifier = pickle.load(open(args.rf_path, 'rb'))

        def cla_fn(x):
            return classifier.predict(np.reshape(x, (-1, 784)))

    elif args.classifier == 'lenet':
        graph_CLA = tf.Graph()
        with graph_CLA.as_default():
            classifier = load_model(args.lenet_path)

        def cla_fn(x):
            with graph_CLA.as_default():
                return np.argmax(classifier.predict_on_batch(np.reshape(x, (-1, 1, 28, 28))), axis=1)

    else:
        sys.exit('Please choose MNIST classifier: rf OR lenet')

    graph_GAN = tf.Graph()
    with graph_GAN.as_default():
        sess_GAN = tf.Session()
        model_GAN = MnistWganInv()
        saver_GAN = tf.train.Saver(max_to_keep=100)
        saver_GAN = tf.train.import_meta_graph('{}.meta'.format(args.gan_path))
        saver_GAN.restore(sess_GAN, args.gan_path)


    def gen_fn(z):
        with sess_GAN.as_default():
            with graph_GAN.as_default():
                x_p = sess_GAN.run(model_GAN.generate(tf.cast(tf.constant(np.asarray(z)), 'float32')))
        return x_p


    def inv_fn(x):
        with sess_GAN.as_default():
            with graph_GAN.as_default():
                z_p = sess_GAN.run(model_GAN.invert(x))
        return z_p


    if args.iterative:
        search = iterative_search
    else:
        search = recursive_search

    _, _, test_data = tflib.mnist.load_data()

    for i in range(10):
        x = test_data[0][i]
        y = test_data[1][i]
        y_pred = cla_fn(x)[0]
        if y_pred != y:
            continue

        adversary = search(gen_fn, inv_fn, cla_fn, x, y,
                           nsamples=args.nsamples, step=args.step, verbose=args.verbose)
        if args.iterative:
            filename = 'mnist_{}_iterative_{}.png'.format(str(i).zfill(4), args.classifier)
        else:
            filename = 'mnist_{}_recursive_{}.png'.format(str(i).zfill(4), args.classifier)

        save_adversary(adversary, os.path.join(args.output_path, filename))


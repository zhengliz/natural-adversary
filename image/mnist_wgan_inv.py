import os, sys, time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import numpy as np
import tensorflow as tf
import argparse

import tflib
import tflib.mnist
import tflib.plot
import tflib.save_images
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear


class MnistWganInv(object):
    def __init__(self, x_dim=784, z_dim=64, latent_dim=64, batch_size=80,
                 c_gp_x=10., lamda=0.1, output_path='./'):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.c_gp_x = c_gp_x
        self.lamda = lamda
        self.output_path = output_path

        self.gen_params = self.dis_params = self.inv_params = None

        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.x_p = self.generate(self.z)

        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        self.z_p = self.invert(self.x)

        self.dis_x = self.discriminate(self.x)
        self.dis_x_p = self.discriminate(self.x_p)
        self.rec_x = self.generate(self.z_p)
        self.rec_z = self.invert(self.x_p)

        self.gen_cost = -tf.reduce_mean(self.dis_x_p)

        self.inv_cost = tf.reduce_mean(tf.square(self.x - self.rec_x))
        self.inv_cost += self.lamda * tf.reduce_mean(tf.square(self.z - self.rec_z))

        self.dis_cost = tf.reduce_mean(self.dis_x_p) - tf.reduce_mean(self.dis_x)

        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
        difference = self.x_p - self.x
        interpolate = self.x + alpha * difference
        gradient = tf.gradients(self.discriminate(interpolate), [interpolate])[0]
        slope = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
        gradient_penalty = tf.reduce_mean((slope - 1.) ** 2)
        self.dis_cost += self.c_gp_x * gradient_penalty

        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.gen_cost, var_list=self.gen_params)
        self.inv_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.inv_cost, var_list=self.inv_params)
        self.dis_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
            self.dis_cost, var_list=self.dis_params)

    def generate(self, z):
        assert z.shape[1] == self.z_dim

        output = tflib.ops.linear.Linear('Generator.Input', self.z_dim,
                                         self.latent_dim * 64, z)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, self.latent_dim * 4, 4, 4])  # 4 x 4

        output = tflib.ops.deconv2d.Deconv2D('Generator.2', self.latent_dim * 4,
                                             self.latent_dim * 2, 5, output)
        output = tf.nn.relu(output)  # 8 x 8
        output = output[:, :, :7, :7]  # 7 x 7

        output = tflib.ops.deconv2d.Deconv2D('Generator.3', self.latent_dim * 2,
                                             self.latent_dim, 5, output)
        output = tf.nn.relu(output)  # 14 x 14

        output = tflib.ops.deconv2d.Deconv2D('Generator.Output',
                                             self.latent_dim, 1, 5, output)
        output = tf.nn.sigmoid(output)  # 28 x 28

        if self.gen_params is None:
            self.gen_params = tflib.params_with_name('Generator')

        return tf.reshape(output, [-1, self.x_dim])

    def discriminate(self, x):
        output = tf.reshape(x, [-1, 1, 28, 28])  # 28 x 28

        output = tflib.ops.conv2d.Conv2D(
            'Discriminator.Input', 1, self.latent_dim, 5, output, stride=2)
        output = tf.nn.leaky_relu(output)  # 14 x 14

        output = tflib.ops.conv2d.Conv2D(
            'Discriminator.2', self.latent_dim, self.latent_dim * 2, 5,
            output, stride=2)
        output = tf.nn.leaky_relu(output)  # 7 x 7

        output = tflib.ops.conv2d.Conv2D(
            'Discriminator.3', self.latent_dim * 2, self.latent_dim * 4, 5,
            output, stride=2)
        output = tf.nn.leaky_relu(output)  # 4 x 4
        output = tf.reshape(output, [-1, self.latent_dim * 64])

        output = tflib.ops.linear.Linear(
            'Discriminator.Output', self.latent_dim * 64, 1, output)
        output = tf.reshape(output, [-1])

        if self.dis_params is None:
            self.dis_params = tflib.params_with_name('Discriminator')

        return output

    def invert(self, x):
        output = tf.reshape(x, [-1, 1, 28, 28])  # 28 x 28

        output = tflib.ops.conv2d.Conv2D(
            'Inverter.Input', 1, self.latent_dim, 5, output, stride=2)
        output = tf.nn.leaky_relu(output)  # 14 x 14

        output = tflib.ops.conv2d.Conv2D(
            'Inverter.2', self.latent_dim, self.latent_dim * 2, 5, output,
            stride=2)
        output = tf.nn.leaky_relu(output)  # 7 x 7

        output = tflib.ops.conv2d.Conv2D(
            'Inverter.3', self.latent_dim * 2, self.latent_dim * 4, 5,
            output, stride=2)
        output = tf.nn.leaky_relu(output)  # 4 x 4
        output = tf.reshape(output, [-1, self.latent_dim * 64])

        output = tflib.ops.linear.Linear(
            'Inverter.4', self.latent_dim * 64, self.latent_dim * 8, output)
        output = tf.nn.leaky_relu(output)

        output = tflib.ops.linear.Linear(
            'Inverter.Output', self.latent_dim * 8, self.z_dim, output)
        output = tf.reshape(output, [-1, self.z_dim])

        if self.inv_params is None:
            self.inv_params = tflib.params_with_name('Inverter')

        return output

    def train_gen(self, sess, x, z):
        _gen_cost, _ = sess.run([self.gen_cost, self.gen_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _gen_cost

    def train_dis(self, sess, x, z):
        _dis_cost, _ = sess.run([self.dis_cost, self.dis_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _dis_cost

    def train_inv(self, sess, x, z):
        _inv_cost, _ = sess.run([self.inv_cost, self.inv_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _inv_cost

    def generate_from_noise(self, sess, noise, frame):
        samples = sess.run(self.x_p, feed_dict={self.z: noise})
        tflib.save_images.save_images(
            samples.reshape((-1, 28, 28)),
            os.path.join(self.output_path, 'examples/samples_{}.png'.format(frame)))
        return samples

    def reconstruct_images(self, sess, images, frame):
        reconstructions = sess.run(self.rec_x, feed_dict={self.x: images})
        comparison = np.zeros((images.shape[0] * 2, images.shape[1]),
                              dtype=np.float32)
        for i in xrange(images.shape[0]):
            comparison[2 * i] = images[i]
            comparison[2 * i + 1] = reconstructions[i]
        tflib.save_images.save_images(
            comparison.reshape((-1, 28, 28)),
            os.path.join(self.output_path, 'examples/recs_{}.png'.format(frame)))
        return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--z_dim', type=int, default=64, help='dimension of z')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='latent dimension')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='training steps')
    parser.add_argument('--dis_iter', type=int, default=5,
                        help='discriminator steps')
    parser.add_argument('--c_gp_x', type=float, default=10.,
                        help='coefficient for gradient penalty x')
    parser.add_argument('--lamda', type=float, default=.1,
                        help='coefficient for divergence of z')
    parser.add_argument('--output_path', type=str, default='./',
                        help='output path')
    args = parser.parse_args()


    # dataset iterator
    train_gen, dev_gen, test_gen = tflib.mnist.load(args.batch_size, args.batch_size)

    def inf_train_gen():
        while True:
            for instances, labels in train_gen():
                yield instances

    _, _, test_data = tflib.mnist.load_data()
    fixed_images = test_data[0][:32]
    del test_data

    tf.set_random_seed(326)
    np.random.seed(326)
    fixed_noise = np.random.randn(64, args.z_dim)

    mnistWganInv = MnistWganInv(
        x_dim=784, z_dim=args.z_dim, latent_dim=args.latent_dim,
        batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path)

    saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        images = noise = gen_cost = dis_cost = inv_cost = None
        dis_cost_lst, inv_cost_lst = [], []
        for iteration in range(args.iterations):
            for i in range(args.dis_iter):
                noise = np.random.randn(args.batch_size, args.z_dim)
                images = inf_train_gen().next()

                dis_cost_lst += [mnistWganInv.train_dis(session, images, noise)]
                inv_cost_lst += [mnistWganInv.train_inv(session, images, noise)]

            gen_cost = mnistWganInv.train_gen(session, images, noise)
            dis_cost = np.mean(dis_cost_lst)
            inv_cost = np.mean(inv_cost_lst)

            tflib.plot.plot('train gen cost', gen_cost)
            tflib.plot.plot('train dis cost', dis_cost)
            tflib.plot.plot('train inv cost', inv_cost)

            if iteration % 100 == 99:
                mnistWganInv.generate_from_noise(session, fixed_noise, iteration)
                mnistWganInv.reconstruct_images(session, fixed_images, iteration)

            if iteration % 1000 == 999:
                save_path = saver.save(session, os.path.join(
                    args.output_path, 'models/model'), global_step=iteration)

            if iteration % 1000 == 999:
                dev_dis_cost_lst, dev_inv_cost_lst = [], []
                for dev_images, _ in dev_gen():
                    noise = np.random.randn(args.batch_size, args.z_dim)
                    dev_dis_cost, dev_inv_cost = session.run(
                        [mnistWganInv.dis_cost, mnistWganInv.inv_cost],
                        feed_dict={mnistWganInv.x: dev_images,
                                   mnistWganInv.z: noise})
                    dev_dis_cost_lst += [dev_dis_cost]
                    dev_inv_cost_lst += [dev_inv_cost]
                tflib.plot.plot('dev dis cost', np.mean(dev_dis_cost_lst))
                tflib.plot.plot('dev inv cost', np.mean(dev_inv_cost_lst))

            if iteration < 5 or iteration % 100 == 99:
                tflib.plot.flush(os.path.join(args.output_path, 'models'))

            tflib.plot.tick()



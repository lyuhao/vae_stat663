import tensorflow as tf
import numpy as np
import vaelib.freyface_dataset as freyface_dataset
import os
import vaelib.vae as vae
import vaelib.plot_utils as plot_utils
import glob

import argparse

from builtins import FileExistsError

IMAGE_SIZE_rows = 28

IMAGE_SIZE_cols = 20


def parse_args():
    desc = "VAE'"
    parser = argparse.ArgumentParser(description=desc)

    ## network hypermeters 
    parser.add_argument('--dim_z', type=int, default=10, help='Dimension of latent vector')
    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    ## train arguments
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    ## result arguments
    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    return parser.parse_args()

"""main function"""
def main(args):

    """ parameters """
    RESULTS_DIR = args.results_path

    n_hidden = args.n_hidden
    dim_img = IMAGE_SIZE_rows*IMAGE_SIZE_cols # number of pixels for a MNIST image
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plotting
    PRR_n_img_x = 4              # number of images along x-axis in a canvas
    PRR_n_img_y = 4              # number of images along y-axis in a canvas
    PRR_resize_factor = 1.0  # resize factor for each image in a canvas

    """ prepare MNIST data """

    train_size, test_size,train_data,test_data = freyface_dataset.load_frey_face_dataset()
    n_samples = train_size

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
    labels = tf.placeholder(tf.float32, shape=[None, 10], name='target_label')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ training """

    # Plot for reproduce performance

    PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_cols, IMAGE_SIZE_rows, PRR_resize_factor)
    x_PRR = test_data[0:PRR.n_tot_imgs, :]
    x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_cols, IMAGE_SIZE_rows)
    PRR.save_images(x_PRR_img, name='input.jpg')


    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_data)
            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data[offset:(offset + batch_size), :]
                batch_xs_target = batch_xs_input

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob : 0.9})

            print("epoch %d: L_tot %03.2f" %(epoch,tot_loss))
            # if minimum loss is updated or final epoch, plot results
            if min_tot_loss > tot_loss or epoch+1 == n_epochs:
                min_tot_loss = tot_loss
                # Plot for reproduce performance
                    #print('reach here !!!')
                y_PRR = sess.run(y, feed_dict={x_hat: x_PRR, keep_prob : 1})
                y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_cols, IMAGE_SIZE_rows)
                PRR.save_images(y_PRR_img, name="/PRR_epoch_%02d" %(epoch) + ".jpg")
if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)

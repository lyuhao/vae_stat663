import tensorflow as tf
import numpy as np
import os
import vaelib.vae as vae



# input placeholders
# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, 128], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, 128], name='target_img')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# network architecture
n_output = 2 
n_hidden = 512
mean,mu = vae.gaussian_MLP_encoder(x_hat, n_hidden, n_output, keep_prob)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

    x = np.random.rand(1,128)
    z_mean,z_mu = sess.run(
        (mean, mu),
        feed_dict={x_hat: x, keep_prob : 0.9})

print(z_mean.shape)
print(z_mu.shape)
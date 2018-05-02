import tensorflow as tf
import numpy as np
import os
import vaelib.vae as vae



# input placeholders
# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
z_hat = tf.placeholder(tf.float32, shape=[None, 4], name='input_img')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# network architecture
n_output = 128 
n_hidden = 512
Y = vae.bernoulli_MLP_decoder(z_hat, n_hidden, n_output, keep_prob, reuse=False)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

    z = np.random.rand(1,4)
    y = sess.run(
        (Y),
        feed_dict={z_hat: z, keep_prob : 0.9})

print(y.shape)
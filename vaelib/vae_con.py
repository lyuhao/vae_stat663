import tensorflow as tf

# Gaussian MLP as encoder
def encoder(x, nb_filters,n_hidden, n_output, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers

        input_layer = tf.reshape(x,[-1,28,28,1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=1,
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=nb_filters,
            strides=(2,2),
            kernel_size=[2, 2],
            padding="same",
            activation=tf.nn.relu)
        
        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=nb_filters,
            strides=(1,1),
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=nb_filters,
            strides=(1,1),
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

        flatten_layer = tf.contrib.layers.flatten(conv4)
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [flatten_layer.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(flatten_layer, w0) + b0
        h0 = tf.nn.relu(h0)
       # h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
       # w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
       # b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        #h1 = tf.matmul(h0, w1) + b1
        #h1 = tf.nn.tanh(h1)
       # h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output * 2], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
        gaussian_params = tf.matmul(h0, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

    return mean, stddev

# Bernoulli MLP as decoder
def decoder(z, nb_filters, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.relu(h0)
       # h0 = tf.nn.dropout(h0, keep_prob)

        w1 = tf.get_variable('w1', [h0.get_shape()[1], nb_filters*14*14], initializer=w_init)
        b1 = tf.get_variable('b1',[nb_filters*14*14],initializer=b_init)

        hidden_values = tf.matmul(h0,w1) + b1

        hidden_values = tf.reshape(hidden_values,[-1,14,14,nb_filters]) 


        conv1 = tf.layers.conv2d_transpose(
            hidden_values,
            filters = nb_filters,
            kernel_size = [3,3],
            strides=(1,1),
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu
            )

        conv2 = tf.layers.conv2d_transpose(
            conv1,
            filters = nb_filters,
            kernel_size = [3,3],
            strides=(1,1),
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu
            )

        conv3 = tf.layers.conv2d_transpose(
            conv2,
            filters = nb_filters,
            kernel_size = [2,2],
            strides=(2,2),
            padding='valid',
            data_format='channels_last',
            activation=tf.nn.relu
            )

        y = tf.layers.conv2d_transpose(
            conv3,
            filters = 1,
            kernel_size = [2,2],
            strides=(1,1),
            padding='same',
            data_format='channels_last',
            activation=tf.nn.sigmoid
            )

    y = tf.reshape(y,[-1,784])
    return y


def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):


    mu, sigma = encoder(x_hat, 64, n_hidden, dim_z, keep_prob)


    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    y = bernoulli_MLP_decoder(z, 64,n_hidden, dim_img, keep_prob)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence


    return y, z, -ELBO, -marginal_likelihood, KL_divergence

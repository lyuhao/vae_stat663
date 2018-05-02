Variational Auto-Encoder for MNIST and frey-face


To install the package:

python3 setup.py install


The implentation of three vaes are in the vaelib/ folder:
1. vae.py: mlp implementaion of vae
2. cvae.py: mlp implementaion of conditional vae
3. vae_con.py: convolutional neural net implementation of vae


four examples are provided in the examples/ folder:
1. vae_tensorflow.py : mlp implementation of vae with mnist dataset
2. cvae_tensorflow.py: mlp implementation of conditional vae with mnist dataset
3. vae_conv_tensorflow.py: convolutional neural net implementaion of vae with mnist dataset
4. vae_freyface_tensorflow.py: mlp implementation of vae with freyface dataset

Two tests are provided in the tests/ folder:
1. test_encoder.py: test whethter the encoder produce the output with the correct shape
2. test_decoder.py: test whether the decoder produce the output with the correct shape
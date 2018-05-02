import gzip
import os
from urllib import request 
import numpy as np
from scipy import io as sio
from scipy.misc import imsave
from scipy.misc import imresize

url =  "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
data_filename = os.path.basename(url)



img_rows=28
img_cols=20
#ff = scipy.io.loadmat('./frey/frey_rawface.mat')
#ff = ff.astype('float32')/255.

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print('Downloading {}'.format(filename))
    request.urlretrieve(source + filename, filename)

def load_frey_face_images(filename):
    if not os.path.exists(filename):
        download(filename, source='http://www.cs.nyu.edu/~roweis/data/')
    #import scipy.io as sio
    data = sio.loadmat(filename)['ff'].T.reshape((-1, img_rows, img_cols))
    return data / np.float32(255)


def load_frey_face_dataset():
    X = load_frey_face_images('frey_rawface.mat')
    X_train, X_val = X[:-565], X[-565:]
    #X_train = X_train.reshape(-1,28*20)
    X_train = X_train.reshape((len(X_train), 28*20))
    X_val = X_val.reshape((len(X_val), 28*20))
    return X.shape[0]-565,565,X_train, X_val



def save_examples(data, n=None, n_cols=20, thumbnail_cb=None,DIR='results',name='input.jpg'):
    if n is None:
        n = len(data)    
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    for k, x in enumerate(data[:n]):
        r = k // n_cols
        c = k % n_cols
        figure[r * img_rows: (r + 1) * img_rows,
               c * img_cols: (c + 1) * img_cols] = x
        if thumbnail_cb is not None:
            thumbnail_cb(locals())
     
    imsave(DIR + "/"+name, figure)   
    #plt.figure(figsize=(12, 10))
    #plt.imshow(figure)
    #plt.axis("off")
    #plt.tight_layout()





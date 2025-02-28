import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros, arange
from pdb import set_trace

def loadMnist(dataset="training", digits=arange(10), path="./mnist/"):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://g.sweyla.com/blog/2012/mnist-numpy/
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    images_float = zeros((N, rows, cols), dtype=float)
    for i in range(len(ind)):
        # normalize to [0,1] and rotate for pyplot.contourf
        images_float[i] = images[i][::-1,:] / 255.

    # To plot images,
    # pyplot.contourf(images[16][::-1,:]/255., cmap='Greys')

    return images_float, labels

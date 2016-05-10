import numpy as np
import sys
#import os
import scipy.io as sio
from PIL import Image

# find caffe root and import caffe
caffe_root = '/usr/local/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb

import h5py # to read mat files

# get numpy version
#print np.__version__


# load training dataset, mat files
path2matfiles='.././matfiles/'
try:
    f = h5py.File(path2matfiles+'traindata_lmdb.mat')
    print f.keys()
    print ('mat file is loaded using h5py')
except:
    mat_contents = sio.loadmat(path2matfiles+'traindata_lmdb.mat')
    f = sio.loadmat(path2matfiles+'testdata_lmdb.mat')
    print ('mat file is loaded using loadmat')

# extract images
X=f['trainData']
print ('training data size: ',X.shape)

# number of images
N=X.shape[0]

# labels
y = f['trainLabels']
print ("label data size: ", y.shape)

#yr=np.reshape(y,[y.shape[1],1])
#print ("label data size: ", yr.shape)

# We need to prepare the database for the size. We'll set it 10 times
h=X.shape[2]
w=X.shape[3]
map_size = 10*N*h*w*64;


# open lmdb file to write
try:
    env = lmdb.open('../caffe/train_lmdb/', map_size=map_size)
    print("lmdb file opened")
except:
   print ("cannot open lmdb file")

# write into lmdb file
with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tostring()
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

print ("conversion to lmdb completed for train data")

# check
try:
    env = lmdb.open('../caffe/train_lmdb/', readonly=True)
    print('lmdb file opend readonly')
except:
    print('cannot open lmdb file')

with env.begin() as txn:raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label

print ('check data size: ', x.shape)
print ('check label size', y)




# load test dataset, mat files
try:
    f = h5py.File(path2matfiles+'testdata_lmdb.mat')
    print f.keys()
    print ('mat file is loaded using h5py')
except:
    mat_contents = sio.loadmat(path2matfiles+'testdata_lmdb.mat')
    f = sio.loadmat(path2matfiles+'testdata_lmdb.mat')
    print ('mat file is loaded using loadmat')


# extract images
X=f['testData']
print ('test data size: ',X.shape)

# number of test images
Ntest=X.shape[0]

# labels
y = f['testLabels']
print ('label data size: ',y.shape)

#yr=np.reshape(y,[y.shape[1],1])
#print ("label data size: ", yr.shape)

# We need to prepare the database for the size. We'll set it 10 times
h=X.shape[2]
w=X.shape[3]
map_size = 10*N*h*w*64;

# open lmdb file to write
try:
    env = lmdb.open('../caffe/test_lmdb/', map_size=map_size)
    print("lmdb file opened")
except:
    print('cannot open lmdb file')


with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(Ntest):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tostring()
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())


print ("conversion to lmdb completed for test data")



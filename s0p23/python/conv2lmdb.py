import numpy as np
import sys
import os,shutil
import scipy.io as sio
from PIL import Image
import matplotlib.pylab as plt

# find caffe root and import caffe
caffe_root = '/usr/local/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb

import h5py # to read mat files

# get numpy version
#print np.__version__


# load training dataset, mat files
fold=5
#path2matfiles='.././matfiles/'
path2matfilesTrain='../matfiles/traindata_lmdb'+str(fold)+'.mat'
path2matfilesTest='../matfiles/testdata_lmdb'+str(fold)+'.mat'

lmdbpathTrain='../caffe/data/fold'+str(fold)+'/train_lmdb/'
lmdbpathTest='../caffe/data/fold'+str(fold)+'/test_lmdb/'

try:
    f = h5py.File(path2matfilesTrain)
    print f.keys()
    print ('mat file is loaded using h5py')
except:
    f = sio.loadmat(path2matfilesTrain)
    print ('mat file is loaded using loadmat')

# extract images
X=f['trainData']
print ('training data size: ',X.shape)

# number of images
N=X.shape[0]

# labels
y = f['trainLabels']
print ("label data size: ", y.shape)

# display sample image
n1=np.random.randint(len(X))
I1=np.squeeze(X[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
plt.title(y[n1])

# RGB to BGR
X = np.array(X,dtype='uint8') # convert to numpy
X=X[:, ::-1, :, :] # RGB to BGR
#n1=np.random.randint(len(X))
I1=np.squeeze(X[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
plt.title(y[n1])
print 'converted RGB to BGR for caffe'

# reshape from N C W H to N C H W
X=np.transpose(X,(0,1,3,2))
I1=np.squeeze(X[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
plt.title(y[n1])
print 'converted to N C H W'


# We need to prepare the database for the size. We'll set it 10 times
h=X.shape[2]
w=X.shape[3]
map_size = 10*N*h*w*64;


# open lmdb file to write
if  os.path.exists(lmdbpathTrain):
    shutil.rmtree(lmdbpathTrain)
    print 'lmdb deleted ...'
os.makedirs(lmdbpathTrain)
print 'mdb path for training data created'


try:
    env = lmdb.open(lmdbpathTrain, map_size=map_size)
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
    env = lmdb.open(lmdbpathTrain, readonly=True)
    print('lmdb file opend readonly for checking ....')
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

# disp;ay
I2=np.transpose(x,(2,1,0))
plt.imshow(I1)
plt.title(y)

########################################################################################
#######################################################################################
##########################################################################################


try:
    f = h5py.File(path2matfilesTest)
    print f.keys()
    print ('mat file is loaded using h5py')
except:
    f = sio.loadmat(path2matfilesTest)
    print ('mat file is loaded using loadmat')


# extract images
X=f['testData']
print ('test data size: ',X.shape)

# number of test images
Ntest=X.shape[0]

# labels
y = f['testLabels']
print ('label data size: ',y.shape)


# display sample image
n1=np.random.randint(len(X))
I1=np.squeeze(X[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
plt.title(y[n1])

# RGB to BGR
X = np.array(X,dtype='uint8') # convert to numpy
X=X[:, ::-1, :, :] # RGB to BGR
#n1=np.random.randint(len(X))
I1=np.squeeze(X[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
plt.title(y[n1])
print 'converted RGB to BGR for caffe'

# reshape from N C W H to N C H W
X=np.transpose(X,(0,1,3,2))
I1=np.squeeze(X[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
plt.title(y[n1])
print 'converted to N C H W'


# We need to prepare the database for the size. We'll set it 10 times
h=X.shape[2]
w=X.shape[3]
map_size = 10*N*h*w*64;


# open lmdb file to write
if  os.path.exists(lmdbpathTest):
    shutil.rmtree(lmdbpathTest)
    print 'lmdb deleted ...'
os.makedirs(lmdbpathTest)
print 'mdb path for test data created'


# open lmdb file to write
try:
    env = lmdb.open(lmdbpathTest, map_size=map_size)
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


print ("conversion to lmdb completed for test data   \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")






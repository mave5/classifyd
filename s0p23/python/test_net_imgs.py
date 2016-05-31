import numpy as np
import sys
import os
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import datetime


# find caffe root and import caffe
caffe_root = '/usr/local/caffeapril2016/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import h5py # to read mat files

fold=4
model = '../caffe/net_deploy.prototxt'
weights = '../caffe/trainedmodels/net_fold'+str(fold)+'.caffemodel'

if not os.path.isfile(weights):
    print("Cannot find caffemodel...")

# set gpu mode and id
gpu_id=2
caffe.set_device(gpu_id)
caffe.set_mode_gpu()
net = caffe.Net(model,weights,caffe.TEST)
print 'net was created ...'


# load test dataset, mat files
img_dir=('/data/dlnets/misc/state/imgs/test/')
from os import listdir
from os.path import isfile, join
imglist = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
N=len(imglist)
print ('number of images: ', N)

meanRGB = np.array([80.25, 97.05, 95.4])  # mean RGB
meanBGR = np.array([95.4, 97.05, 80.25])
Xcrop = np.zeros([5, 3, 224, 224], dtype=np.float32)  # initialize

# function to preprocess and extract five crops
def preparedata(im):
    im = im.resize((320, 240), Image.ANTIALIAS) # resize
    testData = np.array(im, dtype=np.float32) # convert to numpy
    # testData = testData[:, :, ::-1] # RGB to BGR
    testData -= meanRGB # subtract mean
    testData = testData.transpose((2, 1, 0))  # permute to C W H

    # five crops
    Xcrop[0, :, :, :] = testData[:, 0:224, 0:224]
    Xcrop[1, :, :, :] = testData[:, 96:320, 0:224]
    Xcrop[2, :, :, :] = testData[:, 48:272, 8:232]
    Xcrop[3, :, :, :] = testData[:, 0:224, 16:240]
    Xcrop[4, :, :, :] = testData[:, 96:320, 16:240]
    return Xcrop


# loop over all images
net_outMean=np.zeros([N,10])
for k1 in range(N):
    print ('processing ', k1)
    im=Image.open(img_dir+imglist[k1])

    # downsample,subtract,crop data
    Xcrop=preparedata(im)

    pred=np.zeros([5,10]) # init predictions
    #for k2 in range(5):
    net.blobs['data'].data[...] =Xcrop#[k2,:,:,:]
        # perfrom forward operation
    out = net.forward()
        #pred[k2,:]=out['prob']
    pred = out['prob']

    # take average
    net_outMean[k1,:]=np.mean(pred,axis=0)


# create submsssion
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


create_submission(net_outMean,imglist,'nfolds')
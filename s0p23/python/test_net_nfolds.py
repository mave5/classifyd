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

fold=1
model = '../caffe/net_deploy.prototxt'
weights1 = '../caffe/trainedmodels/net_fold'+str(1)+'.caffemodel'
weights2 = '../caffe/trainedmodels/net_fold'+str(2)+'.caffemodel'
weights3 = '../caffe/trainedmodels/net_fold'+str(3)+'.caffemodel'
weights4 = '../caffe/trainedmodels/net_fold'+str(4)+'.caffemodel'
weights5 = '../caffe/trainedmodels/net_fold'+str(5)+'.caffemodel'

if not os.path.isfile(weights1):
    print("Cannot find caffemodel 1...")
elif not os.path.isfile(weights2):
    print("Cannot find caffemodel 2...")
elif not os.path.isfile(weights3):
    print("Cannot find caffemodel 3...")
elif not os.path.isfile(weights4):
    print("Cannot find caffemodel 4...")
elif not os.path.isfile(weights5):
    print("Cannot find caffemodel 5...")

# set gpu mode and id
gpu_id=2
caffe.set_device(gpu_id)
caffe.set_mode_gpu()
net1 = caffe.Net(model,weights1,caffe.TEST)
net2 = caffe.Net(model,weights2,caffe.TEST)
net3 = caffe.Net(model,weights3,caffe.TEST)
net4 = caffe.Net(model,weights4,caffe.TEST)
net5 = caffe.Net(model,weights5,caffe.TEST)

print 'all nets were created ...'


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
net1_outMean=np.zeros([N,10])
net2_outMean=np.zeros([N,10])
net3_outMean=np.zeros([N,10])
net4_outMean=np.zeros([N,10])
net5_outMean=np.zeros([N,10])

net1_outCnt=np.zeros([N,10])
net2_outCnt=np.zeros([N,10])
net3_outCnt=np.zeros([N,10])
net4_outCnt=np.zeros([N,10])
net5_outCnt=np.zeros([N,10])

for k1 in range(N):
    print ('processing ', k1)
    im=Image.open(img_dir+imglist[k1])

    # downsample,subtract,crop data
    Xcrop=preparedata(im)

    pred=np.zeros([5,10]) # init predictions

    net1.blobs['data'].data[...] =Xcrop
    net2.blobs['data'].data[...] = Xcrop
    net3.blobs['data'].data[...] = Xcrop
    net4.blobs['data'].data[...] = Xcrop
    net5.blobs['data'].data[...] = Xcrop

    out1 = net1.forward()
    out2= net2.forward()
    out3 = net3.forward()
    out4 = net4.forward()
    out5 = net5.forward()

    pred1 = out1['prob']
    pred2 = out2['prob']
    pred3 = out3['prob']
    pred4 = out4['prob']
    pred5 = out5['prob']

    # take average
    net1_outMean[k1,:]=np.mean(pred1,axis=0)
    net2_outMean[k1,:]=np.mean(pred2,axis=0)
    net3_outMean[k1,:]=np.mean(pred3,axis=0)
    net4_outMean[k1,:]=np.mean(pred4,axis=0)
    net5_outMean[k1,:]=np.mean(pred5,axis=0)

    # output from the center
    net1_outCnt[k1,:]=pred1[2,:]
    net2_outCnt[k1,:]=pred2[2,:]
    net3_outCnt[k1,:]=pred3[2,:]
    net4_outCnt[k1,:]=pred4[2,:]
    net5_outCnt[k1,:]=pred5[2,:]



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


# create submission for ensamble of n folds
nets_outMean=(net1_outMean+net2_outMean+net3_outMean+net4_outMean+net5_outMean)/5
nets_outCnt=(net1_outCnt+net2_outCnt+net3_outCnt+net4_outCnt+net5_outCnt)/5
create_submission(nets_outMean,imglist,'Meannfolds')
create_submission(nets_outCnt,imglist,'Cntnfolds')

# create submssion for each fold
create_submission(net1_outMean,imglist,'meanfold1')
create_submission(net2_outMean,imglist,'meanfold2')
create_submission(net3_outMean,imglist,'meanfold3')
create_submission(net4_outMean,imglist,'meanfold4')
create_submission(net5_outMean,imglist,'meanfold5')

# create submssion for each fold
create_submission(net1_outCnt,imglist,'Cntfold1')
create_submission(net2_outCnt,imglist,'Cntfold2')
create_submission(net3_outCnt,imglist,'Cntfold3')
create_submission(net4_outCnt,imglist,'Cntfold4')
create_submission(net5_outCnt,imglist,'Cntfold5')

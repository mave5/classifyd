import numpy as np
import sys
import os
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import pandas as pd


#import sys
#clear = lambda: os.system('cls')
#clear()

# find caffe root and import caffe
caffe_root = '/usr/local/caffeapril2016/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import h5py # to read mat files

fold=5
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
path2matfiles='../matfiles/testdata_lmdb'+str(fold)+'.mat'
try:
    f = h5py.File(path2matfiles)
    print f.keys()
    print ('mat file is loaded using h5py')
except:
    f = sio.loadmat(path2matfiles)
    print ('mat file is loaded using loadmat')

# extract images
testData=f['testData']
Imgs=testData
testData = np.array(testData, dtype=np.float32)
print ('test data size: ',testData.shape)
print testData[0,0,:,:]

# number of images
N=testData.shape[0]

# labels
labels = f['testLabels']
print ('label data size: ',labels.shape)


Xcrop=np.zeros([5,3,224,224],dtype=np.float32)
net_outMean=np.zeros([N,10])
net_outMax=np.zeros([N,10])
net_outCnt=np.zeros([N,10])

class_d=np.zeros([N,1],dtype='uint8')

meanRGB=np.array([80.25,97.05,95.4])
#meanRGB=np.array([79.4991,96.3268,94.4022]) # fold 4
#meanRGB=np.array([81.4706,98.4924,97.3605])

testData[:,0,:,:]=testData[:,0,:,:]-meanRGB[0]
testData[:,1,:,:]=testData[:,1,:,:]-meanRGB[1]
testData[:,2,:,:]=testData[:,2,:,:]-meanRGB[2]

for k1 in range(N):
    print ('processing ', k1)

    # five crops
    Xcrop[0,:,:,:]=testData[k1,:,0:224,0:224]
    Xcrop[1,:,:,:]=testData[k1,:,96:320,0:224]
    Xcrop[2,:,:,:]=testData[k1,:,48:272,8:232]
    Xcrop[3,:,:,:]=testData[k1,:,0:224,16:240]
    Xcrop[4,:,:,:]=testData[k1,:,96:320,16:240]

    pred=np.zeros([5,10]) # init predictions
    #for k2 in range(5):
    net.blobs['data'].data[...] =Xcrop
    # perfrom forward operation
    out = net.forward()
    pred=out['prob']
    #print("Predictions are #{}.".format(out['prob'].argmax(axis=1)))
    #print("Predictions are #{}.".format(out['prob']))

    # take average
    net_outMean[k1,:]=np.mean(pred,axis=0)
    # find max row
    net_outMax[k1,:]=pred[np.argmax(pred) / pred.shape[1]]
    net_outCnt[k1, :] = pred[2,:] # center


    # obtain
    class_d[k1]=np.argmax(net_outMean[k1,:],0)
    if class_d[k1] != labels[k1]:
        print k1,class_d[k1],labels[k1]


# display sample
n1=13
I1=np.squeeze(Imgs[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1)
print (n1,class_d[n1],labels[n1],net_outMean[n1,:])



# calculate loss
def cal_loss(scores,labels):
    loss=0
    N=len(labels)
    for k3 in range(N):
        scoreNorm=scores[k3,:]/np.sum(scores[k3,:])
        tmp=int(labels[k3])
        pij=scoreNorm[tmp]
        pij=np.max(np.min(pij,1-1e-15),1e-15)
        loss=loss+np.log(pij)
    logloss=-loss/N
    return logloss


# loss
logloss_mean=cal_loss(net_outMean,labels)
print('logloss using Mean ={},'.format(logloss_mean))


logloss_max=cal_loss(net_outMax,labels)
print('logloss using Max ={},'.format(logloss_max))

logloss_cnt=cal_loss(net_outCnt,labels)
print('logloss using Center ={},'.format(logloss_cnt))



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


# save results
create_submission(net_outMean,labels,'matfile')

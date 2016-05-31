import numpy as np
import sys
import os
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import scipy.misc as scipymisc

#import sys
#clear = lambda: os.system('cls')
#clear()

# find caffe root and import caffe
caffe_root = '/usr/local/caffeapril2016/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import h5py # to read mat files

fold=1
model = '../caffe/net_deploy1.prototxt'
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
X1=f['testData']
Imgs=X1
X1 = np.array(X1, dtype=np.float32)
print ('test data size: ',X1.shape)
#print testData[0,0,:,:]

# resize to 224*240
imr=[224,240]
X=np.zeros([X1.shape[0],3,imr[1],imr[0]],dtype='float32')
for k1 in range(len(X1)):
    I1=np.squeeze(X1[k1, :, :, :])
    I1 = np.transpose(I1, (2, 1, 0))
    I1=scipymisc.imresize(I1, imr, 'bilinear')
    I1=np.transpose(I1,(2,1,0))
    X[k1,:,:,:]=I1
print ('test data size: ', X.shape)



# RGB to BGR
#X = np.array(X,dtype='uint8') # convert to numpy
X=X[:, ::-1, :, :] # RGB to BGR
n1=np.random.randint(len(X))
I1=np.squeeze(X[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
#plt.title(y[n1])
print 'converted RGB to BGR for caffe'

# reshape from N C W H to N C H W
testData=np.transpose(X,(0,1,3,2))
I1=np.squeeze(testData[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1,cmap='Greys_r')
#plt.title(y[n1])
print 'converted to N C H W'


# number of images
N=testData.shape[0]

# labels
labels = f['testLabels']
print ('label data size: ',labels.shape)


net_outCnt=np.zeros([N,10])
#meanRGB=np.array([80.25,97.05,95.4])
meanBGR=np.array([95.4,97.05,80.25])

testData[:,0,:,:]=testData[:,0,:,:]-meanBGR[0]
testData[:,1,:,:]=testData[:,1,:,:]-meanBGR[1]
testData[:,2,:,:]=testData[:,2,:,:]-meanBGR[2]
cx=range(8,232) # center crop

for k1 in range(N):
    print ('processing ', k1)

    net.blobs['data'].data[...] =testData[k1,:,:,8:232]
    out = net.forward()
    pred=out['prob']

    net_outCnt[k1, :] = pred



# display sample
n1=13
I1=np.squeeze(Imgs[n1,:,:,:])
I1=np.transpose(I1,(2,1,0))
plt.imshow(I1)


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
create_submission(net_outCnt,labels,'matfile224')

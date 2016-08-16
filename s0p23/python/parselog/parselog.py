
import numpy as np
import atpy
import matplotlib.pylab as plt


# log file location
fold=2
trainlogfile='../../caffe/caffe'+str(fold)+'/logs/caffelogtrain.txt'
testlogfile='../../caffe/caffe'+str(fold)+'/logs/caffelogtest.txt'
print trainlogfile
print testlogfile

# load loss file as tabe
logTrain = atpy.Table(trainlogfile, type='ascii')
logTest = atpy.Table(testlogfile, type='ascii')

# parse table
lossTest=logTest.loss;
lossTrain=logTrain.loss;
iterTest=logTest.NumIters;
iterTrain=logTrain.NumIters;

# plot loss
plt.plot(iterTest,lossTest,label='test')
#plt.plot(iterTrain[0:-1:10],lossTrain[0:-1:10],label='train')
plt.legend()


# find min
minLoss=np.min(lossTest)
minLossInd=np.argmin(lossTest)
print 'minLoss  %0.3f'   % minLoss
print 'minLossInd %d' % iterTest[minLossInd]

# sort
lossSorted=np.sort(lossTest)
IndSorted=np.argsort(lossTest)
print 'sorted loss ', iterTest[IndSorted[0:5]]
print 'sorted loss ', lossSorted[0:5]
import numpy as np
import sys
import os
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv



# load submission
path2csv='./subm/submission_matfile_2016-05-28-18-32.csv'
N=4011
scores=np.zeros([N,10])
scoresC=np.zeros([N,10])
Labels=np.zeros([N,1])
with open(path2csv, 'rb') as f:
    reader = csv.reader(f)
    k1=-1
    for row in reader:
        #print row
        if k1>-1:
            scores[k1,:]=row[:-1]
            tmp=list(row[-1])
            Labels[k1,:]=int(tmp[1])
        k1=k1+1



# calculate loss
def cal_loss(scores,labels):
    loss=0
    N=len(labels)
    for k3 in range(N):
        scoreNorm=scores[k3,:]/np.sum(scores[k3,:])
        tmp=int(labels[k3])
        pij=scoreNorm[tmp]
        pij=np.max([np.min([pij,1-1e-15]),1e-15])
        loss=loss+np.log(pij)
    logloss=-loss/N
    return logloss



loss=cal_loss(scores,Labels)
print('logloss using Mean ={},'.format(loss))


# find max elements and set them to one
for k1 in range(len(scores)):
    p=np.argmax(scores[k1,:])
    if scores[k1,p]>.5:
        scores[k1,p]=1
        #scores[k1,0:p]=0
        #scores[k1,p+1:] = 0



lossTh=cal_loss(scores,Labels)
print('logloss using Mean ={},'.format(lossTh))



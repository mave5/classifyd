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
path2csv='./subm/submission_Cntnfolds_2016-05-30-19-58.csv'
scores=np.zeros([79726,10])
img=[]
with open(path2csv, 'rb') as f:
    reader = csv.reader(f)
    k1=-1
    for row in reader:
        #print row
        if k1>-1:
            scores[k1,:]=row[:-1]
            img.append(row[-1])
        k1=k1+1





# find max elements and set them to one
for k1 in range(len(scores)):
    p=np.argmax(scores[k1,:])
    if scores[k1,p]>.5:
        scores[k1,p]=1
        print 'changed'


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


# create new submssion
create_submission(scores,img,'nfoldsTh')

print 'new submission was created '




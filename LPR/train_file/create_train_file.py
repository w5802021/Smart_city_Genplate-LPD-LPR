import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

I = glob.glob(r'E:\smart_city\plate\*jpg')
filename = [i.split('\\')[-1] for i in I]
filename = [''.join(i.split(' ')) for i in filename]
D = pd.DataFrame(filename)
D.columns = ['filename']
D['label'] = [''.join(i.split('.')[0].split('_')[0].split(' '))[:7] for i in filename]
dele = []
for i in range(260730):
    if len(D.ix[i,'label']) != 7:
        dele.append(i)

O= D.drop(index=dele)
O = O.reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(O['filename'], O['label'], test_size=0.3)
train = pd.concat([X_train,y_train],axis=1)
valid = pd.concat([X_test,y_test],axis=1)
train.to_csv(r'E:\smart_city\crnn_ctc_ocr\LPR\train.txt',index=None,header=None,sep=" ",encoding='utf-8')
valid.to_csv(r'E:\smart_city\crnn_ctc_ocr\LPR\valid.txt',index=None,header=None,sep=" ",encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GroupKFold,LeaveOneGroupOut,cross_val_score
from sklearn.svm import SVC
n_folds=3
store=pd.HDFStore('train_data.h5')
pca_data=store['data']
labels=store['labels']
access=store['accessions']
store.close()

#cv=KFold(n_folds).split(pca_data)
cv=GroupKFold(n_folds).split(pca_data,labels,access)
#cv=LeaveOneGroupOut().split(pca_data,labels,access)
for train_index , test_index in cv:
    print('Train labels {}'.format(len(np.unique(labels[train_index]))))
    print('Test labels {}'.format(len(np.unique(labels[test_index]))))
    print('Train groups {}'.format(len(np.unique(access[train_index]))))
    print('Test groups {}'.format(len(np.unique(access[test_index]))))
    print()
model=SVC(kernel='linear')
scores=cross_val_score(model,pca_data,labels,access,cv=GroupKFold(n_folds),scoring='accuracy')
print(scores)
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GroupKFold,LeaveOneGroupOut
#from sklearn.SVM import SVC
n_folds=3
store=pd.HDFStore('train_pca.h5')
pca_data=store['data']
labels=store['labels']
access=store['accessions']
store.close()
print(pca_data.shape)
print(labels.shape)
print(access.shape)

#cv=KFold(3).split(pca_data)
cv=GroupKFold(n_folds).split(pca_data,labels,access)
#cv=LeaveOneGroupOut().split(pca_data,labels,access)
for train_index , test_index in cv:
    print('Train labels {}'.format(len(np.unique(labels[train_index]))))
    print('Test labels {}'.format(len(np.unique(labels[test_index]))))
    print('Train groups {}'.format(len(np.unique(access[train_index]))))
    print('Test groups {}'.format(len(np.unique(access[test_index]))))
    print()



#Classification: SVC Linear Kernel; 5-Fold validation
#"""model = SVC(kernel='linear', probability=True)
#kfs = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(rpkm_scaled)
#scores = cross_val_score(model, rpkm_scaled, label, scoring='accuracy', cv = kfs)
#print(scores)

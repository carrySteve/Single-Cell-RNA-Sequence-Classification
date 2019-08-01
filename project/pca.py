import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_val_score
var_threshold=0.99
print("Start")
store = pd.HDFStore('all_data.h5')
rpkm_matrix = store['rpkm']
label = store['labels']
access = store['accessions']
store.close()
print(rpkm_matrix.shape)
print('Data Loading Done')
#Dimension Reduction:PCA
pca=PCA(n_components=100)
pca.fit(rpkm_matrix)
var=0
for i in range(100):
    var_t=var
    var=np.sum(pca.explained_variance_ratio_[:i])
    if ((var_t-var_threshold)*(var-var_threshold)<0):
        n_components=i
        print("n_components:{}".format(n_components))
store=pd.HDFStore('train_data.h5')
train_data=store['rpkm']
train_pca=pd.DataFrame(pca.transform(train_data)[:,:n_components])
print(train_pca.shape)
train_pca.to_hdf('train_pca.h5',key='data',mode='w')
store['labels'].to_hdf('train_pca.h5',key='labels')
store['accessions'].to_hdf('train_pca.h5',key='accessions')
store.close()
print("train transformed")
store=pd.HDFStore('test_data.h5')
test_data=store['rpkm']
test_pca=pd.DataFrame(pca.transform(test_data)[:,:n_components])
print(test_pca.shape)
test_pca.to_hdf('test_pca.h5',key='data',mode='w')
store['labels'].to_hdf('test_pca.h5',key='labels')
store['accessions'].to_hdf('test_pca.h5',key='accessions')
store.close()
print("test transformed")
print('PCA Done')

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_val_score
n_components=0.95
n_folds=5
#Data Loading
print("Start")
store = pd.HDFStore('train_data.h5')
rpkm_matrix = store['rpkm']
label = store['labels']
store.close()
print(rpkm_matrix.shape)
print('Data Loading Done')
#Dimension Reduction:PCA
pca=PCA(n_components=0.95,svd_solver='full')
rpkm_matrix_pca=pca.fit_transform(rpkm_matrix)
print(rpkm_matrix_pca.shape)
print('PCA Done')
###Normalization
#scaler=StandardScaler()
#rpkm_scaled=scaler.fit_transform(rpkm_matrix_pca)
#print('Normalization Done')
#Classification: SVC Linear Kernel; 5-Fold validation
#"""model = SVC(kernel='linear', probability=True)
#kfs = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(rpkm_scaled)
#scores = cross_val_score(model, rpkm_scaled, label, scoring='accuracy', cv = kfs)
#print(scores)


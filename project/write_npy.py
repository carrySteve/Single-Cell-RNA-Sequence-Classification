import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__TRAINDATA_PATH = '../train_data.h5'
__TESTDATA_PATH = '../test_data.h5'

__STORE_PATH = 'data/'

__N_COMPONENTS = {99: 83, 95: 11}

for phase in ['train', 'test']:
    #Data Loading
    print("Start")
    if phase == 'train':
        store = pd.HDFStore(__TRAINDATA_PATH)
    else:
        store = pd.HDFStore(__TESTDATA_PATH)
    rpkm_matrix = store['rpkm']
    label = store['labels']
    store.close()
    print(rpkm_matrix.shape)
    # get element from pandas
    # print(type(label))
    # print(type(rpkm_matrix.iloc[[2]]))
    # print(torch.tensor(rpkm_matrix.iloc[[2]].values))
    # print(type(label.iloc[[2]].values.item()))
    print('Data Loading Done')
    #Dimension Reduction:PCA
    for var, component in __N_COMPONENTS.items():
        pca = PCA(n_components=component)
        rpkm_matrix_pca = pca.fit_transform(rpkm_matrix)
        print(rpkm_matrix_pca.shape)
        print('PCA Done')
        ###Normalization
        scaler = StandardScaler()
        rpkm_scaled = scaler.fit_transform(rpkm_matrix_pca)
        # numpy array
        print('Normalization Done')
        # print(type(rpkm_scaled), rpkm_scaled.shape)
        # print(type(label.to_numpy()), label.to_numpy().shape)

        path = 'data/{}/'.format(phase)
        if (not os.path.exists(path)):
            os.makedirs(path)
        print('saving data')
        np.save(path + 'data-{}.npy'.format(var), rpkm_scaled)
        print('saving label')
        np.save(path + 'label-{}.npy'.format(var), label.to_numpy())

        # test reading
        print('reading data')
        assert np.load(path +
                       'data-{}.npy'.format(var)).shape == rpkm_scaled.shape
        print('reading label')
        assert np.load(path + 'label-{}.npy'.format(var)).shape == label.shape

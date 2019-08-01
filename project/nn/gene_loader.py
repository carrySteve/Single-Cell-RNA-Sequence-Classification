import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ALL_TRAIN_DATA_PATH = '../train_data.h5'
ALL_TEST_DATA_PATH = '../test_data.h5'

VAR = {83: 99, 11: 95}

PCA_DATA_PATH = 'data/'

LABEL = [
    'CL:0002033 short term hematopoietic stem cell',
    'CL:0001056 dendritic cell, human',
    'UBERON:0000044 dorsal root ganglion', 'CL:0000057 fibroblast',
    'UBERON:0001902 epithelium of small intestine', 'CL:0000084 T cell',
    'CL:0000137 osteocyte', 'UBERON:0001997 olfactory epithelium',
    "UBERON:0001954 Ammon's horn", 'UBERON:0000966 retina',
    'UBERON:0010743 meningeal cluster',
    'CL:0002034 long term hematopoietic stem cell',
    'UBERON:0002038 substantia nigra', 'UBERON:0001264 pancreas',
    'UBERON:0000473 testis', 'CL:0000540 neuron', 'CL:0000127 astrocyte',
    'CL:0000081 blood cell', 'CL:0000235 macrophage',
    'UBERON:0001003 skin epidermis', 'UBERON:0001891 midbrain',
    'UBERON:0002048 lung', 'UBERON:0002107 liver',
    'CL:0000746 cardiac muscle cell', 'UBERON:0000007 pituitary gland',
    'UBERON:0002435 striatum', 'UBERON:0001898 hypothalamus',
    'UBERON:0004129 growth plate cartilage', 'CL:1000497 kidney cell',
    'CL:0000763 myeloid cell', 'CL:0000163 endocrine cell',
    'UBERON:0000992 female gonad', 'CL:0000169 type B pancreatic cell',
    'CL:0002322 embryonic stem cell', 'UBERON:0000955 brain',
    'CL:0000037 hematopoietic stem cell', 'CL:0000353 blastoderm cell',
    'UBERON:0001851 cortex', 'UBERON:0000045 ganglion',
    'CL:0008019 mesenchymal cell', 'UBERON:0000115 lung epithelium',
    'UBERON:0000922 embryo', 'CL:0000192 smooth muscle cell',
    'CL:0002365 medullary thymic epithelial cell',
    'CL:0002319 neural cell', 'CL:0002321 embryonic cell'
]

LABEL_TO_ID = {name: idx for idx, name in enumerate(LABEL)}

class GeneDataset(Dataset):
    def __init__(self, phase, cfg):
        global PCA_DATA_PATH, VAR, ALL_TRAIN_DATA_PATH, ALL_TEST_DATA_PATH, LABEL_TO_ID
        self.read_pca = cfg.read_pca
        self.use_pca = cfg.use_pca

        self.label_to_id = LABEL_TO_ID

        if self.read_pca:
            data_path = PCA_DATA_PATH + '{}/data-{}.npy'.format(
                phase, VAR[cfg.n_components])
            label_path = PCA_DATA_PATH + '{}/label-{}.npy'.format(
                phase, VAR[cfg.n_components])

            print('reading data')
            self.rpkm_scaled = np.load(data_path)
            print('reading label')
            self.label = np.load(label_path)
        else:
            path = ALL_TRAIN_DATA_PATH if phase == 'train' else ALL_TEST_DATA_PATH

            print('Start Reading Data')
            store = pd.HDFStore(path)
            self.rpkm_scaled = store['rpkm']
            self.label = store['labels']
            store.close()

            if self.use_pca:
                pca = PCA(cfg.n_components)
                rpkm_matrix_pca = pca.fit_transform(self.rpkm_scaled)
                print('PCA Done')
                ###Normalization
                scaler = StandardScaler()
                self.rpkm_scaled = scaler.fit_transform(rpkm_matrix_pca)
                # numpy array
                print('Normalization Done')

    def __getitem__(self, index):

        if self.read_pca:
            rpkm = torch.tensor(self.rpkm_scaled[index]).float()
            label = self.label_to_id[self.label[index]]
        else:
            if self.use_pca:
                # numpy array to tensor
                rpkm = torch.tensor(self.rpkm_scaled[index]).float()
                label = self.label_to_id[self.label[index]]
            else:
                # pandas df to tensor
                rpkm = torch.tensor(
                    self.rpkm_scaled.iloc[[index]].values).squeeze().float()
                label = self.label_to_id[self.label.iloc[[index]].values.item()]


        return rpkm, label

    def __len__(self):
        return self.rpkm_scaled.shape[0]

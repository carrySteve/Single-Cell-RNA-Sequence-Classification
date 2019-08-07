import os
import pickle

import numpy as np
import pandas as pd
import torch

# __TRAINDATA_PATH = 'train_data.h5'
# __TESTDATA_PATH = 'test_data.h5'

# with open('new_gene_idx.pkl', 'rb') as sf:
#     new_list_idx = pickle.loads(sf.read())
# # 1817

# for phase in ['train', 'test']:
# # for phase in ['test']:
#     #Data Loading
#     print("Start")
#     if phase == 'train':
#         store = pd.HDFStore(__TRAINDATA_PATH)
#     else:
#         store = pd.HDFStore(__TESTDATA_PATH)
#     print('store', type(store))
    
#     rpkm_matrix = store['rpkm']
#     label = store['labels']
#     store.close()
#     print(rpkm_matrix.shape)

#     new_data = []
#     for gene_idx in new_list_idx:
#         column = rpkm_matrix.iloc[:, gene_idx].values
#         new_data.append(column.reshape(-1, 1))

#     new_data = np.hstack(new_data)

#     path = 'relation/'
#     if (not os.path.exists(path)):
#         os.makedirs(path)
#     print('saving data')
#     np.save(path + '{}_relation.npy'.format(phase), new_data)
#     print('saving label')
#     np.save(path + '{}_label.npy'.format(phase), label.to_numpy())


__GENE_FILE = 'test.txt'
score_path = 'new_score.pkl'

feature_dim = 1817


with open('new_gene.pkl', 'rb') as sf:
    new_list = pickle.loads(sf.read())

if __name__ == "__main__":
    # init_relation_graph = torch.eye(feature_dim)
    init_relation_graph = torch.zeros(feature_dim, feature_dim)
    # # init_relation_graph = init_relation_graph * 999
    # init_relation_graph.requires_grad = False

    gene_list = []
    new_gene_list = []

    with open(score_path, 'rb') as f:
        score_dict = pickle.loads(f.read())
    print('opening score dict done')

    count = 0
    
    for gene_pair in score_dict:
        gene1 = gene_pair[0]
        gene2 = gene_pair[1]
        print('gene1', gene1)
        gene1_idx = new_list.index(gene1)
        gene2_idx = new_list.index(gene2)
        score = score_dict[gene_pair]
        # rule out irrevelent two genes
        init_relation_graph[gene1_idx, gene2_idx] = score
        print(gene1, gene2, score)
        count += 1
        print(count)
    
    init_relation_graph /= 999

    torch.save(init_relation_graph, 'init graph diag-0 reduced normal.pth')
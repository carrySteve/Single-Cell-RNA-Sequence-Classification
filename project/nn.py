import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import DataLoader, Dataset

__TRAIN_DATA_PATH = '../train_data.h5'
__TEST_DATA_PATH = '../test_data.h5'

__LABEL = [
    'CL:0002033 short term hematopoietic stem cell',
    'CL:0001056 dendritic cell, human', 'UBERON:0000044 dorsal root ganglion',
    'CL:0000057 fibroblast', 'UBERON:0001902 epithelium of small intestine',
    'CL:0000084 T cell', 'CL:0000137 osteocyte',
    'UBERON:0001997 olfactory epithelium', "UBERON:0001954 Ammon's horn",
    'UBERON:0000966 retina', 'UBERON:0010743 meningeal cluster',
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
    'CL:0002365 medullary thymic epithelial cell', 'CL:0002319 neural cell',
    'CL:0002321 embryonic cell'
]

__LABEL_TO_ID = {name: idx for idx, name in enumerate(__LABEL)}

__BATCH_SIZE = 8
__NUM_WORKERS = 4

__FEATURE_DIM = 20499
__EMBED_DIM = 2048
__HIDDEN_DIM = 512
__LABEL_NUM = 46

__LEARNING_RATE = 1e-5
__WEIGHT_DECAY = 4e-5
__DROPOUT_RATIO = 0.5

__NUM_EPOCHS = 1000
__TEST_GAP = 1
__STORE_GAP = 10

__LOG_PATH = 'log-embedonly.txt'
__MODEL_PATH = 'model/'

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


class GeneDataset(Dataset):
    def __init__(self, phase, label_to_id):
        if phase == 'train':
            path = '../train_data.h5'
        elif phase == 'test':
            path = '../test_data.h5'
        else:
            assert False
        store = pd.HDFStore(path)
        self.rpkm_matrix = store['rpkm']
        self.label = store['labels']
        store.close()
        self.label_to_id = label_to_id

    def __getitem__(self, index):
        rpkm = torch.tensor(
            self.rpkm_matrix.iloc[[index]].values).squeeze().float()
        label = self.label_to_id[self.label.iloc[[index]].values.item()]

        return rpkm, label

    def __len__(self):
        return self.rpkm_matrix.shape[0]


class ClassifierFC(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, dropout_ratio,
                 label_num):
        super(ClassifierFC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, embed_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout_ratio),
            # nn.Linear(embed_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(embed_dim, label_num))

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, feature):
        label_logits = self.classifier(feature)

        return label_logits


def train(classifier, dataloaders_dict, criterion_dict, optimizer):
    since = time.time()

    for epoch in range(__NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, __NUM_EPOCHS))
        print('-' * 10)

        if (epoch + 1) % __TEST_GAP != 0:
            phases = ['train']
        else:
            phases = ['train', 'test']

        for phase in phases:
            total_loss = 0.0
            total_acc = 0.0
            total_len = 0.0

            if phase == 'train':
                classifier.train()
            else:
                classifier.eval()

            for feature, label in dataloaders_dict[phase]:
                with torch.set_grad_enabled(phase == 'train'):
                    # print(feature.shape)
                    feature = feature.to(device)
                    label = torch.tensor(label).to(device)

                    label_logits = classifier(feature)

                    loss = criterion_dict(label_logits, label)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, cell_label = torch.max(label_logits, 1)

                    total_loss += loss.data
                    total_acc += torch.sum(cell_label == label.data).double()
                    # print(torch.sum(cell_label == label.data))
                    # print(cell_label, label)
                    total_len += feature.shape[0]
                    # print(total_len)
                    # sys.exit(0)

                avg_loss = total_loss / total_len
                if phase == 'test':
                    print(phase, total_acc, total_len)
                avg_acc = total_acc / total_len

                print('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                print('{} {} Loss: {} Acc: {} \n'.format(
                    epoch + 1, phase, avg_loss, avg_acc))

            print('|{}|{}|'.format(
                time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))))
            print('{} {} Loss: {} Acc: {} \n'.format(epoch + 1, phase,
                                                     avg_loss, avg_acc))

            with open(__LOG_PATH, 'a') as f:
                f.write('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                f.write('{} {} Loss: {} Acc: {} \n'.format(
                    epoch + 1, phase, avg_loss, avg_acc))

            if (epoch + 1) % __STORE_GAP == 0:
                state = {
                    'epoch': epoch + 1,
                    'state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    state, __MODEL_PATH +
                    'classifier lr-{} weight_decay-{} embed_dim-{} hidden_dim-{} dropout_ratio-{} epoch-{}.pth'
                    .format(__LEARNING_RATE, __WEIGHT_DECAY, __EMBED_DIM,
                            __HIDDEN_DIM, __DROPOUT_RATIO, epoch + 1))
                # sys.exit(0)


if __name__ == "__main__":
    print('creating classifier')

    classifier = ClassifierFC(feature_dim=__FEATURE_DIM,
                              embed_dim=__EMBED_DIM,
                              hidden_dim=__HIDDEN_DIM,
                              label_num=__LABEL_NUM,
                              dropout_ratio=__DROPOUT_RATIO).to(device)

    dataloaders_dict = {
        phase: DataLoader(GeneDataset(phase, label_to_id=__LABEL_TO_ID),
                          batch_size=__BATCH_SIZE,
                          shuffle=True,
                          num_workers=__NUM_WORKERS)
        #   collate_fn=collate_fn)
        for phase in ['train', 'test']
    }

    criterion_dict = nn.CrossEntropyLoss()

    optimizer = optim.Adam(classifier.parameters(),
                           lr=__LEARNING_RATE,
                           weight_decay=__WEIGHT_DECAY)

    train(classifier, dataloaders_dict, criterion_dict, optimizer)

import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import *
from utils import *
from gene_loader import GeneDataset
from models import ClassifierFC


def train(classifier, dataloaders_dict, criterion_dict, optimizer, cfg):
    since = time.time()

    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, cfg.num_epochs))
        print('-' * 10)

        if (epoch + 1) % cfg.test_gap != 0:
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
                    feature = feature.to(cfg.device)
                    label = torch.tensor(label).to(cfg.device)

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

            with open(cfg.log_path, 'a') as f:
                f.write('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                f.write('{} {} Loss: {} Acc: {} \n'.format(
                    epoch + 1, phase, avg_loss, avg_acc))

            if (epoch + 1) % cfg.store_gap == 0 and phase == 'test':
                state = {
                    'epoch': epoch + 1,
                    'state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    state,
                    cfg.result_path + 'classifier epoch-{} acc-{}.pth'.format(
                        epoch + 1, avg_acc))


def init(cfg):
    cfg.init_config()
    # show_config(cfg)

    print('creating classifier')
    classifier = ClassifierFC(cfg).to(cfg.device)
    print_log(cfg.log_path, classifier)

    print('creating dataloader')
    dataloaders_dict = {
        phase: DataLoader(GeneDataset(phase, cfg),
                          batch_size=cfg.batch_size,
                          shuffle=cfg.shuffle,
                          num_workers=cfg.num_workers)
        for phase in ['train', 'test']
    }
    print('creating dataloader done')

    criterion_dict = nn.CrossEntropyLoss()

    optimizer = optim.Adam(classifier.parameters(),
                           lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    print('start training')
    train(classifier, dataloaders_dict, criterion_dict, optimizer, cfg)

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
from models import AutoEncoder


def train(model, dataloaders_dict, criterion_dict, optimizer, cfg):
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
            total_len = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for feature, _ in dataloaders_dict[phase]:
                with torch.set_grad_enabled(phase == 'train'):
                    # print(feature.shape)
                    feature = feature.to(cfg.device)
                    decoded_feature = model(feature)

                    loss = criterion_dict(decoded_feature, feature)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    total_len += feature.shape[0]

                    total_loss += loss.data

                avg_loss = total_loss / total_len

            print('|{}|{}|'.format(
                time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))))
            print('{} {} Loss: {}'.format(epoch + 1, phase, avg_loss))

            with open(cfg.log_path, 'a') as f:
                f.write('|{}|{}|'.format(
                    time.strftime('%m-%d %H:%M:%S', time.localtime(since)),
                    time.strftime('%m-%d %H:%M:%S',
                                  time.localtime(time.time()))))
                f.write('{} {} Loss: {} \n'.format(epoch + 1, phase, avg_loss))

            if (epoch + 1) % cfg.store_gap == 0 and phase == 'test':
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    state, cfg.result_path +
                    'autoencoder epoch-{} loss-{}.pth'.format(epoch + 1, avg_loss))


def init_auto(cfg):
    cfg.init_config()
    show_config(cfg)

    print('creating model')
    model = AutoEncoder(cfg).to(cfg.device)
    print_log(cfg.log_path, model)

    print('creating dataloader')
    dataloaders_dict = {
        phase: DataLoader(GeneDataset(phase, cfg),
                          batch_size=cfg.batch_size,
                          shuffle=cfg.shuffle,
                          num_workers=cfg.num_workers)
        for phase in ['train', 'test']
    }
    print('creating dataloader done')

    criterion_dict = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    print('start training')
    train(model, dataloaders_dict, criterion_dict, optimizer, cfg)

import numpy as np

import torch.nn as nn


class ClassifierFC(nn.Module):
    def __init__(self, cfg):
        super(ClassifierFC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.embed_dim),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Dropout(p=cfg.dropout_ratio),
            # nn.Linear(embed_dim, hidden_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=cfg.dropout_ratio),
            nn.Linear(cfg.embed_dim, cfg.label_num))

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, feature):
        label_logits = self.classifier(feature)

        return label_logits


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.embed_dim),
            nn.Tanh(),
            nn.Dropout(p=cfg.dropout_ratio),
            # nn.Linear(embed_dim, hidden_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=cfg.dropout_ratio),
            nn.Linear(cfg.embed_dim, cfg.label_num))

        self.decoder = nn.Sequential(
            nn.Linear(cfg.label_num, cfg.embed_dim),
            nn.Tanh(),
            nn.Dropout(p=cfg.dropout_ratio),
            # nn.Linear(embed_dim, hidden_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=cfg.dropout_ratio),
            nn.Linear(cfg.embed_dim, cfg.feature_dim))

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, feature):
        feature = self.encoder(feature)
        feature = self.decoder(feature)
        return feature
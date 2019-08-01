import numpy as np

import torch.nn as nn


class ClassifierFC(nn.Module):
    def __init__(self, cfg):
        super(ClassifierFC, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(cfg.feature_dim, cfg.embed_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout_ratio),
            # nn.Linear(embed_dim, hidden_dim, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=cfg.dropout_ratio),
            nn.Linear(cfg.feature_dim, cfg.label_num))

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, feature):
        label_logits = self.classifier(feature)

        return label_logits
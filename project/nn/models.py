import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RelationNetwork(nn.Module):
    def __init__(self, cfg):
        super(RelationNetwork, self).__init__()

        self.cfg = cfg

        # self.embed_fc = nn.Sequential(
        #     nn.Linear(cfg.feature_dim, cfg.embed_dim),
        #     nn.Tanh(),
        #     nn.Dropout(p=cfg.dropout_ratio))

        self.graph_dim = cfg.graph_dim
        self.relation_dim = cfg.relation_dim

        self.theta = nn.Linear(self.graph_dim, self.relation_dim)
        self.phi = nn.Linear(self.graph_dim, self.relation_dim)

        self.mixed_dropout = nn.Dropout(p=cfg.dropout_ratio)

        self.classifier_fc = ClassifierFC(cfg)
        # self.classifier_fc = nn.Linear(cfg.embed_dim, cfg.label_num)

    def forward(self, feature, init_graph):
        # [B, feature_dim]

        # graph_feature = self.embed_fc(feature).unsqueeze(dim=-1)
        B = feature.shape[0]
        graph_feature = feature.unsqueeze(dim=-1)
        init_graph = init_graph.unsqueeze(dim=0)
        # init_graph = init_graph.repeat(B, self.cfg.feature_dim,
        #                                 self.cfg.feature_dim).view(
        #                                     B, self.cfg.feature_dim,
        #                                     self.cfg.feature_dim)
        # [B, feature_dim] -> [B, embed_dim] -> [B, embed_dim, 1]

        # [B, graph_dim, 1]
        theta_feature = self.theta(graph_feature)
        phi_feature = self.phi(graph_feature)

        similarity_relation_graph = torch.matmul(theta_feature,
                                                 phi_feature.permute(0, 2, 1))
        # [B, graph_dim, 1] [B, 1, graph_dim] -> [B, graph_dim, graph_dim]

        similarity_relation_graph = similarity_relation_graph / np.sqrt(
            self.graph_dim)

        relation_graph = torch.softmax(similarity_relation_graph, dim=2)
        # [B, graph_dim, graph_dim] -> [B, graph_dim, graph_dim(1)]

        relation_feature = F.relu(
            torch.matmul(init_graph + relation_graph, graph_feature).squeeze(dim=-1))
        # [B, graph_dim, graph_dim(1)] [B, graph_dim, 1] -> [B, graph_dim, 1]

        mixed_feature = relation_feature + graph_feature.squeeze(dim=-1)

        mixed_feature = self.mixed_dropout(mixed_feature)

        label_logits = self.classifier_fc(mixed_feature)

        return label_logits

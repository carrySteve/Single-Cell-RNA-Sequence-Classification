import os
import time
import torch


class Config(object):
    def __init__(self):
        # Global
        self.batch_size = 8
        self.num_workers = 4
        self.shuffle = True

        self.read_pca = True
        self.use_pca = False
        self.n_components = 83
        self.feature_dim = 83
        assert self.n_components == self.feature_dim
        # 20499(ALL) 83(VAR 99) 11(VAR 95)
        # self.embed_dim = 128
        # 256
        self.hidden_dim = 512
        self.label_num = 46

        self.learning_rate = 1e-5
        self.weight_decay = 4e-5
        self.dropout_ratio = 0.5

        self.num_epochs = 100
        self.test_gap = 1
        self.store_gap = 10

        self.exp_note = 'Fully Connected Classifier'
        self.exp_name = None

        self.device = torch.device(
            'cuda: 0' if torch.cuda.is_available() else 'cpu')

    def init_config(self):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s]<%s>' % (self.exp_note, time_str)

        self.result_path = 'result/%s' % self.exp_name
        self.log_path = 'result/%s/log.txt' % self.exp_name

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

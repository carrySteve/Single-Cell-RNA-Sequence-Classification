import sys
sys.path.append('.')
from nn import *

cfg=Config()

cfg.read_pca = True
cfg.use_pca = False
cfg.n_components = 83
cfg.feature_dim = 83
# 20499(ALL) 83(VAR 99) 11(VAR 95)
cfg.embed_dim = 128

cfg.learning_rate = 1e-5
cfg.weight_decay = 4e-5
cfg.dropout_ratio = 0.5

cfg.num_epochs = 100
cfg.test_gap = 1
cfg.store_gap = 10

cfg.exp_note = 'Fully Connected Classifier'

init(cfg)
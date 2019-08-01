import sys
sys.path.append('.')
from auto import *

cfg = Config()

cfg.batch_size = 64

cfg.read_pca = False
cfg.use_pca = False
cfg.n_components = 20499
cfg.feature_dim = 20499
# 20499(ALL) 83(VAR 99) 11(VAR 95)
cfg.embed_dim = 2048

cfg.learning_rate = 1e-5
cfg.weight_decay = 4e-5
cfg.dropout_ratio = 0.5

cfg.num_epochs = 500
cfg.test_gap = 1
cfg.store_gap = 10

cfg.exp_note = 'Fully Connected Classifier'
cfg.exp_name = 'Auto read_pca-{} use_pca-{} n_components-{} feature_dim-{} embed_dim-{}'.format(
    cfg.read_pca, cfg.use_pca, cfg.n_components, cfg.feature_dim,
    cfg.embed_dim)

init_auto(cfg)
# Single cell RNA Sequence Classification with Machine Learning

## Abstract

To fully realize how our bodies grow and work, it’s necessary to discover what types of cells we have. RNA-seq is used for us to get expression data for such prediction. However, the expression data can be extremely huge. So, to analyze it properly, we firstly implement feature selection(ANOVA) and dimension reduction(PCA) methods to reduce input dimension. Then, we implement machine learning methods, e.g. SVM, Naïve Bayes, AdaBoost and Neural Networks. We also smooth features using graph convolutional network, which utilizes attention mechanism to learn the individual interaction graph for every sample (cell), residual connection to keep the original information. Also, we utilize interaction scores given by STRING to build global interaction graph among gene features, which help genes to focus on the others that interact actively with them.

## Files Illustration

- nn
  - `data_preprocess/get_reduced.py` is used to construct global interaction graph.
  - `scripts/` contains training parameters config. files.
  - `auto.py` implements AutoEncoder.
  - `config.py` is the default config. file.
  - `gene_loader.py` is used to read expression data.
  - `get_score.py` aims to read and store interaction score given by STRING.
  - `models.py` provides the neural network implementation.
  - `monitor.py` is used to delete useless saved network weights.
  - `nn.py` is the main training function.
  - `utils.py` provides usual tools
  - `write_ini.pt` aims to store the initial graph in PyTorch tensor.
  - `write_npy.py` is used to write data in NumPy files.
- `data_intro.txt` provides data explanation.
- `GNB - No PCA.py` provides the implementation of Gaussian Naive Bayes classifier without PCA.
- `groupkfold.py` aims to test cross validation speed.
- `pca.py` is used to implement PCA
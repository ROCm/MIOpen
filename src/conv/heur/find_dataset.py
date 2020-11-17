import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FindDataset(Dataset):
  def __init__(self, df_filename, arch, transforms = None, direction = None):
    fdb = pd.read_pickle(df_filename)
    if direction:
        fdb = fdb[fdb.direction == direction]
    fdb = fdb[fdb.arch == arch]
    fdb['label'] = fdb.sol_name_0.astype('category')
    y = fdb.label.cat.codes.to_numpy()
    self.cat_map = dict(enumerate(fdb.label.cat.categories))
    self.feature_names = ['batchsize', 'conv_stride_h', 'conv_stride_w', 'fil_h', 'fil_w',
           'in_channels', 'in_h', 'in_w', 'out_channels', 'pad_h', 'pad_w']
    features = fdb[self.feature_names]
    features2 = features.copy()
    features2['batchsize'] = np.log2(features2.batchsize)
    features2['in_h'] = np.log2(features2.in_h)
    features2['in_w'] = np.log2(features2.in_w)
    features2['in_channels'] = np.log2(features2.in_channels)
    features2['out_channels'] = np.log2(features2.out_channels)
    X = features2.to_numpy()

    mu = X.mean(axis=0)
    sig = np.sqrt(X.var(axis=0))

    newX = (X - mu) / (sig)
    self.X = newX.astype('float32')
    self.y = y.astype('int64')
    self.mu = mu
    self.sigma = sig
    self.num_features = len(self.feature_names)
    self.num_solvers = len(self.cat_map)
    self.df = fdb
  def __len__(self):
    return len(self.y)
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    return torch.from_numpy(self.X[idx]), self.y[idx]

import os
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD, NAdam, AdamW
from tqdm import tqdm
from lp_ist import LpISTs2
from iht import IHT
from utils import real_density, real_support_rate
from torch.utils.data import Dataset, DataLoader


class TensorDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        """
        Args:
            features: 输入特征张量，形状如 (N, C, H, W) 或 (N, D)
            labels: 标签张量，形状如 (N,) 或 (N, num_classes)
            transform: 可选的数据增强
        """
        # 确保是张量
        self.features = torch.as_tensor(features)
        self.labels = torch.as_tensor(labels)
        self.transform = transform

        # 验证长度一致
        assert len(self.features) == len(self.labels), "特征和标签长度不匹配"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class OutLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        return x @ self.weight.t()


class Mlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, classes: int, activation=nn.ReLU):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.linear_layer = nn.Linear(input_dim, hidden_dims[0])
        self.out_layers = nn.ModuleList()
        self.hidden_dims = self.hidden_dims + [classes,]
        for i in range(len(self.hidden_dims) - 1):
            self.out_layers.append(OutLayer(self.hidden_dims[i], self.hidden_dims[i + 1]))
        self.act = activation()

    def forward(self, x):
        x = self.linear_layer(x)
        for layer in self.out_layers:
            x = self.act(x)
            x = layer(x)
        return x


def preprocess_data(data: pd.DataFrame, mean=0, std=1):
    """
    preprocess data
    1. drop none or nan values
    2. normalize data
    3. sort by id
    """
    # 1. drop none or nan values
    data = data.dropna()

    # 2. normalize data
    feature_columns = data.columns
    data_mean = data[feature_columns].mean()
    data_std = data[feature_columns].std()
    data = data - data_mean + mean
    mask_std = data_std > 0
    data.loc[:, mask_std] = (data.loc[:, mask_std] - data_mean[mask_std]) * (std / data_std[mask_std])

    # 3. sort by id
    data = data.sort_index()
    return data


def preprocess_labels(labels: pd.DataFrame, data_index):
    """
    preprocess labels
    1. select rows by data_index
    2. convert labels to int
    """
    # 1. select rows by data_index
    labels = labels.reindex(data_index)

    # 2. convert labels to int
    labels['Class_encoded'] = pd.Categorical(labels[labels.columns[0]]).codes
    return labels


def load_data_and_preprocess(path: str, data_file="data.csv", label_file="labels.csv", mean=0, std=1):
    """
    load TCGA-PANCAN-HiSeq-801x20531 data and labels from path
    """
    print(f"load data from {path}/{data_file} ..")
    data = pd.read_csv(os.path.join(path, data_file), header=0, index_col=0)
    print(f"data loaded, shape: {data.shape}")
    print(f"load labels from {path}/{label_file} ..")
    labels = pd.read_csv(os.path.join(path, label_file), header=0, index_col=0)
    print(f"labels loaded, shape: {labels.shape}")

    print(f"preprocess data ..")
    data = preprocess_data(data, mean=mean, std=std)

    print(f"preprocess labels ..")
    selected_rows = data.index.tolist()
    labels = preprocess_labels(labels, selected_rows)
    return data, labels


def lr_lambda(epoch, warmup_epochs=10, total_epochs=5000, min_r=0.1):
    if epoch < warmup_epochs:
        return min_r * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_r ** progress


def cosine_lambda(epoch, total_epochs=100, warmup_epochs=5, min_r=0.1):
    if epoch < warmup_epochs:
        return min_r + (1 - min_r) * epoch / warmup_epochs  # 预热
    else:
        # 余弦衰减
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_r + (1 - min_r) * 0.5 * (1 + math.cos(math.pi * progress))

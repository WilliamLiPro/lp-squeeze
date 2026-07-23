import os
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD, NAdam, AdamW
from tqdm import tqdm
from lp_ist import LpIST, LpISTx, LpISTs, LpISTs2, LpISTsx
from lpsqueeze import lpSqueeze, lpSqueezes, lpSqueeze2, lpSqueezes2
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


def simple_trainer(model, criterion, opt, lr_scheduler, constrain, dataloader, epochs, device):
    tq = tqdm(range(epochs))
    loss_list = []
    nonzero_rate_list = []
    support_rate_list = []
    lr_list = []
    model.train().to(device)
    sample_n = sum(x.shape[0] for x, y in dataloader)

    epoch_loss = 0
    for epoch in range(epochs):
        samples = 0
        lr_scheduler.step()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            constrain.constrain_param(epoch + samples / sample_n)

            epoch_loss += loss.item() * x.shape[0]
            samples += x.shape[0]
            # if loss.isnan():
            #     print(f"epoch {epoch}, loss is nan, p={constrain.curr_p}")
            #     raise ValueError
        epoch_loss /= samples
        nonzero_r = real_density(model)
        support_r = real_support_rate(model, constrain)

        tq.update()
        tq.set_postfix({"loss": f"{epoch_loss:.2e}", "nonzero rate": f"{nonzero_r:.2e}", "support rate": f"{support_r:.2e}"})

        loss_list.append(epoch_loss)
        nonzero_rate_list.append(nonzero_r)
        support_rate_list.append(support_r)
        lr_list.append(opt.param_groups[0]["lr"])
    return loss_list, nonzero_rate_list, support_rate_list, lr_list


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


def train_tcga_lpists(
        batch_sz=256,
        epoch_st=0,
        epoch_end=1800,
        epochs=3200,
        nonzero_fraction=0.001,
        p_st=1,
        p_end=0.2,
        update_topk_every=1,
        relative_zero_h=1e-6,
        init_nonzero_fraction=0.01,
        lr=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    path = "./TCGA-PANCAN-HiSeq-801x20531"
    data, labels = load_data_and_preprocess(path)
    x = torch.from_numpy(data.values).to(torch.float).to(device)
    y = torch.from_numpy(labels['Class_encoded'].values).to(torch.long).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)

    n = x.shape[1]
    classes = len(torch.unique(y))
    hidden_dims = [32, ]
    net_0 = Mlp(n, hidden_dims, classes).to(device)

    criterion = nn.CrossEntropyLoss()
    constrain_list = [
        IHT(nonzero_fraction, epoch_st, epoch_end, 0.1),
        # LpIST(nonzero_fraction, epoch_st, epoch_end, p_st, p_end, update_topk_every, relative_zero_h, init_nonzero_fraction),
        # LpISTx(nonzero_fraction, epoch_st, epoch_end, p_st, p_end, update_topk_every, relative_zero_h, init_nonzero_fraction),
        # LpISTs(nonzero_fraction, epoch_st, epoch_end, p_st, p_end, update_topk_every, relative_zero_h, init_nonzero_fraction),
        LpISTs2(nonzero_fraction, epoch_st, epoch_end, p_st, p_end, update_topk_every, relative_zero_h,
               init_nonzero_fraction),
        # lpSqueezes(nonzero_fraction, epoch_st, epoch_end, p_st, p_end, update_topk_every, relative_zero_h,
        #        init_nonzero_fraction),
    ]

    for constrain in constrain_list:
        print(constrain.__class__.__name__)
        net = Mlp(n, hidden_dims, classes).to(device)
        net.load_state_dict(net_0.state_dict())
        if constrain.__class__.__name__ == 'IHT':
            constrain.parameters(net, ["linear"])
        else:
            constrain.parameters(net, ["linear"],)
        opt = AdamW(net.parameters(), lr=lr, weight_decay=0.0,)
        # opt = SGD(net.parameters(), lr=lr * 10, momentum=0.9)
        lr_scheduler = LambdaLR(opt, lr_lambda=lambda epoch: cosine_lambda(epoch, total_epochs=epochs, warmup_epochs=500, min_r=0.02))
        loss_list, nonzero_rate_list, support_rate_list, lr_list = simple_trainer(
            net, criterion, opt, lr_scheduler, constrain, dataloader, epochs, device=device)

        torch.save({"loss_list": loss_list, "nonzero_rate_list": nonzero_rate_list,
                    "support_rate_list": support_rate_list, "lr_list": lr_list},
                   f"./output/TCGA_pancan_HiSeq_sparse_nn_{constrain.__class__.__name__}.pt")


def test_load_data_and_preprocess():
    """
    test load_data_and_preprocess function
    """
    path = "E:\Dataset\TCGA-PANCAN-HiSeq-801x20531"
    data, labels = load_data_and_preprocess(path)
    assert data is not None
    assert labels is not None


if __name__ == "__main__":
    # test_load_data_and_preprocess()
    train_tcga_lpists()

"""
最后一层稠密：
nonzero_fraction=0.005,
IHT     100%|██████████| 3200/3200 [00:37<00:00, 86.42it/s, loss=2.26e-07, nonzero rate=5.30e-03, support rate=5.00e-03]
LpIST   100%|██████████| 3200/3200 [00:44<00:00, 72.41it/s, loss=1.13e-06, nonzero rate=5.30e-03, support rate=5.00e-03]
LpISTx  100%|██████████| 3200/3200 [00:44<00:00, 71.15it/s, loss=9.74e-06, nonzero rate=5.30e-03, support rate=5.00e-03]
LpISTs  100%|██████████| 3200/3200 [00:44<00:00, 71.77it/s, loss=1.63e-07, nonzero rate=5.30e-03, support rate=5.00e-03]
LpISTsx 100%|██████████| 3200/3200 [00:46<00:00, 69.52it/s, loss=3.03e-07, nonzero rate=5.30e-03, support rate=5.00e-03]

nonzero_fraction=0.001,
IHT     100%|██████████| 3200/3200 [00:37<00:00, 85.20it/s, loss=3.22e-07, nonzero rate=1.30e-03, support rate=1.00e-03]
LpIST   100%|██████████| 3200/3200 [00:44<00:00, 71.67it/s, loss=4.76e-06, nonzero rate=1.30e-03, support rate=1.00e-03]
LpISTx  100%|██████████| 3200/3200 [00:45<00:00, 70.41it/s, loss=5.61e-06, nonzero rate=1.30e-03, support rate=1.00e-03]
LpISTs  100%|██████████| 3200/3200 [00:43<00:00, 73.04it/s, loss=6.84e-07, nonzero rate=1.30e-03, support rate=1.00e-03]
LpISTsx 100%|██████████| 3200/3200 [00:45<00:00, 70.32it/s, loss=1.97e-06, nonzero rate=1.30e-03, support rate=1.00e-03]

nonzero_fraction=0.0005,
IHT     100%|██████████| 2400/2400 [00:27<00:00, 88.12it/s, loss=5.83e-06, nonzero rate=8.00e-04, support rate=5.01e-04]
LpIST   100%|██████████| 2400/2400 [00:32<00:00, 74.84it/s, loss=1.81e-05, nonzero rate=8.00e-04, support rate=5.01e-04]
LpISTx  100%|██████████| 2400/2400 [00:32<00:00, 73.46it/s, loss=1.70e-03, nonzero rate=8.00e-04, support rate=5.01e-04]
LpISTs  100%|██████████| 2400/2400 [00:36<00:00, 65.81it/s, loss=1.79e-06, nonzero rate=7.93e-04, support rate=5.01e-04]
LpISTsx 100%|██████████| 2400/2400 [00:33<00:00, 72.40it/s, loss=1.12e-04, nonzero rate=8.00e-04, support rate=5.01e-04]

nonzero_fraction=0.0002,
IHT     100%|██████████| 2400/2400 [00:29<00:00, 82.00it/s, loss=2.88e-05, nonzero rate=5.01e-04, support rate=2.01e-04]
LpIST   100%|██████████| 2400/2400 [00:33<00:00, 72.23it/s, loss=1.09e-04, nonzero rate=5.01e-04, support rate=2.01e-04]
LpISTs  100%|██████████| 2400/2400 [00:32<00:00, 74.35it/s, loss=2.27e-06, nonzero rate=5.01e-04, support rate=2.01e-04]

nonzero_fraction=0.0001,
IHT     100%|██████████| 2400/2400 [00:27<00:00, 86.20it/s, loss=1.60e-04, nonzero rate=4.00e-04, support rate=1.00e-04]
LpIST   100%|██████████| 2400/2400 [00:31<00:00, 75.90it/s, loss=3.61e-03, nonzero rate=4.00e-04, support rate=1.00e-04]
LpISTx  100%|██████████| 2400/2400 [00:32<00:00, 73.58it/s, loss=4.10e-03, nonzero rate=4.00e-04, support rate=1.00e-04]
LpISTs  100%|██████████| 2400/2400 [00:29<00:00, 80.07it/s, loss=1.10e-05, nonzero rate=4.00e-04, support rate=1.00e-04]
LpISTsx 100%|██████████| 2400/2400 [00:32<00:00, 74.55it/s, loss=2.61e-04, nonzero rate=4.00e-04, support rate=1.00e-04]
"""
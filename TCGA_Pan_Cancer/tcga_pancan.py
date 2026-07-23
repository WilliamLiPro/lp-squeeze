import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, NAdam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from lp_ist import LpISTs, LpISTs2, LpISTx
# from lpsqueeze import lpSqueeze, lpSqueezes, lpSqueezesM, lpSqueeze2
from iht import IHT
from Cardinality_Constrained_Least_Squares import FISTA, prox_l1
from TCGA_Pan_Cancer import CGIHT
from TCGA_pancan_HiSeq_sparse_nn import Mlp, load_data_and_preprocess, cosine_lambda, TensorDataset
from utils import real_density, real_support_rate
from torch.utils.data import DataLoader


def simple_trainer(model, criterion, opt, lr_scheduler, constrain, dataloader, epochs, device):
    tq = tqdm(range(epochs))
    loss_list = []
    acc_list = []
    nonzero_rate_list = []
    support_rate_list = []
    lr_list = []
    model.train().to(device)
    sample_n = sum(x.shape[0] for x, y in dataloader)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        samples = 0
        lr_scheduler.step()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            if constrain is not None:
                if hasattr(constrain, "update_g"):
                    constrain.update_g(epoch + samples / sample_n)
            opt.step()
            if constrain is not None:
                constrain.constrain_param(epoch + samples / sample_n)

            epoch_acc += (y_pred.argmax(dim=1) == y).sum().item()
            epoch_loss += loss.item() * x.shape[0]
            samples += x.shape[0]
            # if loss.isnan():
            #     print(f"epoch {epoch}, loss is nan, p={constrain.curr_p}")
            #     raise ValueError
        epoch_loss /= samples
        epoch_acc /= samples

        nonzero_r = real_density(model)
        if constrain is not None:
            support_r = real_support_rate(model, constrain)
        else:
            support_r = nonzero_r

        tq.update()
        tq.set_postfix({"loss": f"{epoch_loss:.2e}", "acc": f"{epoch_acc:.2e}", "nonzero rate": f"{nonzero_r:.2e}", "support rate": f"{support_r:.2e}"})

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        nonzero_rate_list.append(nonzero_r)
        support_rate_list.append(support_r)
        lr_list.append(opt.param_groups[0]["lr"])
    return loss_list, acc_list, nonzero_rate_list, support_rate_list, lr_list


def single_run(
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
        u=0.001,
        lr=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    path = "../TCGA-PANCAN-HiSeq-801x20531"
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
    method_list = [
        # "IHT-AdamW",
        # "Iterative-HTP-AdamW",
        # "CGiht-AdamW",
        "FISTA",
        # "LpISTs2-AdamW",
    ]

    for method in method_list:
        print(method)
        net = Mlp(n, hidden_dims, classes).to(device)
        net.load_state_dict(net_0.state_dict())
        if 'IHT' in method:
            print(f"running IHT with alpha={nonzero_fraction}")
            constrain = IHT(nonzero_fraction, epoch_st, epoch_end, 0.1)
            constrain.parameters(net, ["linear"])
        elif 'Iterative-HTP' in method:
            print(f"running Iterative-HTP with alpha={nonzero_fraction}")
            constrain = IHT(nonzero_fraction, epoch_st, epoch_end, update_topk_every)
            constrain.parameters(net, ["linear"])
        elif 'CGiht' in method:
            print(f"running CGIHT with alpha={nonzero_fraction}")
            constrain = CGIHT(nonzero_fraction, epoch_st, epoch_end)
            constrain.parameters(net, ["linear"])
        elif 'FISTA' in method:
            print(f"running FISTA with u={u}")
            prox = lambda x, t: prox_l1(x, t, u)
            lin_weight = net.linear_layer.weight
            idx = torch.topk(lin_weight.data.abs().view(-1), int(init_nonzero_fraction * lin_weight.numel()))[1]
            lin_weight.data.view(-1)[idx] = 0
            opt = FISTA([{'params': net.linear_layer.parameters(), "use_fista": True},
                         {'params': net.out_layers.parameters(), "use_fista": False}],
                        prox, lr, momentum=0.9)
            constrain = None
        elif 'LpISTs2' in method:
            constrain = LpISTs2(nonzero_fraction, epoch_st, epoch_end, p_st, p_end, update_topk_every, relative_zero_h,
               init_nonzero_fraction)
            constrain.parameters(net, ["linear"])
        else:
            raise ValueError(f"method {method} is not implemented")

        if "AdamW" in method:
            opt = AdamW(net.parameters(), lr=lr, weight_decay=0.0,)
        elif "SGD" in method:
            opt = SGD(net.parameters(), lr=lr * 10, momentum=0.9)

        lr_scheduler = LambdaLR(opt, lr_lambda=lambda epoch: cosine_lambda(epoch, total_epochs=epochs, warmup_epochs=500, min_r=0.02))
        loss_list, acc_list, nonzero_rate_list, support_rate_list, lr_list = simple_trainer(
            net, criterion, opt, lr_scheduler, constrain, dataloader, epochs, device=device)

        if method is "FISTA":
            torch.save({"loss_list": loss_list, "acc_list": acc_list, "nonzero_rate_list": nonzero_rate_list,
                        "support_rate_list": support_rate_list, "lr_list": lr_list},
                       f"TCGA_pancan_{method}_u_{u}.pt")
        else:
            torch.save({"loss_list": loss_list, "acc_list": acc_list, "nonzero_rate_list": nonzero_rate_list,
                        "support_rate_list": support_rate_list, "lr_list": lr_list},
                       f"TCGA_pancan_{method}_alpha_{nonzero_fraction}.pt")


if __name__ == "__main__":
    single_run(nonzero_fraction=0.0008, init_nonzero_fraction=0.01, u=0.15,)
    single_run(nonzero_fraction=0.0005, init_nonzero_fraction=0.01, u=0.17,)
    single_run(nonzero_fraction=0.00025, init_nonzero_fraction=0.01, u=0.18, )
    single_run(nonzero_fraction=0.0001, init_nonzero_fraction=0.01, u=0.2, )


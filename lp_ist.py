"""
Lp-norm constrained Iterative Soft Thresholding
"""
import math
import torch
import torch.nn as nn


def gradient_descent_shrinkage(param, data_abs_p, mask, d_lp_norm_p, p, eta=1e-3, max_iter=4):
    lp_norm_p_no_topk = data_abs_p[~mask].sum().item()
    data = param.data
    if lp_norm_p_no_topk <= d_lp_norm_p:
        return data

    for i in range(max_iter):
        fa = lp_norm_p_no_topk - d_lp_norm_p
        if fa < d_lp_norm_p * eta:
            break
        data_abs = data.abs()
        d_fa_abs = p * (data_abs[~mask] + 1e-6) ** (p - 1)
        beta = fa * d_fa_abs / (d_fa_abs ** 2).sum()
        data[~mask] = data[~mask].sign() * (data_abs[~mask] - beta).relu()
        lp_norm_p_no_topk = (data[~mask].abs() ** p).sum().item()

    data[~mask] = (d_lp_norm_p / lp_norm_p_no_topk) ** (1 / p) * param.data[~mask]
    return data


class LpIST:
    def __init__(
            self,
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st=2,
            p_end=0.1,
            update_topk_every=10.0,
            relative_zero_h=1e-4,
            init_nonzero_fraction=None,
    ):
        self.expected_nonzero_fraction = expected_nonzero_fraction
        self.epoch_st = epoch_st
        self.epoch_end = epoch_end
        assert epoch_st < epoch_end, "epoch_st must be less than epoch_end"
        self.p_st = p_st
        self.p_end = p_end
        assert p_st > p_end, "p_st must be greater than p_end"
        assert p_end > 0, "p_end must be greater than 0"
        self.update_topk_every = update_topk_every
        self.relative_zero_h = relative_zero_h

        self.curr_p = None
        self.curr_epoch = 0
        self.update_epoch = 0
        self.lp_st_norm_p = None
        self.params = None
        self.topk_mask = []
        self.real_nonzero_fraction = None
        self.init_nonzero_fraction = init_nonzero_fraction
        self.topk_h = None

        self.initialize()

    def initialize(self):
        self.curr_p = self.p_st
        self.curr_epoch = 0
        self.update_epoch = self.epoch_st
        self.lp_st_norm_p = None
        self.params = None
        self.real_nonzero_fraction = None

    def parameters(self, net: nn.Module, layer_filter=['linear', 'conv2d', 'conv3d'], nonzero_method='topk'):
        """
        filter parameters by layer name
        """
        self.params = nn.ParameterList()
        for name, p in net.named_parameters():
            if p.requires_grad and 'weight' in name:
                if layer_filter is None or any(layer in name for layer in layer_filter):
                    self.params.append(p)

        if self.init_nonzero_fraction is not None:
            if nonzero_method == 'topk':
                self.real_nonzero_fraction = []
                for param in self.params:
                    param_abs = param.data.abs()
                    param_abs = param_abs.view(-1)
                    _, indices = torch.topk(param_abs, int(math.ceil(param_abs.numel() * self.init_nonzero_fraction)))
                    mask = torch.zeros_like(param_abs, dtype=torch.bool)
                    mask.scatter_(0, indices, True)
                    param.data[~mask.view_as(param)] = 0
                    self.real_nonzero_fraction.append(mask.sum().item() / mask.numel())
            elif nonzero_method == 'zoom':
                for param in self.params:
                    param.data *= self.init_nonzero_fraction

    def state_dict(self):
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def initial_lp_st_norm_p(self):
        self.lp_st_norm_p = []
        for p in self.params:
            self.lp_st_norm_p.append((p.data.abs() ** self.p_st).sum().item())

    def update_p(self, epoch):
        self.curr_epoch = epoch
        if self.epoch_st <= epoch <= self.epoch_end:
            t = ((self.curr_epoch - self.epoch_st) / (self.epoch_end - self.epoch_st))
            # self.curr_p = self.p_st + (self.p_end - self.p_st) * t
            self.curr_p = self.p_st ** (1 - t) * self.p_end ** t

    def update_real_nonzero_fraction(self):
        self.real_nonzero_fraction = []
        for param in self.params:
            # zero elements
            param_abs = param.data.abs()
            zero_h = self.relative_zero_h * param_abs.mean()
            self.real_nonzero_fraction.append((param_abs > zero_h).sum().item() / param.numel())

    def update_topk(self):
        if self.curr_epoch < self.update_epoch:
            return

        self.update_epoch = self.curr_epoch + self.update_topk_every

        self.topk_mask = []
        for param in self.params:
            # topk mask
            param_abs = param.data.abs()
            param_abs = param_abs.view(-1)
            value, indices = torch.topk(param_abs, int(math.ceil(param_abs.numel() * self.expected_nonzero_fraction)))
            mask = torch.zeros_like(param_abs, dtype=torch.bool)
            mask.scatter_(0, indices, True)
            self.topk_mask.append(mask.view_as(param))
            self.topk_h = value[-1]

    def constrain_param(self, epoch):
        if self.params is None:
            raise ValueError("params is None, please call parameters() first")

        if epoch < self.epoch_st:
            return
        elif epoch > self.epoch_end:
            self.curr_epoch = epoch
            self.update_topk()
            for i, param in enumerate(self.params):
                # set no support value to zero
                mask = self.topk_mask[i]
                param.data[~mask] = 0
            return

        self.update_p(epoch)

        if self.lp_st_norm_p is None:
            self.initial_lp_st_norm_p()

        if self.real_nonzero_fraction is None:
            self.update_real_nonzero_fraction()

        # update topk mask
        self.update_topk()

        for i, param in enumerate(self.params):
            mask = self.topk_mask[i]
            data = param.data
            data_abs = data.abs()
            if (data_abs > 0).sum().item() / param.numel() < self.expected_nonzero_fraction:
                continue
            data_abs_p = data_abs ** self.curr_p

            # update lp norm p max
            expected_nonzero_n = self.expected_nonzero_fraction * param.numel()
            lp_norm_p_max = (expected_nonzero_n * (self.lp_st_norm_p[i] / expected_nonzero_n) ** (
                    self.curr_p / self.p_st))

            # update lp norm p topk
            lp_norm_p_topk = data_abs_p[mask].sum().item()
            d_lp_norm_p = lp_norm_p_max - lp_norm_p_topk

            # update no topk value
            if d_lp_norm_p <= 0:
                if d_lp_norm_p < 0:
                    self.lp_st_norm_p[i] = ((lp_norm_p_topk / expected_nonzero_n) ** (self.p_st / self.curr_p) *
                                            expected_nonzero_n)
                param.data[~mask] = 0
            else:
                lp_norm_p_no_topk = data_abs_p[~mask].sum().item()
                if lp_norm_p_no_topk > d_lp_norm_p:
                    fa = lp_norm_p_no_topk - d_lp_norm_p
                    d_fa_abs = self.curr_p * (data_abs[~mask] + 1e-6) ** (self.curr_p - 1)
                    beta = fa * d_fa_abs / (d_fa_abs ** 2).sum()
                    param.data[~mask] = data[~mask].sign() * (data_abs[~mask] - beta).relu()
                    lp_norm_p_no_topk = (param.data[~mask].abs() ** self.curr_p).sum().item()
                    param.data[~mask] = (d_lp_norm_p / lp_norm_p_no_topk) ** (
                            1 / self.curr_p) * param.data[~mask]

                # set short value to zero
                zero_h = self.relative_zero_h * self.topk_h
                param.data[param.data.abs() < zero_h] = 0


class LpISTx(LpIST):
    def __init__(
            self,
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st=2,
            p_end=0.1,
            update_topk_every=10.0,
            relative_zero_h=1e-4,
            init_nonzero_fraction=None,
    ):
        super().__init__(
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st,
            p_end,
            update_topk_every,
            relative_zero_h,
            init_nonzero_fraction,
        )

    def constrain_param(self, epoch):
        if self.params is None:
            raise ValueError("params is None, please call parameters() first")

        if epoch < self.epoch_st:
            return
        elif epoch > self.epoch_end:
            self.curr_epoch = epoch
            self.update_topk()
            for i, param in enumerate(self.params):
                # set no support value to zero
                mask = self.topk_mask[i]
                param.data[~mask] = 0
            return

        self.update_p(epoch)

        if self.real_nonzero_fraction is None:
            self.update_real_nonzero_fraction()

        # update topk mask
        self.update_topk()

        for i, param in enumerate(self.params):
            # lp norm p
            data = param.data
            data_abs = data.abs()
            if (data_abs > 0).sum().item() / param.numel() < self.expected_nonzero_fraction:
                continue

            # update lp norm p max
            lp_norm_p_st = (data_abs ** self.p_st).sum().item()
            expected_nonzero_n = self.expected_nonzero_fraction * param.numel()
            lp_norm_p_max = (expected_nonzero_n * (lp_norm_p_st / expected_nonzero_n) ** (
                        self.curr_p / self.p_st))

            # update lp norm p topk
            mask = self.topk_mask[i]
            data_abs_p = data_abs ** self.curr_p
            lp_norm_p_topk = data_abs_p[mask].sum().item()
            d_lp_norm_p = lp_norm_p_max - lp_norm_p_topk

            # update no topk value
            if d_lp_norm_p <= 0:
                param.data[~mask] = 0
            else:
                lp_norm_p_no_topk = data_abs_p[~mask].sum().item()
                if lp_norm_p_no_topk > d_lp_norm_p:
                    param.data[~mask] = (d_lp_norm_p / lp_norm_p_no_topk) ** (
                            1 / self.curr_p) * data[~mask]

                # set short value to zero
                zero_h = self.relative_zero_h * self.topk_h
                param.data[param.data.abs() < zero_h] = 0


class LpISTs(LpIST):
    def __init__(
            self,
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st=2,
            p_end=0.1,
            update_topk_every=10.0,
            relative_zero_h=1e-6,
            init_nonzero_fraction=None,
    ):
        super().__init__(
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st,
            p_end,
            update_topk_every,
            relative_zero_h,
            init_nonzero_fraction,
        )
        self.expected_nonzero_n = []

    def update_topk(self):
        if self.curr_epoch < self.update_epoch:
            return

        self.update_epoch = self.curr_epoch + self.update_topk_every

        self.topk_mask = []
        self.expected_nonzero_n = []
        for i, param in enumerate(self.params):
            # calculate k
            t = ((self.curr_epoch - self.epoch_st) / (self.epoch_end - self.epoch_st))
            expected_nonzero_n = int(math.ceil((self.real_nonzero_fraction[i] ** (1 - t) *
                                                self.expected_nonzero_fraction ** t) * param.numel()))

            """
            expected_nonzero_n 直接基于 epoch
            """

            # topk mask
            param_abs = param.data.abs()
            param_abs = param_abs.view(-1)
            _, indices = torch.topk(param_abs, expected_nonzero_n)
            mask = torch.zeros_like(param_abs, dtype=torch.bool)
            mask.scatter_(0, indices, True)
            self.topk_mask.append(mask.view_as(param))
            self.expected_nonzero_n.append(expected_nonzero_n)

    def simple_update_topk(self):
        if self.curr_epoch < self.update_epoch:
            return

        self.update_epoch = self.curr_epoch + self.update_topk_every

        self.topk_mask = []
        for param in self.params:
            param_abs = param.data.abs().view(-1)
            _, indices = torch.topk(param_abs, int(math.ceil(param_abs.numel() * self.expected_nonzero_fraction)))
            mask = torch.zeros_like(param_abs, dtype=torch.bool)
            mask.scatter_(0, indices, True)
            self.topk_mask.append(mask.view_as(param))

    def constrain_param(self, epoch):
        if self.params is None:
            raise ValueError("params is None, please call parameters() first")

        if epoch < self.epoch_st:
            return
        elif epoch > self.epoch_end:
            self.curr_epoch = epoch
            self.simple_update_topk()
            for i, param in enumerate(self.params):
                # set no support value to zero
                mask = self.topk_mask[i]
                param.data[~mask] = 0
            return

        self.update_p(epoch)

        if self.lp_st_norm_p is None:
            self.initial_lp_st_norm_p()

        if self.real_nonzero_fraction is None:
            self.update_real_nonzero_fraction()

        # update topk mask
        self.update_topk()

        for i, param in enumerate(self.params):
            # update lp norm p max
            expected_nonzero_n = self.expected_nonzero_n[i]
            lp_norm_p_max = (expected_nonzero_n * (self.lp_st_norm_p[i] / expected_nonzero_n) ** (
                        self.curr_p / self.p_st))

            # update lp norm p topk
            mask = self.topk_mask[i]
            data = param.data
            data_abs = data.abs()
            if (data_abs > 0).sum().item() / param.numel() < self.expected_nonzero_fraction:
                continue
            data_abs_p = data_abs ** self.curr_p
            lp_norm_p_topk = data_abs_p[mask].sum().item()
            # lp_norm_p_max = (lp_norm_p_topk * (self.lp_st_norm_p[i] / lp_norm_p_topk) ** (
            #         self.curr_p / self.p_st))
            d_lp_norm_p = lp_norm_p_max - lp_norm_p_topk

            # update no topk value
            if d_lp_norm_p <= 0:
                if d_lp_norm_p < 0:
                    self.lp_st_norm_p[i] = ((lp_norm_p_topk / expected_nonzero_n) ** (self.p_st / self.curr_p) *
                                            expected_nonzero_n) # 允许增大范数上限
                param.data[~mask] = 0
            else:
                lp_norm_p_no_topk = data_abs_p[~mask].sum().item()
                if lp_norm_p_no_topk > d_lp_norm_p:
                    data_nos_abs = data_abs[~mask]
                    for _ in range(2):
                        fa = lp_norm_p_no_topk - d_lp_norm_p
                        if fa < d_lp_norm_p * 0.01:
                            break
                        d_fa_abs = self.curr_p * data_nos_abs.clamp_min(1e-16) ** (self.curr_p - 1)
                        beta = fa * d_fa_abs / (d_fa_abs ** 2).sum()
                        param.data[~mask] = param.data[~mask].sign() * (data_nos_abs - beta).relu()
                        data_nos_abs = param.data[~mask].abs()
                        lp_norm_p_no_topk = (data_nos_abs ** self.curr_p).sum().item()
                    param.data[~mask] = (d_lp_norm_p / lp_norm_p_no_topk) ** (
                            1 / self.curr_p) * param.data[~mask]

                # set short value to zero
                zero_h = self.relative_zero_h * data_abs[mask].min()
                param.data[param.data.abs() < zero_h] = 0


class LpISTs2(LpISTs):
    """
    当前最优方法
    """
    def __init__(
            self,
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st=2,
            p_end=0.1,
            update_topk_every=10.0,
            relative_zero_h=1e-6,
            init_nonzero_fraction=None,
    ):
        super().__init__(
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st,
            p_end,
            update_topk_every,
            relative_zero_h,
            init_nonzero_fraction,
        )

    def update_p(self, epoch):
        self.curr_epoch = epoch
        if self.epoch_st <= epoch <= self.epoch_end:
            t = ((self.curr_epoch - self.epoch_st) / (self.epoch_end - self.epoch_st))
            # self.curr_p = self.p_st + (self.p_end - self.p_st) * t
            self.curr_p = self.p_st ** (1 - t) * self.p_end ** t

    def update_topk(self):
        if self.curr_epoch < self.update_epoch:
            return

        self.update_epoch = self.curr_epoch + self.update_topk_every

        self.topk_mask = []
        self.expected_nonzero_n = []
        for i, param in enumerate(self.params):
            # calculate k
            t = ((self.curr_p - self.p_end) / (self.p_st - self.p_end))
            """
            expected_nonzero_n 受到 p 调控，而非直接基于 epoch
            """
            expected_nonzero_n = int(math.ceil((self.real_nonzero_fraction[i] ** t *
                                                self.expected_nonzero_fraction ** (1 - t)) * param.numel()))
            # expected_nonzero_n = int(math.ceil((self.real_nonzero_fraction[i] * t +
            #                                     self.expected_nonzero_fraction * (1 - t)) * param.numel()))

            # topk mask
            param_abs = param.data.abs() #+ param.grad * param.data.sign()
            param_abs = param_abs.view(-1)
            _, indices = torch.topk(param_abs, expected_nonzero_n)
            mask = torch.zeros_like(param_abs, dtype=torch.bool)
            mask.scatter_(0, indices, True)
            self.topk_mask.append(mask.view_as(param))
            self.expected_nonzero_n.append(expected_nonzero_n)


class LpISTsx(LpISTs):
    def __init__(
            self,
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st=2,
            p_end=0.1,
            update_topk_every=10.0,
            relative_zero_h=1e-4,
            init_nonzero_fraction=None,
    ):
        super().__init__(
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            p_st,
            p_end,
            update_topk_every,
            relative_zero_h,
            init_nonzero_fraction,
        )

    def constrain_param(self, epoch):
        if self.params is None:
            raise ValueError("params is None, please call parameters() first")

        if epoch < self.epoch_st:
            return
        elif epoch > self.epoch_end:
            self.curr_epoch = epoch
            self.simple_update_topk()
            for i, param in enumerate(self.params):
                # set no support value to zero
                mask = self.topk_mask[i]
                param.data[~mask] = 0
            return

        self.update_p(epoch)

        if self.real_nonzero_fraction is None:
            self.update_real_nonzero_fraction()

        # update topk mask
        self.update_topk()

        for i, param in enumerate(self.params):
            # update lp norm p
            data = param.data
            data_abs = data.abs()
            if (data_abs > 0).sum().item() / param.numel() < self.expected_nonzero_fraction:
                continue
            lp_norm_p_st = (data_abs ** self.p_st).sum().item()

            # update lp norm p max
            expected_nonzero_n = self.expected_nonzero_n[i]
            lp_norm_p_max = (expected_nonzero_n * (lp_norm_p_st / expected_nonzero_n) ** (
                        self.curr_p / self.p_st))

            # update lp norm p topk
            mask = self.topk_mask[i]
            data_abs_p = data_abs ** self.curr_p
            lp_norm_p_topk = data_abs_p[mask].sum().item()
            d_lp_norm_p = lp_norm_p_max - lp_norm_p_topk

            # update no topk value
            if d_lp_norm_p <= 0:
                param.data[~mask] = 0
            else:
                lp_norm_p_no_topk = data_abs_p[~mask].sum().item()
                if lp_norm_p_no_topk > d_lp_norm_p:
                    param.data[~mask] = (d_lp_norm_p / lp_norm_p_no_topk) ** (
                            1 / self.curr_p) * data[~mask]

                # set short value to zero
                zero_h = self.relative_zero_h * data_abs[mask].min()
                param.data[param.data.abs() < zero_h] = 0


def adjust_expected_nonzero_fraction(
        expected_nonzero_fraction: float,
        all_params: list[nn.Parameter],
        selected_params: list[nn.Parameter],
        ):
    n_all = sum([param.numel() for param in all_params])
    n_select = sum([param.numel() for param in selected_params])
    return 1 - (1 - expected_nonzero_fraction) * n_all / n_select


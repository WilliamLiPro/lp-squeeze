"""
Iterative Hard Thresholding
"""


import math
import torch
import torch.nn as nn

class IHT:
    def __init__(
            self,
            expected_nonzero_fraction,
            epoch_st,
            epoch_end,
            update_topk_every=10.0,
    ):
        self.expected_nonzero_fraction = expected_nonzero_fraction
        self.epoch_st = epoch_st
        self.epoch_end = epoch_end
        assert epoch_st < epoch_end, "epoch_st must be less than epoch_end"
        self.update_topk_every = update_topk_every

        self.curr_p = None
        self.curr_epoch = 0
        self.update_epoch = 0
        self.params = None
        self.topk_mask = []

        self.initialize()

    def initialize(self):
        self.curr_epoch = 0
        self.update_epoch = self.epoch_st
        self.params = None

    def parameters(self, net: nn.Module, layer_filter=['linear', 'conv2d', 'conv3d']):
        """
        filter parameters by layer name
        """
        self.params = nn.ParameterList()
        for name, p in net.named_parameters():
            if p.requires_grad and 'weight' in name:
                if layer_filter is None or any(layer in name for layer in layer_filter):
                    self.params.append(p)

    def update_topk(self):
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
        # elif epoch > self.epoch_end:
        #     for i, param in enumerate(self.params):
        #         mask = self.topk_mask[i]
        #         param.data[~mask] = 0

        # update topk mask
        self.curr_epoch = epoch
        self.update_topk()

        for i, param in enumerate(self.params):
            mask = self.topk_mask[i]

            # update no topk value
            param.data[~mask] = 0


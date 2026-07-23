import torch.nn as nn


def real_support_rate(model: nn.Module, constrain=None):
    """
    计算模型整体的非0稠密度
    """
    if constrain is None:
        return 0
    total_params = sum([param.numel() for param in model.parameters()])
    # print(f"Total params: {total_params}")
    support_params = sum(mask.sum().item() for mask in constrain.topk_mask)
    # print(f"Support params: {support_params}")
    return support_params / total_params



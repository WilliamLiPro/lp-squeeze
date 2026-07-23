import torch
import torch.nn as nn
from torchvision.models import resnet50, vit_b_16


def build_model(model_name, num_classes):
    """构建模型"""
    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'vit_base_patch16_224':
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(768, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 初始化权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    return model


def real_density(model: nn.Module, threshold=0):
    """
    计算模型整体的非0稠密度
    """
    total_params = sum([param.numel() for param in model.parameters()])
    # print(f"Total params: {total_params}")
    nonzero_params = sum([(param.abs() > threshold).sum().item() for param in model.parameters()])
    # print(f"Nonzero params: {nonzero_params}")
    return nonzero_params / total_params


def test_real_density(path="./output/checkpoint_epoch_300.pth", model_name='vit_base_patch16_224', num_classes=1000):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint['model'])
    print(f"Real Density: {real_density(model)}")


if __name__ == "__main__":
    test_real_density()
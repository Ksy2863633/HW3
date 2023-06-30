"""
作者: ksy
日期: 2023年 06月 24日
"""
from vit_pytorch import ViT, SimpleViT


# 计算准确率
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def get_vit_model():
    v = SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=256,
        depth=2,
        heads=4,
        mlp_dim=128
    )
    return v

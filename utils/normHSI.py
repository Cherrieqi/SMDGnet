import torch
from torchvision import transforms as tf


# 归一化
def normHSI_all(image, rate=1):
    """
    :param image: tensor,[c, h, w]
    :param rate:
    :return: image_norm: tensor,[c, h, w]
    """
    max_value = torch.max(image)
    min_value = torch.min(image)
    image_norm = rate * (image - min_value) / (max_value - min_value)

    return image_norm


def normHSI_smp_s(image, rate=1, eps=0.00000000001):
    # 将每个样本的值进行归一化
    image_norm = torch.zeros(image.shape)
    for i in range(image.shape[0]):
        max_value = torch.max(image[i])
        min_value = torch.min(image[i])
        image_norm[i] = rate * (image[i] - min_value) / (max_value - min_value+eps)

    return image_norm


def normHSI_net(image, rate=1):
    """
    :param image: tensor,[b, c]
    :param rate:
    :return: image_norm: tensor, [b, c]
    """
    max_value = torch.max(image, dim=0).values
    min_value = torch.min(image, dim=0).values
    image_norm = rate * (image - min_value) / (max_value - min_value)

    return image_norm


# # 标准化
# def normHSI_all(image):
#     """
#     :param image: tensor,[c, h, w]
#     :return: image_norm: tensor,[c, h, w]
#     """
#     mean = torch.mean(image)
#     std = torch.std(image)
#     image_norm = (image - mean) / std
#
#     return image_norm
#
#
# def normHSI_smp_s(image):
#     # 将每个样本的值进行标注化
#     image_norm = torch.full(image.shape, 0.)
#     for i in range(0, image.shape[0]-1, 2):
#         mean = torch.mean(image[i:i+2])
#         std = torch.std(image[i:i+2])
#         image_norm[i:i+2] = (image[i:i+2] - mean) / std
#
#         return image_norm


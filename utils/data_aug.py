import torch
import random


def data_aug(img, gt, class_list: list, num: list, alpha=0.2):
    """
    生成新的数据
    :param img: [N, c, slice, slice]
    :param gt: [N] one-hot之前
    :param class_list: list,分别设置需要生成样本的类别
    :param num: list, 按照类别分别设置需要生成的数量
    :param alpha: 生成数据的加权比例
    :return: SE_img_aug, img_aug, gt_aug 都是新生成的部分，需要跟原数据concat tenser 原来形状换个数量 N'
    """
    img_aug = torch.full([sum(num), img.shape[1], img.shape[2], img.shape[3]], 0.)
    gt_aug = torch.full([sum(num)], 0.)

    random.seed(1234)
    a = 0
    for i, cls in enumerate(class_list):
        idx_list = torch.where(gt==cls)
        for n in range(num[i]):
            random_idx_1 = random.randint(0, idx_list[0].shape[0] - 1)
            random_idx_2 = random.randint(0, idx_list[0].shape[0] - 1)
            if random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(idx_list) - 1)
            img_aug[a] = alpha*img[idx_list[0][random_idx_1]] + (1-alpha)*img[idx_list[0][random_idx_2]]
            gt_aug[a] = cls

            a = a+1

    return img_aug, gt_aug


if __name__ == '__main__':
    img = torch.randn([10, 5, 3, 3])
    gt = torch.cat((torch.full([5], 1), torch.full([5], 2)), dim=0)
    img_aug, gt_aug = data_aug(img, gt, [1, 2], [6, 3])

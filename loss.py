import torch
import torch.nn as nn
import torch.nn.functional as F


def simi_cal(feature_1, feature_2):
    b = feature_1.shape[0]
    feature_1 = torch.reshape(feature_1, [b, -1])
    feature_2 = torch.reshape(feature_2, [b, -1])

    norm_1 = torch.sqrt(torch.sum(torch.mul(feature_1, feature_1), dim=1))
    norm_2 = torch.sqrt(torch.sum(torch.mul(feature_2, feature_2), dim=1))

    feature_1 = feature_1/norm_1.unsqueeze(-1)
    feature_2 = feature_2/norm_2.unsqueeze(-1)

    similarity = feature_1@feature_2.t()

    return similarity


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, similarity, cons_flag):
        diff_vec = similarity - cons_flag*0.5 - 0.5
        diff_loss = torch.mean(diff_vec**2)

        return diff_loss


class DynamicSDPLoss(nn.Module):
    def __init__(self, lambda_trend=1.0, lambda_smooth=0.5):
        super(DynamicSDPLoss, self).__init__()
        self.lambda_trend = lambda_trend
        self.lambda_smooth = lambda_smooth

    def forward(self, similarity_all, cons_flag):
        pair_num = len(similarity_all)
        trend_loss = 0
        for i in range(pair_num-1):  # 正样本 惩罚下降，负样本 惩罚上升
            trend_loss += F.relu(cons_flag*similarity_all[i]-cons_flag*similarity_all[i+1])**2

        smooth_loss = 0
        for i in range(pair_num-2):
            smooth_loss += ((similarity_all[i+2]-similarity_all[i+1])-(similarity_all[i+1]-similarity_all[i]))**2

        prog_loss = self.lambda_trend*torch.mean(trend_loss) + self.lambda_smooth*torch.mean(smooth_loss)

        return prog_loss


class AlignLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(AlignLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        N_s, C = source.size()
        N_t = target.size(0)
        total = torch.cat([source, target], dim=0)  # [N_s + N_t, C]

        sum_total = (total ** 2).sum(1).view(-1, 1)  # [N_s + N_t, 1]
        L2_distance = sum_total + sum_total.t() - 2 * total @ total.t()  # [N_s + N_t, N_s + N_t]

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            n_samples = N_s + N_t
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        assert source.size(1) == target.size(1)
        B_s = source.size(0)
        kernels = self.gaussian_kernel(source, target)

        XX = kernels[:B_s, :B_s]
        YY = kernels[B_s:, B_s:]
        XY = kernels[:B_s, B_s:]
        YX = kernels[B_s:, :B_s]

        loss = torch.mean(XX + YY - XY - YX)
        return loss


import torch
import torch.nn as nn
from networks.HDAM import HDAM
from networks.SE_AEM import SE_AEM
from networks.cal_domain_shift import cal_domain_shift
from networks.IFEH import IFEH
from networks.IFEM import IFEM


class SMDGnet(nn.Module):
    def __init__(self, in_ch, out_ch_shift, out_ch_ifem: list, out_ch: list, img_size, band_num, class_num, slice_size):
        super(SMDGnet, self).__init__()
        self.hdam = HDAM(in_ch)
        self.se_aem = SE_AEM(in_ch, out_ch_shift, img_size)
        self.cal_domain_shift = cal_domain_shift(out_ch_shift, slice_size)
        self.ifeh = IFEH(band_num, out_ch_shift)
        self.ifem = IFEM(out_ch_shift, out_ch_ifem)

        self.classifier = nn.Sequential(nn.Conv2d(out_ch_ifem[3], out_ch[0], kernel_size=slice_size, padding=0, groups=4),
                                        nn.Conv2d(out_ch[0], out_ch[1], kernel_size=1, padding=0, groups=4),
                                        nn.Conv2d(out_ch[1], class_num, kernel_size=1, padding=0))


    def forward(self, x_SE, x_ori, flag=True):
        if flag:
            x_SE = torch.tanh(x_SE)
            x_ori = torch.tanh(x_ori)
        x_SE_new = self.hdam(x_SE)
        y_local = self.se_aem(x_SE_new)
        y_global = self.ifeh(x_ori)
        y_shift = self.cal_domain_shift(y_local, y_global)
        y, y1, y2, y3, y4 = self.ifem(y_shift)

        y = self.classifier(y)
        y = y.squeeze(-1).squeeze(-1)

        y_shift = y_global.reshape(y_shift.shape[0], -1)

        return y, y1, y2, y3, y4, y_shift

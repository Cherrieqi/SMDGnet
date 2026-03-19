import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True, K=4, reduction=1):
        super(DynamicConv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_channels))
        else:
            self.bias = None

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, K, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        B, C, H, W = x.size()

        attention_weights = self.attention(x)

        out = []
        for i in range(self.K):
            conv_out = F.conv2d(
                x,
                self.weight[i],
                bias=self.bias[i] if self.bias is not None else None,
                stride=self.stride,
                padding=self.padding
            )
            out.append(conv_out)

        out = torch.stack(out, dim=0).permute(1, 0, 2, 3, 4)

        attention_weights = attention_weights.view(B, self.K, 1, 1, 1)
        out = (attention_weights * out).sum(dim=1)

        return out


class HDAM(nn.Module):
    def __init__(self, in_ch):
        super(HDAM, self).__init__()
        self.conv = nn.Sequential(DynamicConv(in_ch, in_ch),
                                  Attention())

    def forward(self, x):
        y = self.conv(x)

        return y


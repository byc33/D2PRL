import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import cv2
from torch.utils.checkpoint import checkpoint
from deep_PM import PatchMatch
from scse import SCSEUnet


class DLFerror(nn.Module):
    def __init__(self, VV_mvf7, required_grad):
        super(DLFerror, self).__init__()
        VV_mvf = sio.loadmat(VV_mvf7)
        VV_mvf = VV_mvf['VV_mvf']
        VV_mvf = torch.tensor(VV_mvf, dtype=torch.float32)
        k_size,_,_ = VV_mvf.shape
        output_size = 1
        VV_mvf = VV_mvf.repeat(output_size, 1, 1, 1, 1)
        VV_1 = VV_mvf[:, :, :, :, 0]
        VV_2 = VV_mvf[:, :, :, :, 1]
        VV_3 = VV_mvf[:, :, :, :, 2]
        VV_4 = VV_mvf[:, :, :, :, 3]
        self.VV_1 = torch.nn.Parameter(VV_1, requires_grad=required_grad)
        self.VV_2 = torch.nn.Parameter(VV_2, requires_grad=required_grad)
        self.VV_3 = torch.nn.Parameter(VV_3, requires_grad=required_grad)
        self.VV_4 = torch.nn.Parameter(VV_4, requires_grad=required_grad)
        self.bias_1 = nn.Parameter(torch.zeros(output_size), requires_grad=required_grad)
        self.bias_2 = nn.Parameter(torch.zeros(output_size), requires_grad=required_grad)
        self.bias_3 = nn.Parameter(torch.zeros(output_size), requires_grad=required_grad)
        self.bias_4 = nn.Parameter(torch.zeros(output_size), requires_grad=required_grad)
        self.pad_size = int(np.floor(k_size/2))

    def forward(self, x):
        _, a, _, _ = x.shape
        x1 = x[:, :int(a/2), :, :]
        y1 = x[:, int(a/2):, :, :]
        x_vv1 = F.conv2d(input=torch.mul(x1,x1), weight=self.VV_1, bias=self.bias_1, stride=1, padding=self.pad_size,
                          groups=1)
        x_vv2 = F.conv2d(input=x1, weight=self.VV_2, bias=self.bias_2, stride=1, padding=self.pad_size,
                          groups=1)
        x_vv3 = F.conv2d(input=x1, weight=self.VV_3, bias=self.bias_3, stride=1, padding=self.pad_size,
                          groups=1)
        x_vv4 = F.conv2d(input=x1, weight=self.VV_4, bias=self.bias_4, stride=1, padding=self.pad_size,
                          groups=1)
        x_vv = x_vv1 - (torch.mul(x_vv2,x_vv2) + torch.mul(x_vv3,x_vv3) + torch.mul(x_vv4,x_vv4))

        y_vv1 = F.conv2d(input=torch.mul(y1,y1), weight=self.VV_1, bias=self.bias_1, stride=1, padding=self.pad_size,
                          groups=1)
        y_vv2 = F.conv2d(input=y1, weight=self.VV_2, bias=self.bias_2, stride=1, padding=self.pad_size,
                          groups=1)
        y_vv3 = F.conv2d(input=y1, weight=self.VV_3, bias=self.bias_3, stride=1, padding=self.pad_size,
                          groups=1)
        y_vv4 = F.conv2d(input=y1, weight=self.VV_4, bias=self.bias_4, stride=1, padding=self.pad_size,
                          groups=1)
        y_vv = y_vv1 - (torch.mul(y_vv2,y_vv2) + torch.mul(y_vv3,y_vv3) + torch.mul(y_vv4,y_vv4))

        xy_error = x_vv + y_vv
        return xy_error

class ZM_polar_conv(nn.Module):
    def __init__(self, ZM_polar_filter, input_size, required_grad):
        super(ZM_polar_conv, self).__init__()
        ZM_polar = sio.loadmat(ZM_polar_filter)
        ZM_polar = ZM_polar['ZM_polar_k']
        k_size, k_size, output_size, = ZM_polar.shape
        output_size = int(output_size / 2)
        ZM_polar_real = ZM_polar[:, :, :output_size]
        ZM_polar_imag = ZM_polar[:, :, output_size:]
        T_real = torch.tensor(ZM_polar_real, dtype=torch.float32)
        T_imag = torch.tensor(ZM_polar_imag, dtype=torch.float32)
        T_real = T_real.repeat(input_size, 1, 1, 1).permute([3, 0, 1, 2])
        T_imag = T_imag.repeat(input_size, 1, 1, 1).permute([3, 0, 1, 2])
        self.real_weight = torch.nn.Parameter(T_real, requires_grad=required_grad)
        self.imag_weight = torch.nn.Parameter(T_imag, requires_grad=required_grad)
        self.real_bias = nn.Parameter(torch.zeros(output_size), requires_grad=required_grad)
        self.imag_bias = nn.Parameter(torch.zeros(output_size), requires_grad=required_grad)
        self.pad_size = int(np.floor(k_size / 2))

    def forward(self, x):
        x_real = F.conv2d(input=x, weight=self.real_weight, bias=self.real_bias, stride=1, padding=self.pad_size,
                          groups=1)
        x_imag = F.conv2d(input=x, weight=self.imag_weight, bias=self.imag_bias, stride=1, padding=self.pad_size,
                          groups=1)
        x_conv = torch.sqrt(torch.mul(x_real, x_real) + torch.mul(x_imag, x_imag) + 1e-10)
        return x_conv


class DPM(nn.Module):
    def __init__(self, size, batch_size, pmiter):
        super().__init__()
        self.patchmatch = PatchMatch(pmiter, 50, size, batch_size)
        self.ZM_conv = ZM_polar_conv('ZM_polar_k13.mat', 3, required_grad=False)
        self.head_mask = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
        )
        self.last_mask = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.head_mask2 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.head_mask3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.head_mask4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.last_mask2 = nn.Sequential(
            nn.Conv2d(1344, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.DLFerror11 = DLFerror('VV_mvf11.mat', required_grad=False)
        self.DLFerror7 = DLFerror('VV_mvf7.mat', required_grad=False)
        self.DLFerror9 = DLFerror('VV_mvf9.mat', required_grad=False)
        self.unet = SCSEUnet(backbone_arch='senet154', num_channels=3)
        self.head_mask.apply(weights_init_normal)
        self.head_mask2.apply(weights_init_normal)
        self.head_mask3.apply(weights_init_normal)
        self.head_mask4.apply(weights_init_normal)
        self.last_mask.apply(weights_init_normal)
        self.last_mask2.apply(weights_init_normal)


    def forward(self, x):
        b, c, h, w = x.shape
        xx = F.interpolate(x, size=(int(h//1.5), int(w//1.5)), mode='bilinear', align_corners=True)
        xxx = F.interpolate(x, size=(int(h//0.75), int(w//0.75)), mode='bilinear', align_corners=True)
        x0 = self.ZM_conv(x)
        xx0 = self.ZM_conv(xx)
        xxx0 = self.ZM_conv(xxx)
        x1 = self.head_mask(F.pad(x, pad=[7, 7, 7, 7], mode='reflect'))
        xx1 = self.head_mask(F.pad(xx, pad=[7, 7, 7, 7], mode='reflect'))
        xxx1 = self.head_mask(F.pad(xxx, pad=[7, 7, 7, 7], mode='reflect'))
        xx0 = F.interpolate(xx0, size=(h, w), mode='bilinear', align_corners=True)
        xx1 = F.interpolate(xx1, size=(h, w), mode='bilinear', align_corners=True)
        xxx0 = F.interpolate(xxx0, size=(h, w), mode='bilinear', align_corners=True)
        xxx1 = F.interpolate(xxx1, size=(h, w), mode='bilinear', align_corners=True)
        x_zm = torch.cat([xxx0, x0, xx0], dim=-3)
        x_cnn_g = torch.cat([xxx1, x1, xx1], dim=-3)
        x_cnn = x_cnn_g.clone().detach_()
        xoff, yoff, xoff2, yoff2, x_cor, y_cor, x_cor2, y_cor2 = self.patchmatch(x_zm.half(), x_cnn.half(), x_cnn_g.half())
        x3 = self.DLFerror7(torch.cat([x_cor, y_cor], dim=-3))

        x4 = self.DLFerror7(torch.cat([x_cor2, y_cor2], dim=-3))

        x5 = self.DLFerror9(torch.cat([x_cor, y_cor], dim=-3))

        x6 = self.DLFerror9(torch.cat([x_cor2, y_cor2], dim=-3))

        x7 = self.DLFerror11(torch.cat([x_cor, y_cor], dim=-3))

        x8 = self.DLFerror11(torch.cat([x_cor2, y_cor2], dim=-3))

        x3 = 2 * torch.sigmoid(1 / (x3 + 1e-10)) - 1
        x4 = 2 * torch.sigmoid(1 / (x4 + 1e-10)) - 1
        x5 = 2 * torch.sigmoid(1 / (x5 + 1e-10)) - 1
        x6 = 2 * torch.sigmoid(1 / (x6 + 1e-10)) - 1
        x7 = 2 * torch.sigmoid(1 / (x7 + 1e-10)) - 1
        x8 = 2 * torch.sigmoid(1 / (x8 + 1e-10)) - 1
        xc = torch.cat([xoff, yoff, x3, x5, x7, xoff2, yoff2, x4, x6, x8], dim=-3)

        out1 = self.last_mask(xc)
        out9 = self.unet(x)
        out111 = out1
        if torch.sum(out1*out9)>(torch.sum(out9)*0.5):
            out111 = torch.where(out1 > out9, out1, out9)
        x_cor = x_cor.clone().detach_()
        y_cor = y_cor.clone().detach_()
        x_cor2 = x_cor2.clone().detach_()
        y_cor2 = y_cor2.clone().detach_()
        out12 = out1.clone().detach_()
        out92 = out9.clone().detach_()
        if torch.sum(out12*out92)>(torch.sum(out92)*0.5):
            out12 = torch.where(out12 > out92, out12, out92)
        out11 = out12.round()


        x11 = self.head_mask2(x)
        x12 = self.head_mask3(x11)
        x13 = self.head_mask4(x12)

        x_cor2 -= (x_cor2.size()[3] - 1) / 2
        x_cor2 /= ((x_cor2.size()[3] - 1) / 2)
        y_cor2 -= (y_cor2.size()[2] - 1) / 2
        y_cor2 /= ((y_cor2.size()[2] - 1) / 2)

        x_cor -= (x_cor.size()[3] - 1) / 2
        x_cor /= ((x_cor.size()[3] - 1) / 2)
        y_cor -= (y_cor.size()[2] - 1) / 2
        y_cor /= ((y_cor.size()[2] - 1) / 2)

        samples = torch.cat((x_cor.unsqueeze(4), y_cor.unsqueeze(4)), dim=4)
        samples = samples.view(samples.size()[0] * samples.size()[1], samples.size()[2], samples.size()[3],
                               samples.size()[4])
        cor1 = F.grid_sample(out11, samples, align_corners=True)
        cor1m = out11 * cor1
        cor2m = 1 - cor1m
        xcor3 = x_cor2 * cor1m.round() + x_cor * cor2m.round()
        ycor3 = y_cor2 * cor1m.round() + y_cor * cor2m.round()

        b1, c1, h1, w1 = x13.shape
        out22 = F.interpolate(out11, size=(h1, w1), mode='bilinear', align_corners=True)
        out21 = out22.round()
        x_cor3 = F.interpolate(x_cor, size=(h1, w1), mode='bilinear', align_corners=True)
        y_cor3 = F.interpolate(y_cor, size=(h1, w1), mode='bilinear', align_corners=True)
        x_cor4 = F.interpolate(x_cor2, size=(h1, w1), mode='bilinear', align_corners=True)
        y_cor4 = F.interpolate(y_cor2, size=(h1, w1), mode='bilinear', align_corners=True)
        x_cor5 = F.interpolate(xcor3, size=(h1, w1), mode='bilinear', align_corners=True)
        y_cor5 = F.interpolate(ycor3, size=(h1, w1), mode='bilinear', align_corners=True)
        samples = torch.cat((x_cor3.unsqueeze(4), y_cor3.unsqueeze(4)), dim=4)
        samples = samples.view(samples.size()[0] * samples.size()[1], samples.size()[2], samples.size()[3],
                               samples.size()[4])
        samples1 = torch.cat((x_cor4.unsqueeze(4), y_cor4.unsqueeze(4)), dim=4)
        samples1 = samples1.view(samples1.size()[0] * samples1.size()[1], samples1.size()[2], samples1.size()[3],
                                 samples1.size()[4])
        samples2 = torch.cat((x_cor5.unsqueeze(4), y_cor5.unsqueeze(4)), dim=4)
        samples2 = samples2.view(samples2.size()[0] * samples2.size()[1], samples2.size()[2], samples2.size()[3],
                                 samples2.size()[4])
        x12 = F.interpolate(x12, size=(h1, w1), mode='bilinear', align_corners=True)
        x11 = F.interpolate(x11, size=(h1, w1), mode='bilinear', align_corners=True)
        x112 = torch.cat([x11, x12, x13], dim=-3)
        x113 = F.grid_sample(x112, samples, align_corners=True)
        x114 = F.grid_sample(x112, samples1, align_corners=True)
        out3 = self.last_mask2(torch.cat([x112 * out22, x113 * out22, x114 * out22], dim=-3))
        out3 = out3 * out21 + (1 - out21) * 0.5
        out31 = F.grid_sample(out3, samples2, align_corners=True)
        out4 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=True)
        out41 = F.interpolate(out31, size=(h, w), mode='bilinear', align_corners=True)
        out5 = out4 - out41
        out6 = out5.clone() * out11
        out6[out6 > 0] = 0
        out7 = out5.clone() * out11
        out7[out7 < 0] = 0
        out8 = torch.cat([-out6, out7, 1 - out11], dim=-3)
        return out111, -out6, out7


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class RandomSampler(nn.Module):
    def __init__(self):
        super(RandomSampler, self).__init__()
        self.range_multiplier = torch.arange(0.0, 2.0, 1).view(2, 1, 1)

    def forward(self, min_offset_x, max_offset_x, min_offset_y, max_offset_y, offset_x, offset_y):
        range_multiplier = self.range_multiplier.to(offset_x.get_device())
        noise = torch.rand(min_offset_x.repeat(1, 2, 1, 1).size(), device=offset_x.get_device())
        offset_x1 = min_offset_x + ((max_offset_x - min_offset_x) / 2) * (range_multiplier + noise)
        offset_y1 = min_offset_y + ((max_offset_y - min_offset_y) / 2) * (range_multiplier + noise)
        offset_x1 = offset_x1.unsqueeze_(1).expand(-1, offset_y1.size()[1], -1, -1, -1)
        offset_x1 = offset_x1.contiguous().view(
            offset_x1.size()[0], offset_x1.size()[1] * offset_x1.size()[2], offset_x1.size()[3], offset_x1.size()[4])

        offset_y1 = offset_y1.unsqueeze_(2).expand(-1, -1, offset_y1.size()[1], -1, -1)
        offset_y1 = offset_y1.contiguous().view(
            offset_y1.size()[0], offset_y1.size()[1] * offset_y1.size()[2], offset_y1.size()[3], offset_y1.size()[4])
        offset_x = torch.cat((offset_x1, offset_x), dim=1)
        offset_y = torch.cat((offset_y1, offset_y), dim=1)
        return offset_x, offset_y


class Evaluate_ZM(nn.Module):
    def __init__(self, left_x_coordinate, left_y_coordinate):
        super(Evaluate_ZM, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.left_x_coordinate = left_x_coordinate
        self.left_y_coordinate = left_y_coordinate

    def forward(self, features, offset_x, offset_y):

        left_x_coordinate = self.left_x_coordinate.to(offset_x.get_device())
        left_y_coordinate = self.left_y_coordinate.to(offset_x.get_device())
        if features.size()[0] != self.left_y_coordinate.size()[0]:
            left_x_coordinate = left_x_coordinate[:features.size()[0],:,:,:]
            left_y_coordinate = left_y_coordinate[:features.size()[0], :, :, :]
        offset_strength_list = []
        right_x_coordinate = torch.clamp(left_x_coordinate + offset_x, min=0, max=features.size()[3] - 1)
        right_y_coordinate = torch.clamp(left_y_coordinate + offset_y, min=0, max=features.size()[2] - 1)
        right_x_coordinate -= (right_x_coordinate.size()[3]-1) / 2
        right_x_coordinate /= ((right_x_coordinate.size()[3]-1) / 2)
        right_y_coordinate -= (right_y_coordinate.size()[2]-1) / 2
        right_y_coordinate /= ((right_y_coordinate.size()[2]-1) / 2)
        samples = torch.cat((right_x_coordinate.unsqueeze(4), right_y_coordinate.unsqueeze(4)), dim=4)
        for i in range(samples.size()[1]):
            a = F.grid_sample(features, samples[:,i,:,:,:].half(), align_corners=True)
            a2 = torch.roll(a, 12, 1)
            a3 = torch.roll(a, 24, 1)
            offset_strength1 = (-1.0 * torch.abs(features - a))
            offset_strength2 = (-1.0 * torch.abs(features - a2))
            offset_strength3 = (-1.0 * torch.abs(features - a3))
            offset_strength = torch.cat([torch.mean(offset_strength1[:, 0:12, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength1[:, 12:24, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength1[:, 24:, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength2[:, 0:12, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength2[:, 12:24, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength2[:, 24:, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength3[:, 0:12, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength3[:, 12:24, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength3[:, 24:, :, :], dim=1, keepdim=True),
                                         ], dim=1)
            offset_strength, ind = torch.sort(offset_strength, dim=1, descending=True)
            offset_strength = offset_strength[:, 0, :, :].unsqueeze(1)
            offset_strength_list.append(offset_strength)
        offset_strength = torch.cat(offset_strength_list, dim=1)
        offset_strength = torch.softmax(offset_strength*1000, dim=1)
        offset_x = torch.sum(offset_x * offset_strength, dim=1, keepdim=True)
        offset_y = torch.sum(offset_y * offset_strength, dim=1, keepdim=True)
        offset_y = torch.clamp(offset_y + left_y_coordinate, min=0, max=features.size()[2]-1)-left_y_coordinate
        offset_x = torch.clamp(offset_x + left_x_coordinate, min=0, max=features.size()[3]-1)-left_x_coordinate
        return offset_x, offset_y


class Evaluate_CNN(nn.Module):
    def __init__(self, left_x_coordinate, left_y_coordinate):
        super(Evaluate_CNN, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.left_x_coordinate = left_x_coordinate
        self.left_y_coordinate = left_y_coordinate

    def forward(self, features, offset_x, offset_y):
        left_x_coordinate = self.left_x_coordinate.to(offset_x.get_device())
        left_y_coordinate = self.left_y_coordinate.to(offset_x.get_device())
        if features.size()[0] != self.left_y_coordinate.size()[0]:
            left_x_coordinate = left_x_coordinate[:features.size()[0],:,:,:]
            left_y_coordinate = left_y_coordinate[:features.size()[0], :, :, :]
        offset_strength_list = []
        right_x_coordinate = torch.clamp(left_x_coordinate + offset_x, min=0, max=features.size()[3] - 1)
        right_y_coordinate = torch.clamp(left_y_coordinate + offset_y, min=0, max=features.size()[2] - 1)
        right_x_coordinate -= (right_x_coordinate.size()[3]-1) / 2
        right_x_coordinate /= ((right_x_coordinate.size()[3]-1) / 2)
        right_y_coordinate -= (right_y_coordinate.size()[2]-1) / 2
        right_y_coordinate /= ((right_y_coordinate.size()[2]-1) / 2)
        samples = torch.cat((right_x_coordinate.unsqueeze(4), right_y_coordinate.unsqueeze(4)), dim=4)
        for i in range(samples.size()[1]):
            a = F.grid_sample(features, samples[:,i,:,:,:].half(), align_corners=True)
            a2 = torch.roll(a, 32, 1)
            a3 = torch.roll(a, 64, 1)
            offset_strength1 = (-1.0 * torch.abs(features - a))
            offset_strength2 = (-1.0 * torch.abs(features - a2))
            offset_strength3 = (-1.0 * torch.abs(features - a3))
            offset_strength = torch.cat([torch.mean(offset_strength1[:, 0:32, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength1[:, 32:64, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength1[:, 64:, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength2[:, 0:32, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength2[:, 32:64, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength2[:, 64:, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength3[:, 0:32, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength3[:, 32:64, :, :], dim=1, keepdim=True),
                                         torch.mean(offset_strength3[:, 64:, :, :], dim=1, keepdim=True),
                                         ], dim=1)
            offset_strength, ind = torch.sort(offset_strength, dim=1, descending=True)
            offset_strength = offset_strength[:, 0, :, :].unsqueeze(1)
            offset_strength_list.append(offset_strength)
        offset_strength = torch.cat(offset_strength_list, dim=1)
        offset_strength = torch.softmax(offset_strength*1000, dim=1)
        offset_x = torch.sum(offset_x * offset_strength, dim=1, keepdim=True)
        offset_y = torch.sum(offset_y * offset_strength, dim=1, keepdim=True)
        offset_y = torch.clamp(offset_y + left_y_coordinate, min=0, max=features.size()[2]-1)-left_y_coordinate
        offset_x = torch.clamp(offset_x + left_x_coordinate, min=0, max=features.size()[3]-1)-left_x_coordinate
        return offset_x, offset_y


class PatchMatch(nn.Module):
    def __init__(self, iteration_count, random_search_window_size,size,batch_size):
        super(PatchMatch, self).__init__()
        self.batch_size = batch_size
        self.size = size
        self.iteration_count = iteration_count
        self.window_size = random_search_window_size
        self.random_sample = RandomSampler()
        self.left_x_coordinate = torch.arange(0.0, size).repeat(
            size).view(size, size)

        self.left_x_coordinate = torch.clamp(self.left_x_coordinate, min=0, max=size - 1)
        self.left_x_coordinate = self.left_x_coordinate.expand(batch_size, -1, -1).unsqueeze(1)

        self.left_y_coordinate = torch.arange(0.0, size).unsqueeze(1).repeat(
            1, size).view(size, size)

        self.left_y_coordinate = torch.clamp(self.left_y_coordinate, min=0, max=size - 1)
        self.left_y_coordinate = self.left_y_coordinate.expand(batch_size, -1, -1).unsqueeze(1)
        self.evaluate_ZM = Evaluate_ZM(self.left_x_coordinate, self.left_y_coordinate)
        self.evaluate_CNN = Evaluate_CNN(self.left_x_coordinate, self.left_y_coordinate)
        self.min_offset_x = -1.0 * self.left_x_coordinate
        self.min_offset_y = -1.0 * self.left_y_coordinate
        self.max_offset_x = size-self.left_x_coordinate-1
        self.max_offset_y = size-self.left_y_coordinate-1

    @staticmethod
    def propagation(offset_x, offset_y):
        offset_x = torch.cat((
            offset_x,
            torch.roll(offset_x, 1, 3),
            torch.roll(offset_x, -1, 3),
            torch.roll(offset_x, 1, 2),
            torch.roll(offset_x, -1, 2),
            (2 * torch.roll(offset_x, 1, 3)) - torch.roll(offset_x, 2, 3),
            (2 * torch.roll(offset_x, -1, 3)) - torch.roll(offset_x, -2, 3),
            (2 * torch.roll(offset_x, 1, 2)) - torch.roll(offset_x, 2, 2),
            (2 * torch.roll(offset_x, -1, 2)) - torch.roll(offset_x, -2, 2),
            (2 * torch.roll(offset_x, shifts=(1, 1), dims=(2, 3))) - torch.roll(offset_x, shifts=(2, 2),
                                                                                dims=(2, 3)),
            (2 * torch.roll(offset_x, shifts=(-1, -1), dims=(2, 3))) - torch.roll(offset_x, shifts=(-2, -2),
                                                                                  dims=(2, 3)),
            (2 * torch.roll(offset_x, shifts=(1, -1), dims=(2, 3))) - torch.roll(offset_x, shifts=(2, -2),
                                                                                 dims=(2, 3)),
            (2 * torch.roll(offset_x, shifts=(-1, 1), dims=(2, 3))) - torch.roll(offset_x, shifts=(-2, 2),
                                                                                 dims=(2, 3))
        ),
            dim=1)
        offset_y = torch.cat((
            offset_y,
            torch.roll(offset_y, 1, 3),
            torch.roll(offset_y, -1, 3),
            torch.roll(offset_y, 1, 2),
            torch.roll(offset_y, -1, 2),
            (2 * torch.roll(offset_y, 1, 3)) - torch.roll(offset_y, 2, 3),
            (2 * torch.roll(offset_y, -1, 3)) - torch.roll(offset_y, -2, 3),
            (2 * torch.roll(offset_y, 1, 2)) - torch.roll(offset_y, 2, 2),
            (2 * torch.roll(offset_y, -1, 2)) - torch.roll(offset_y, -2, 2),
            (2 * torch.roll(offset_y, shifts=(1, 1), dims=(2, 3))) - torch.roll(offset_y, shifts=(2, 2),
                                                                                dims=(2, 3)),
            (2 * torch.roll(offset_y, shifts=(-1, -1), dims=(2, 3))) - torch.roll(offset_y, shifts=(-2, -2),
                                                                                  dims=(2, 3)),
            (2 * torch.roll(offset_y, shifts=(1, -1), dims=(2, 3))) - torch.roll(offset_y, shifts=(2, -2),
                                                                                 dims=(2, 3)),
            (2 * torch.roll(offset_y, shifts=(-1, 1), dims=(2, 3))) - torch.roll(offset_y, shifts=(-2, 2),
                                                                                 dims=(2, 3))
        ),
            dim=1)
        return offset_x, offset_y

    def fix_out_of_coordinate(self, offset_x, offset_y):
        left_y_coordinate = self.left_y_coordinate.to(offset_y.get_device())
        left_x_coordinate = self.left_x_coordinate.to(offset_x.get_device())
        if offset_x.size()[0] != self.left_y_coordinate.size()[0]:
            left_x_coordinate = left_x_coordinate[:offset_x.size()[0],:,:,:]
            left_y_coordinate = left_y_coordinate[:offset_x.size()[0], :, :, :]
        offset_y = offset_y + left_y_coordinate
        offset_x = offset_x + left_x_coordinate
        offset_y = torch.where(offset_y > 0, offset_y, offset_y % self.size)
        offset_y = torch.where(offset_y < self.size, offset_y,
                               offset_y % self.size)
        offset_x = torch.where(offset_x > 0, offset_x, offset_x % self.size)
        offset_x = torch.where(offset_x < self.size, offset_x,
                               offset_x % self.size)
        offset_y = offset_y - left_y_coordinate
        offset_x = offset_x - left_x_coordinate
        return offset_x, offset_y

    def non_local(self, offset_x, offset_y, limit_u):
        min_offset_x = self.min_offset_x.to(offset_x.get_device())
        min_offset_y = self.min_offset_y.to(offset_y.get_device())
        if offset_x.size()[0] != self.min_offset_x.size()[0]:
            min_offset_x = min_offset_x[:offset_x.size()[0],:,:,:]
            min_offset_y = min_offset_y[:offset_x.size()[0], :, :, :]
        limit = offset_x ** 2 + offset_y ** 2
        limit1 = torch.zeros_like(limit)
        limit2 = torch.ones_like(limit)
        limit = torch.where(limit > limit_u, limit1, limit2)
        offset_x1 = min_offset_x + self.size * torch.rand(min_offset_x.size(), device=offset_x.get_device())
        offset_y1 = min_offset_y + self.size * torch.rand(min_offset_y.size(), device=offset_y.get_device())
        offset_x = torch.where(limit > 0.5, offset_x1, offset_x)
        offset_y = torch.where(limit > 0.5, offset_y1, offset_y)
        return offset_x, offset_y

    def random_search_window(self, offset_x, offset_y):
        min_offset_x = offset_x - self.window_size // 2
        max_offset_x = offset_x + self.window_size // 2
        min_offset_y = offset_y - self.window_size // 2
        max_offset_y = offset_y + self.window_size // 2
        return min_offset_x, max_offset_x, min_offset_y, max_offset_y

    def forward(self, ZM_features, CNN_features, CNN_features_g):
        device = CNN_features.get_device()
        left_x_coordinate = self.left_x_coordinate.to(device)
        left_y_coordinate = self.left_y_coordinate.to(device)
        min_offset_x_zm = self.min_offset_x.to(device)
        max_offset_x_zm = self.max_offset_x.to(device)
        min_offset_y_zm = self.min_offset_y.to(device)
        max_offset_y_zm = self.max_offset_y.to(device)
        min_offset_x_cnn = self.min_offset_x.to(device)
        max_offset_x_cnn = self.max_offset_x.to(device)
        min_offset_y_cnn = self.min_offset_y.to(device)
        max_offset_y_cnn = self.max_offset_y.to(device)
        if CNN_features.size()[0] != self.batch_size:
            left_x_coordinate = left_x_coordinate[:CNN_features.size()[0],:,:,:]
            left_y_coordinate = left_y_coordinate[:CNN_features.size()[0], :, :, :]
            min_offset_x_zm = min_offset_x_zm[:CNN_features.size()[0], :, :, :]
            max_offset_x_zm = max_offset_x_zm[:CNN_features.size()[0], :, :, :]
            min_offset_y_zm = min_offset_y_zm[:CNN_features.size()[0], :, :, :]
            max_offset_y_zm = max_offset_y_zm[:CNN_features.size()[0], :, :, :]
            min_offset_x_cnn = min_offset_x_cnn[:CNN_features.size()[0], :, :, :]
            max_offset_x_cnn = max_offset_x_cnn[:CNN_features.size()[0], :, :, :]
            min_offset_y_cnn = min_offset_y_cnn[:CNN_features.size()[0], :, :, :]
            max_offset_y_cnn = max_offset_y_cnn[:CNN_features.size()[0], :, :, :]
        offset_x_zm = min_offset_x_zm + self.size * torch.rand(min_offset_x_zm.size(), device=device)
        offset_y_zm = min_offset_y_zm + self.size * torch.rand(min_offset_x_zm.size(), device=device)
        offset_x_cnn = min_offset_x_cnn + self.size * torch.rand(min_offset_x_zm.size(), device=device)
        offset_y_cnn = min_offset_y_cnn + self.size * torch.rand(min_offset_x_zm.size(), device=device)

        for iter_num in range(self.iteration_count):
            offset_x_zm, offset_y_zm = self.random_sample(min_offset_x_zm, max_offset_x_zm,
                                                          min_offset_y_zm, max_offset_y_zm,
                                                          offset_x_zm, offset_y_zm)
            offset_x_zm, offset_y_zm = self.fix_out_of_coordinate(offset_x_zm, offset_y_zm)
            offset_x_zm, offset_y_zm = self.evaluate_ZM(ZM_features, offset_x_zm, offset_y_zm)

            offset_x_cnn, offset_y_cnn = self.random_sample(min_offset_x_cnn, max_offset_x_cnn,
                                                            min_offset_y_cnn, max_offset_y_cnn,
                                                            offset_x_cnn, offset_y_cnn)
            offset_x_cnn, offset_y_cnn = self.fix_out_of_coordinate(offset_x_cnn, offset_y_cnn)
            offset_x_cnn, offset_y_cnn = self.evaluate_CNN(CNN_features, offset_x_cnn, offset_y_cnn)
            if iter_num < self.iteration_count-1:
                offset_x_zm, offset_y_zm = self.propagation(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm = self.fix_out_of_coordinate(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm = self.evaluate_ZM(ZM_features, offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm = self.propagation(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm = self.fix_out_of_coordinate(offset_x_zm, offset_y_zm)
                offset_x_zm, offset_y_zm = self.evaluate_ZM(ZM_features, offset_x_zm, offset_y_zm)

                offset_x_cnn, offset_y_cnn = self.propagation(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.fix_out_of_coordinate(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.evaluate_CNN(CNN_features, offset_x_cnn, offset_y_cnn)

                offset_x_zm, offset_y_zm = self.non_local(offset_x_zm, offset_y_zm, 25)
                offset_x_cnn, offset_y_cnn = self.non_local(offset_x_cnn, offset_y_cnn, 25)

                min_offset_x_zm, max_offset_x_zm, min_offset_y_zm, max_offset_y_zm = self.random_search_window(
                    offset_x_zm, offset_y_zm)
                min_offset_x_cnn, max_offset_x_cnn, min_offset_y_cnn, max_offset_y_cnn = self.random_search_window(
                    offset_x_cnn, offset_y_cnn)

            else:
                offset_x_cnn, offset_y_cnn = self.propagation(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.fix_out_of_coordinate(offset_x_cnn, offset_y_cnn)
                offset_x_cnn, offset_y_cnn = self.evaluate_CNN(CNN_features_g, offset_x_cnn, offset_y_cnn)

        x_cor_zm = left_x_coordinate + offset_x_zm
        y_cor_zm = left_y_coordinate + offset_y_zm
        x_cor_cnn = left_x_coordinate + offset_x_cnn
        y_cor_cnn = left_y_coordinate + offset_y_cnn

        return offset_x_zm, offset_y_zm, offset_x_cnn,\
            offset_y_cnn, x_cor_zm, y_cor_zm,\
            x_cor_cnn, y_cor_cnn


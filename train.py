
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import models_D2PRL as models
import argparse
import cv2
import time
from getitem import Config,Data
from sklearn.metrics import precision_recall_fscore_support
device_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dir_name = time.strftime("weight_%Y_%m_%d_%H:%M:%S", time.localtime())
input_size = 448
savepath = './'+dir_name
batchsize = 3
lr = 1e-3
save_interval = 1000


def dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    Calculate dice loss for PyTorch.
    Dice loss is a loss function used for binary classification problems, where y_true (ground truth labels)
    and y_pred (predicted labels) are two binary tensors. The formula for calculating dice loss is
    1 - (2 * intersection(y_true, y_pred) + epsilon) / (sum(y_true) + sum(y_pred) + epsilon), where
    'intersection' is the element-wise AND operation between the two tensors.
    :param y_true: Ground truth labels (tensor)
    :param y_pred: Predicted labels (tensor)
    :param epsilon: Small value to avoid division by zero (scalar)
    :return: Dice loss (scalar)
    """

    y_true = y_true.flatten(start_dim=1)
    y_pred = y_pred.flatten(start_dim=1)

    intersection = torch.sum(y_true * y_pred, dim=1)
    union = torch.sum(y_true, dim=1) + torch.sum(y_pred, dim=1)

    dice = 1 - (2 * intersection + epsilon) / (union + epsilon)

    return dice.mean()


def save_checkpoint(model, name):
    torch.save(model.state_dict(), os.path.join(savepath, name))


if __name__ == "__main__":
    train_path = './train'
    cfg = Config(train_path=train_path, mode='train', batch=batchsize, size=input_size)
    data = Data(cfg, train_path=train_path)
    train_data_loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    device = torch.device("cuda:0")
    np.random.seed(33)
    torch.manual_seed(33)
    torch.cuda.manual_seed_all(33)
    model = models.DPM(input_size, batchsize, 32)
    # model_path = 'pretrain.pth'
    # moel_dict = model.state_dict()
    # for k, v in torch.load(model_path).items():
    #     moel_dict.update({k[7 :]: v for k, v in torch.load(model_path).items()})



    model.load_state_dict(moel_dict)

    os.makedirs(savepath, exist_ok=True)
    opt_list = list(model.head_mask.parameters())\
               + list(model.last_mask.parameters()) \
               + list(model.head_mask2.parameters()) \
               + list(model.last_mask2.parameters()) \
               + list(model.head_mask3.parameters()) \
               + list(model.head_mask4.parameters()) \
               + list(model.unet.parameters())
    optimizer = torch.optim.Adam(opt_list, lr)

    model = nn.DataParallel(model, device_ids=device_ids)

    num_iter_per_epoch = len(train_data_loader)
    step = 0
    for epoch in range(10):
        model.train()
        epoch = epoch + 1
        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []
        epoch_loss3 = []
        loss_ls = []
        d_loss = 0
        g_loss = 0
        det_loss = 0
        loc_loss = 0
        progress_bar = tqdm(train_data_loader)
        for iter, data in enumerate(progress_bar):
            if step < 0:
                step += 1
                continue
            imgs, gts, _ = data
            imgs = imgs.to(device)
            gts = gts.to(device)
            optimizer.zero_grad()
            simi_gts = (1 - gts[:, 2, :, :]).unsqueeze(1).type(torch.float)
            simi_gts2 = (gts[:, 0, :, :]).unsqueeze(1).type(torch.float)
            simi_gts3 = (gts[:, 1, :, :]).unsqueeze(1).type(torch.float)

            g_out1, g_out2, g_out3, g_out4 = model(imgs, gts)
            _, fusion_gts = gts.max(dim=1)

            g_loss1 = dice_loss(g_out1, simi_gts)
            g_loss2 = dice_loss(g_out4, simi_gts2)
            er, _ = torch.max(torch.cat(
                [0.05 * torch.ones_like(g_out3).to(g_out3.device) - g_out3 * (simi_gts3 - simi_gts2),
                 torch.zeros_like(g_out3).to(g_out3.device)], dim=-3), dim=1, keepdim=True)
            g_loss3 = torch.sum(g_out2 * simi_gts * er * 0.0001)

            loss = g_loss1 + g_loss2 + g_loss3
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss))
            epoch_loss1.append(float(g_loss1))
            epoch_loss2.append(float(g_loss2))
            epoch_loss3.append(float(g_loss3))
            step += 1
            progress_bar.set_description(
                'St: {}. Epo: {}/{}. Iter: {}/{}. loss1: {:.5f}.loss2: {:.5f}.loss3: {:.5f}.loss1_total: {:.5f}.loss2_total: {:.5f}.loss3_total: {:.5f}'.format(
                    step, epoch, 10, iter + 1, num_iter_per_epoch, g_loss1.item(), g_loss2.item(), g_loss3.item(),
                    np.mean(epoch_loss1), np.mean(epoch_loss2), np.mean(epoch_loss3)))

            if step % 1000 == 0 and step > 0:
                save_checkpoint(model, f'{step}.pth')
                print('checkpoint...')
                epoch_loss1 = []
                epoch_loss2 = []
                print('Epoch: {}/{}. loss: {:1.5f}'.format(epoch, 10, np.mean(epoch_loss)))
        print('Epoch: {}/{}. loss: {:1.5f}'.format(epoch, 10, np.mean(epoch_loss)))
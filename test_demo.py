import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import cv2
import time
from pathlib import Path
import models_D2PRL as models
from skimage import morphology
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def post_1c(mask):
    return morphology.remove_small_objects(mask > 0.5, min_size=500)


def post_3c(mask_t, mask_s):
    single_mask = morphology.remove_small_objects((mask_t + mask_s) > 0, min_size=500)
    mask_t = mask_t * single_mask
    mask_s = single_mask - mask_t
    before_fliter_mask = mask_t - mask_s
    fliter_kernel = np.ones([50,50])
    after_fliter_mask = cv2.filter2D(before_fliter_mask, -1, fliter_kernel, borderType=cv2.BORDER_CONSTANT)
    mask_t = (after_fliter_mask > 0)*single_mask
    mask_s = 1.0*single_mask - 1.0*mask_t
    mask_b = 1.0 - (mask_t + mask_s)
    return mask_t, mask_s, mask_b


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    np.random.seed(22)
    torch.manual_seed(22)
    torch.cuda.manual_seed_all(22)
    bce_criterion = nn.BCELoss()
    input_size = 448
    model = models.DPM(input_size, 1, 40)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
    ])
    model_path = './pretrain.pth'
    moel_dict = model.state_dict()
    for k, v in torch.load(model_path).items():
        moel_dict.update({k: v})
    model.load_state_dict(moel_dict)
    model = model.cuda()
    model.eval()

    img_path = "./testpic/1.tif"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_o, w_o, _ = img.shape
    image = test_transform(img)
    c, h, w = image.shape
    image = image.reshape(1, c, h, w)
    image = image.to(device)
    with torch.no_grad():
        pred, predt, preds = model(image)
    pred_ = pred.permute(0, 2, 3, 1).reshape(pred.shape[0], pred.shape[2],
                                             pred.shape[3], 1).cpu().detach().numpy().round()
    pred_ = pred_[0, :, :, 0]
    pred_ = post_1c(pred_)
    cv2.imwrite('./testpic/1_1c_result.png', cv2.resize((pred_ * 255).astype(np.uint8), (w_o, h_o)))
    preds_ = preds.permute(0, 2, 3, 1).reshape(preds.shape[0], preds.shape[2], preds.shape[3],
                                               1).cpu().detach().numpy()
    preds_ = preds_[0, :, :, 0]
    preds_[preds_ > 0] = 1
    predt_ = predt.permute(0, 2, 3, 1).reshape(predt.shape[0], predt.shape[2], predt.shape[3],
                                               1).cpu().detach().numpy()
    predt_ = predt_[0, :, :, 0]
    predt_[predt_ > 0] = 1
    predt_, preds_, predb_ = post_3c(predt_, preds_)
    pred3c = np.concatenate([np.expand_dims(predb_, axis=2), np.expand_dims(preds_, axis=2),
                             np.expand_dims(predt_, axis=2)], axis=2)
    cv2.imwrite('./testpic/1_3c_result.png',  cv2.resize((pred3c * 255).astype(np.uint8), (w_o, h_o)))
    print('test done')


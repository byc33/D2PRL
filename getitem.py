from torch.utils.data import Dataset, DataLoader
import os
import cv2
import skimage
from skimage import io
import numpy as np
import torch
from torchvision import transforms
import random
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def JP_IO(img):
    QF_list = [40, 50, 60, 70, 80, 90, 100]
    QF = random.choice(QF_list)
    buffer = BytesIO()
    img1 = Image.fromarray(img)
    img1.save(buffer, "JPEG", quality=QF)
    buffer.seek(0)  # find the byte stream
    res = buffer.read()
    byte_stream = BytesIO(res)
    image = Image.open(byte_stream)
    image = np.array(image)
    return image

def noise(img):
    mean = 0
    var_list = [0.009, 0.005, 0.0005]
    var = random.choice(var_list)
    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    return np.array(out)

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class CustomTransform:
    def __init__(self, size=224):
        if isinstance(size, int) or isinstance(size, float):
            self.size = (size, size)
        else:
            self.size = size
        self.to_tensor = transforms.ToTensor()

    def resize(self, img=None, mask=None):
        if img is not None:
            img = skimage.img_as_float32(img)
            if img.shape[0] != self.size[0] or img.shape[1] != self.size[1]:
                img = cv2.resize(
                    img, self.size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            if mask.shape[0] != self.size[0] or mask.shape[1] != self.size[1]:
                mask = cv2.resize(
                    mask, self.size, interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __call__(self, img=None, mask=None, other_tfm=None,target_trm=None):
        img, mask = self.resize(img, mask)
        if other_tfm is not None:
            img = other_tfm(img)
            mask = target_trm(mask)
        if img is not None:
            img = self.to_tensor(img).float()

        if mask is not None:
            mask = self.to_tensor(mask).float()

        return img, mask


class Data(Dataset):
    def __init__(self, cfg, train_path=None, val_path=None, casia_path=None, train_flag=True):
        self.cfg = cfg
        self.flag = train_flag
        self.size = cfg.size
        self.train_path = train_path
        self.val_path = val_path
        self.casia_path = casia_path

        #
        if cfg is not None:
            self.transform = CustomTransform(size=cfg.size)
        if train_path != None:
            self.sample1 = os.listdir(train_path + '/image')
        if val_path != None:
            self.sample2 = os.listdir(val_path + '/image')
        if casia_path != None:
            self.sample3 = os.listdir(casia_path + '/image')

    def __getitem__(self, idx):

        if self.cfg.mode == 'train':
            name = self.sample1[idx]
            image = io.imread(self.train_path + '/image/' + name)

            if image.size < 1024*1024*2:
                image = np.expand_dims(image, axis=2)
                image = np.repeat(image, 3, 2)
            mask = cv2.imread(self.train_path + '/mask/' + name).astype(np.float32)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.cfg.mode == 'valid':
            name = self.sample2[idx]
            image = io.imread(self.val_path + '/image/' + name)[:, :, :3]
            mask = cv2.imread(self.val_path + '/mask/' + name[:-4]+'.png').astype(np.float32)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.cfg.mode == 'casia':
            name = self.sample3[idx]
            image = io.imread(self.casia_path + '/image/' + name)[:, :, :3]
            mask = cv2.imread(self.casia_path + '/mask/' + name[:-4]+'.png').astype(np.float32)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mode_list = ['zero', 'zero', 'JC', 'JC', 'JC', 'NA']
        mode = random.choice(mode_list)

        if self.flag:
            if mode == 'zero':
                image = image
            elif mode == 'JC':
                image = JP_IO(image)
            elif mode == 'NA':
                image = noise(image)

        if np.max(mask)==255.0:
            mask = mask/255
        img, cmd_mask = self.transform(image, mask, other_tfm=None, target_trm=None)
        return img, cmd_mask, name

    def __len__(self):
        if self.cfg.mode == 'train':
            return len(self.sample1)
        elif self.cfg.mode == 'valid':
            return len(self.sample2)
        elif self.cfg.mode == 'casia':
            return len(self.sample3)

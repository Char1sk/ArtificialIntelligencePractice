import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class FixScaleCrop(object):
    def __init__(self, crop_size, isImg=True):
        if isinstance(crop_size, tuple):
            self.crop_size_h, self.crop_size_w = crop_size
        else:
            self.crop_size_h, self.crop_size_w = crop_size, crop_size
        self.isImg = isImg

    def __call__(self, sample):
        w, h = sample.size
        if w/self.crop_size_w > h/self.crop_size_h:
            oh = self.crop_size_h
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size_w
            oh = int(1.0 * h * ow / w)
        sample = sample.resize((ow, oh), Image.BILINEAR if self.isImg else Image.NEAREST)
        # center crop
        w, h = sample.size
        x1 = int(round((w - self.crop_size_w) / 2.))
        y1 = int(round((h - self.crop_size_h) / 2.))
        sample = sample.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))

        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=8, isImg=True):
        if isinstance(base_size, tuple):
            self.base_size_h, self.base_size_w = base_size
        else:
            self.base_size_h, self.base_size_w = base_size, base_size
        if isinstance(crop_size, tuple):
            self.crop_size_h, self.crop_size_w = crop_size
        else:
            self.crop_size_h, self.crop_size_w = crop_size, crop_size
        self.fill = fill
        self.isImg = isImg

    def __call__(self, sample):
        # random scale (short edge)
        # short_size_h = random.randint(int(self.base_size_h * 0.5), int(self.base_size_h * 2.0))
        # short_size_w = random.randint(int(self.base_size_w * 0.5), int(self.base_size_w * 2.0))
        short_size_h = random.randint(int(self.base_size_h * 1.0), int(self.base_size_h * 1.0))
        short_size_w = random.randint(int(self.base_size_w * 1.0), int(self.base_size_w * 1.0))
        w, h = sample.size
        if w/self.base_size_w > h/self.base_size_h:
            oh = short_size_h
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size_w
            oh = int(1.0 * h * ow / w)
        sample = sample.resize((ow, oh), Image.BILINEAR if self.isImg else Image.NEAREST)
        # pad crop
        padh, padw = 0, 0
        if oh < self.crop_size_h:
            padh = self.crop_size_h - oh if oh < self.crop_size_h else 0
        if ow < self.crop_size_w:
            padw = self.crop_size_w - ow if ow < self.crop_size_w else 0
        sample = ImageOps.expand(sample, border=(0, 0, padw, padh), fill=(0 if self.isImg else self.fill))
            
        # random crop crop_size
        w, h = sample.size
        x1 = random.randint(0, w - self.crop_size_w)
        y1 = random.randint(0, h - self.crop_size_h)
        sample = sample.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h))

        return sample


class PadCrop(object):
    def __init__(self, base_size, fill=8, isImg=True):
        if isinstance(base_size, tuple):
            self.base_size_h, self.base_size_w = base_size
        else:
            self.base_size_h, self.base_size_w = base_size, base_size
        self.fill = fill
        self.isImg = isImg
    
    def __call__(self, sample):
        # center pad
        w, h = sample.size
        pad_wl, pad_wr = 0, 0
        if self.base_size_w > w:
            pad_wl = int((self.base_size_w - w) / 2)
            pad_wr = self.base_size_w - w - pad_wl
        pad_ht, pad_hd = 0, 0
        if self.base_size_h > h:
            pad_ht = int((self.base_size_h - h) / 2)
            pad_hd = self.base_size_h - h - pad_ht
        sample = ImageOps.expand(sample, border=(pad_wl, pad_ht, pad_wr, pad_hd), fill=(0 if self.isImg else self.fill))
        # center crop
        w, h = sample.size
        x1 = int(round((w - self.base_size_w) / 2.))
        y1 = int(round((h - self.base_size_h) / 2.))
        sample = sample.crop((x1, y1, x1 + self.base_size_w, y1 + self.base_size_h))
        
        return sample

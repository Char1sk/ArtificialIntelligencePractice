import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class ScaleCrop(object):
    def __init__(self, crop_size, isImg):
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


# class ScaleCrop(object):
#     def __init__(self, crop_size):
#         self.crop_size = crop_size

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         w, h = img.size
#         if w > h:
#             oh = self.crop_size
#             ow = int(1.0 * w * oh / h)
#         else:
#             ow = self.crop_size
#             oh = int(1.0 * h * ow / w)
#         img = img.resize((ow, oh), Image.BILINEAR)
#         mask = mask.resize((ow, oh), Image.NEAREST)
#         # center crop
#         w, h = img.size
#         x1 = int(round((w - self.crop_size) / 2.))
#         y1 = int(round((h - self.crop_size) / 2.))
#         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

#         return {'image': img,
#                 'label': mask}

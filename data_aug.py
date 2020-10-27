"""
Demo of Data Augmentation

Author: HearyShen
Date: 2020.10.28
"""
import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation, RandomVerticalFlip

PLT_ROWS = 3
PLT_COLS = 5

default_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

default_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     #  default_normalize
     ])

train_transforms = transforms.Compose(
    [transforms.RandomRotation(180),
     #  transforms.RandomVerticalFlip(),
     #  transforms.RandomHorizontalFlip(),
     transforms.Resize((256, 256)),
     transforms.RandomCrop((224, 224)),
     transforms.ColorJitter(
         brightness=0.2,
         contrast=0.2,
         saturation=0.2,
         hue=0.02
    ),
        transforms.ToTensor(),
        # default_normalize
    ])


def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


# image_root = r'D:\Datasets\Flickr30k\flickr30k_images'
# image_root = r'D:\Datasets\ForestrySecurity\insects98'
# image_root = r'.\imgs\flickr'
image_root = r'.\imgs\insects'


for root, dirs, files in os.walk(image_root):
    for filename in files:
        if filename.split('.')[-1].lower() not in ('jpg', 'png'):
            continue

        image_path = os.path.join(root, filename)
        image = Image.open(image_path).convert('RGB')

        plt.figure(figsize=(16, 9))
        for i in range(PLT_ROWS*PLT_COLS):
            if i == 0:
                image_t = default_transforms(image)
            else:
                image_t = train_transforms(image)

            plt.subplot(PLT_ROWS, PLT_COLS, i+1)
            plt.xticks([])
            plt.yticks([])
            imshow(image_t)
            if i == 0:
                plt.title("original")
        plt.show()

import torch
import torchvision
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import glob
import os

from skimage import transform
import random


emocat = {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'}


def getEmotionImage(folder_path, emotion):
    catemo = {emo: idx for idx, emo in emocat.items()}
    cat_idx = catemo[emotion]
    folder_path = os.path.normpath(folder_path)
    images_path = []
    for f in glob.glob(folder_path+'/**/0'+str(cat_idx+1)+'-*.jpg', recursive=True):
            images_path.append(os.path.normpath(f))
    file_name = random.choice(images_path)
    image = plt.imread(os.path.join(folder_path, file_name))
    return image


def getRandomImage(folder_path):
    folder_path = os.path.normpath(folder_path)
    file_name = random.choice(os.listdir(folder_path))
    image = plt.imread(os.path.join(folder_path, file_name))
    cat = int(file_name.split('-')[0])-1
    return image, cat


class FaceEmotionDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_path = []
        for f in glob.glob(root_dir+'/**/*.jpg', recursive=True):
            self.images_path.append(os.path.normpath(f))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = plt.imread(self.images_path[idx])
        cat = int(os.path.basename(self.images_path[idx]).split('-')[0])-1
        sample = {'image': image, 'cat':cat}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if type(sample) is dict:
            image = sample['image']
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w: new_h, new_w = self.output_size * h / w, self.output_size
                else: new_h, new_w = self.output_size, self.output_size * w / h
            else: new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)
            image = transform.resize(image, (new_h, new_w))
            return {'image': image, 'cat': sample['cat']}
        else:
            h, w = sample.shape[:2]
            if isinstance(self.output_size, int):
                if h > w: new_h, new_w = self.output_size * h / w, self.output_size
                else: new_h, new_w = self.output_size, self.output_size * w / h
            else: new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)
            image = transform.resize(sample, (new_h, new_w))
            return image

        
class CenterCrop(object):
    """Crop the image at the center.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if type(sample) is dict:
            image = sample['image']
            h, w = image.shape[:2]
            new_h, new_w = self.output_size
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            image = image[top: top + new_h,left: left + new_w]
            return {'image': image, 'cat': sample['cat']}
        else:
            h, w = sample.shape[:2]
            new_h, new_w = self.output_size
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            image = sample[top: top + new_h,left: left + new_w]
            return image
           
    
class RandomCrop(CenterCrop):
    
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        super(RandomCrop, self).__init__(output_size)

    def __call__(self, sample):
        if type(sample) is dict:
            image = sample['image']
            h, w = image.shape[:2]
            new_h, new_w = self.output_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            image = image[top: top + new_h,left: left + new_w]
            return {'image': image, 'cat': sample['cat']}
        else:
            h, w = sample.shape[:2]
            new_h, new_w = self.output_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            image = sample[top: top + new_h,left: left + new_w]
            return image

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if type(sample) is dict:
            image = sample['image']

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                    'cat': torch.tensor(sample['cat'])} 
        else:
            image = sample.transpose((2, 0, 1))
            return torch.from_numpy(image)
    
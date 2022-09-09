from __future__ import print_function, division
import os
from numpy.core.fromnumeric import transpose
from skimage import io,transform ,filters
import matplotlib.pyplot as plt
import numpy as np
import glob
import skimage
import torch
from torch.utils.data import Dataset,DataLoader
import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['img'], sample['mask']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = np.transpose(image,(2,0,1))
        label[label < 127] = 0.0
        label[label > 127] = 1.0
        image = torch.from_numpy(image.astype(np.float32)) / 255.0 
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'img': image, 'mask': label.long()}
        return sample


class CrackSegDataset(Dataset):
    
    def __init__(self, partition = "train",transform = None):
       
       self.transform = transform  # using transform in torch!
       self.base_dir = os.path.join(str(os.getcwd()),"datasets")
       self.dataset_dir = os.path.join(self.base_dir,"crack_segmentation_dataset")
       self.mask_dir =  os.path.join(os.path.join(self.dataset_dir,partition),"masks")
       self.img_dir =  os.path.join(os.path.join(self.dataset_dir,partition),"images")
       self.imgs = os.listdir(self.img_dir)
       self.masks = os.listdir(self.mask_dir)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx) :
       
       img = io.imread(os.path.join(self.img_dir,self.imgs[idx]))
       mask = io.imread(os.path.join(self.mask_dir,self.masks[idx]))
       sample = {"img" : img ,"mask" : mask}
       if self.transform:
            sample = self.transform(sample)
       
       return sample

       


if __name__ == "__main__":
    db_train = CrackSegDataset(partition = "train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[448,448])]))
    
    trainloader = DataLoader(db_train, batch_size= 4, shuffle=True, num_workers=8, pin_memory=True)
    
    for sample in trainloader:
        print(sample["img"].shape)
        print(sample["mask"].shape)
 
        
        break
    
  


     
              
              
              


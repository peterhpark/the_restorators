'''Create dataset classes to import the data into the appropriate format.'''
from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class RestoratorsDataset(Dataset, ABC):
    '''A PyTorch dataset to serve as a base class for our custom transformer datasets.'''
    def __init__(self, root_dir):
        assert self.root_dir is not None
        return None
    
    @abstractmethod
    def __len__(self):
        return None
    
    @abstractmethod
    def __getitem__(self, index):
        '''Describes what happens when you index into a Dataset
        Parameters:
            index: int
        Returns: a 2-tuple of 3D tensors of the form (C, H, W)    
        '''
        return None

class BirefringenceDataset(RestoratorsDataset):
    """A PyTorch dataset to load polarized light field images and birefringent objects"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the parent directory of samples
        self.source = os.listdir(os.path.join(self.root_dir, 'images')) # source domain
        self.target = os.listdir(os.path.join(self.root_dir, 'objects')) # target domain
        self.img_transform = self.img_transform  # transformations to apply to raw LF image only
        self.transform = transform  # transformations for augmentations
        #  transformations to apply just to inputs
        self.input_transform = transforms.ToTensor()

    # # get the total number of samples
    # def __len__(self):
    #     return len(self.source)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        source_path = os.path.join(self.root_dir, 'images', self.source[idx])
        # We'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        source = Image.open(source_path)
        source = self.input_transform(source)
        target_path = os.path.join(self.root_dir, 'objects', self.target[idx])
        target = Image.open(target_path)
        target = self.input_transform(target)
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the source and target
            seed = torch.seed()
            torch.manual_seed(seed)
            source = self.transform(source)
            torch.manual_seed(seed)
            target = self.transform(target)
        # if self.img_transform is not None:
        #     source = self.img_transform(source)
        return source, target

    def img_transform(self, image):
        '''Transforms the light field image into a stack of perspective views.'''
        pinhole_stack = image
        # TODO: write function
        # pinhole_stack = transform_into_pinhole_2channels(image)
        return pinhole_stack


class NucleiDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(self, root_dir, transform=None, img_transform=None):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),  # some of the images are RGB
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]), # 0.5 = mean and 0.5 = variance 
            ]
        )

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.samples[idx], "image.tif")
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = Image.open(img_path)
        image = self.inp_transforms(image)
        mask_path = os.path.join(self.root_dir, self.samples[idx], "mask.tif")
        mask = transforms.ToTensor()(Image.open(mask_path))
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask

def show_random_dataset_image(dataset):
    idx = np.random.randint(0,len(dataset),1)[0]
    img, mask = dataset[idx]
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()

def load():
    TRAIN_DATA_PATH = "src/nuclei_train_data"
    train_data = NucleiDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

    VAL_DATA_PATH = "src/nuclei_val_data"
    val_data = NucleiDataset(VAL_DATA_PATH, transforms.RandomCrop(256))
    val_loader = DataLoader(val_data, batch_size=5)
    return train_loader, val_loader

if __name__ == "__main__":
    # train_loader, val_loader = load()
    TRAIN_DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"

    train_data = BirefringenceDataset(TRAIN_DATA_PATH)
    train_data[0]

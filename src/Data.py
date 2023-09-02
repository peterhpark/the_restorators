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
import tifffile

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
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir  # the directory with all the training samples
        self.samples = os.listdir(root_dir)  # list the parent directory of samples
        self.source = os.listdir(os.path.join(self.root_dir, 'images')) # source domain
        self.target = os.listdir(os.path.join(self.root_dir, 'objects')) # target domain
        self.img_transform = self.img_transform  # transformations to apply to raw LF image only
        self.transform = transform  # transformations for augmentations
        #  transformations to apply just to inputs
        # self.input_transform # transforms.ToTensor() only works on smaller dim images
        num_train = int(0.60 * len(self.source)) # reducing this will speed up the epochs when training
        num_test = int(0.16 * len(self.source))
        num_val = int(0.24 * len(self.source))
        if split=='train':
            self.source = self.source[:num_train]
            self.target = self.target[:num_train]
        elif split=='val':
            self.source = self.source[num_train:num_train+num_val]
            self.target = self.target[num_train:num_train+num_val]
        elif split=='test':
            self.source = self.source[num_train+num_val:]
            self.target = self.target[num_train+num_val:]

    # get the total number of samples
    def __len__(self):
        return len(self.source)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        source_path = os.path.join(self.root_dir, 'images', self.source[idx])
        # We'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        source = tifffile.imread(source_path)
        source = self.numpy2tensor(source).to(torch.float32)
        target_path = os.path.join(self.root_dir, 'objects', self.target[idx])
        target = tifffile.imread(target_path)
        target = self.numpy2tensor(target).to(torch.float32)
        return source, target

    def img_transform(self, image):
        '''Transforms, normally in a simple way, the source data.'''
        return image
    
    def numpy2tensor(self, array):
        return torch.from_numpy(array)




class SimpleMonalisaDataset(Dataset):
    """A PyTorch dataset to reconstruted pairs of monalisa data (e.g. for upsampling)"""

    def __init__(self, input_dir, gt_dir, transform=None, input_transform=None,mean_input=None,std_input=None,mean_gt=None,std_gt=None):
        self.input_dir = input_dir  
        self.gt_dir = gt_dir
        self.input_list = sorted(os.listdir(self.input_dir))
        self.gt_list = sorted(os.listdir(self.gt_dir))
        self.transform = (
            transform  # transformations to apply to both inputs and gt
        )
        self.input_transform = input_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs


        if mean_input is None:

            self.avg_input = 0
            self.avg_gt = 0
            self.std_input = 0
            self.std_gt = 0

            for file in self.input_list:
                input_path = os.path.join(self.input_dir,file)
                input = transforms.ToTensor()(Image.open(input_path))
                
                self.avg_input += torch.mean(input)
                self.std_input += torch.std(input)
            self.avg_input = self.avg_input / len(self.input_list)
            self.std_input = self.std_input / len(self.input_list)

            for file in self.gt_list:
                gt_path = os.path.join(self.gt_dir,file)
                gt = transforms.ToTensor()(Image.open(gt_path))
                
                self.avg_gt += torch.mean(gt)
                self.std_gt += torch.std(gt)
            self.avg_gt = self.avg_gt / len(self.gt_list)
            self.std_gt = self.std_gt / len(self.gt_list)

            
        else:
            self.avg_input=mean_input
            self.std_input=std_input
            self.avg_gt=mean_gt
            self.std_gt=std_gt

        print(self.avg_input,self.std_input,self.avg_gt,self.std_gt)

        self.inp_transforms = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    # get the total number of samples
    def __len__(self):
        return len(self.input_list)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir,self.input_list[idx])
        gt_path = os.path.join(self.gt_dir,self.gt_list[idx])

        input = Image.open(input_path)
        input = self.inp_transforms(input)
        input = (input - self.avg_input) / self.std_input

        gt = transforms.ToTensor()(Image.open(gt_path))
        gt = (gt - self.avg_gt) / self.std_gt

        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            input = self.transform(input)
            torch.manual_seed(seed)
            gt = self.transform(gt)
        if self.input_transform is not None:
            input = self.input_transform(input)
        return input, gt


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

    train_data = BirefringenceDataset(TRAIN_DATA_PATH, split='test')
    train_data[0] # pair 0
    train_data[1] # pair 1

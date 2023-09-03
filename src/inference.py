#import libraries
import torch
import os
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from Models import UNet
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision
from tifffile import imsave
from Data import SimpleMonalisaDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

avg_input = 55.0787
std_input = 144.3149
avg_gt = 66.3622
std_gt = 177.3616


input_dir_test = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/input"
gt_dir_test = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/gt"

transform = transforms.CenterCrop(1024)

test_data = SimpleMonalisaDataset(input_dir_test,gt_dir_test, transform=transform, mean_input = avg_input, std_input = std_input, mean_gt = avg_gt, std_gt = std_gt)
test_loader = DataLoader(test_data,batch_size=5)

model = torch.load("model_L2loss.pt")
device = torch.device ("cuda")

for idx, (input, gt) in enumerate(test_loader):
    
    input = input.to(device)
    prediction = model(input)
    
    
    input = input.cpu().numpy()
    gt = gt.cpu().numpy()
    prediction = prediction.cpu().detach().numpy()

    input =input[0,0,...]
    gt = gt[0,0,...]
    prediction = prediction [0,0,...]

    psnr_test = psnr(gt,prediction,data_range=60)
    ssim_test = ssim(gt,prediction,data_range=60)
    
    imsave(f"inference_results/pred{idx:02d}.tiff",prediction)
    imsave(f"inference_results/gt{idx:02d}.tiff",gt)
    imsave(f"inference_results/input{idx:02d}.tiff",input)
        
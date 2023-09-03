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
writer=SummaryWriter()

#import custom method
#from Data import NucleiDataset
from Data import SimpleMonalisaDataset


### simple monalisa data loading ###

#paths for train, val, test

input_dir_train = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/input"
gt_dir_train = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/gt"

input_dir_val = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/val/input"
gt_dir_val = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/val/gt"

input_dir_test = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/input"
gt_dir_test = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/gt"

#
list_transforms = transforms.Compose(
    [
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(degrees=[90,180,270]),
        #torchvision.transforms.ElasticTransform()
        #transforms.TenCrop(256, vertical_flip=False)
    ]
)

#list_transforms = transforms.RandomCrop(256)

avg_input = 55.0787
std_input = 144.3149
avg_gt = 66.3622
std_gt = 177.3616

#creating loaders train,val, test
train_data = SimpleMonalisaDataset(input_dir_train,gt_dir_train,transform=list_transforms)
train_loader = DataLoader(train_data,batch_size=8,shuffle=True)

val_data = SimpleMonalisaDataset(input_dir_val,gt_dir_val, transform=transforms.CenterCrop(256), mean_input = avg_input, std_input = std_input, mean_gt = avg_gt, std_gt = std_gt)
val_loader = DataLoader(val_data,batch_size=5)

test_data = SimpleMonalisaDataset(input_dir_test,gt_dir_test, transform=transforms.CenterCrop(256), mean_input = avg_input, std_input = std_input, mean_gt = avg_gt, std_gt = std_gt)
test_loader = DataLoader(test_data,batch_size=5)


#test data
""" TRAIN_DATA_PATH = "src/nuclei_train_data"
train_data = NucleiDataset(TRAIN_DATA_PATH, transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

VAL_DATA_PATH = "src/nuclei_val_data"
val_data = NucleiDataset(VAL_DATA_PATH, transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5) """

# model parameters
in_channels=1
out_channels=1
depth=5
final_activation=None

#building model
model = UNet(in_channels, out_channels, depth, final_activation=None)
simple_net = UNet(1,1,depth=1,final_activation=torch.nn.Sigmoid())


# training parameters
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
loss_function = torch.nn.MSELoss()
metric = None
n_epochs = 2000


#validate function
def validate(model,
    loader,
    loss_function,
    metric,
    device=None,
    epoch=None
):
            
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model.eval()
    model.to(device)

    # running loss and metric values
    avg_val_loss = 0
    avg_val_metric = 0
    avg_val_psnr = 0
    avg_val_ssim = 0
    
    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # TODO: evaluate this example with the given loss and metric
            prediction = model(x)
            #print(prediction.shape)
            #print(y.shape)
            avg_val_loss += loss_function(prediction,y).item()
            avg_val_psnr += psnr(y[0,0,...].detach().cpu().numpy(),prediction[0,0,...].detach().cpu().numpy(),data_range=60)
            avg_val_ssim+= ssim(y[0,0,...].detach().cpu().numpy(),prediction[0,0,...].detach().cpu().numpy(),data_range=60)
            if metric is not None:
                avg_val_metric += metric(prediction,y).item()

            input_img = x[0,...]
            input_img = (input_img -torch.min(input_img))
            input_img = 255 * input_img / torch.max(input_img)
            input_img = input_img.type(torch.uint8)
            print(input_img.shape)

            gt_img = y[0,...]
            gt_img = (gt_img -torch.min(gt_img))
            gt_img = 255 * gt_img / torch.max(gt_img)
            gt_img = gt_img.type(torch.uint8)
            
            output_img = prediction[0,...]
            output_img = (output_img -torch.min(output_img))
            output_img = 255 * output_img / torch.max(output_img)
            output_img = output_img.type(torch.uint8)
            
            writer.add_image('input', input_img, epoch)
            writer.add_image('prediction', output_img, epoch)
            writer.add_image('ground truth', gt_img, epoch)

        #imsave(f"src/visualize/gt_val_epoch{epoch}.tiff" ,y[0,0,...].detach().cpu().numpy())
        #imsave(f"src/visualize/pred_val_epoch{epoch}.tiff" ,prediction[0,0,...].detach().cpu().numpy())
        

    avg_val_loss = avg_val_loss / len(loader)
    avg_val_psnr = avg_val_psnr / len(loader)
    avg_val_ssim = avg_val_ssim / len(loader)

    #print("Val_loss: ", val_loss, "Val_metric: ", val_metric)

    return avg_val_loss, avg_val_psnr, avg_val_ssim

# train for one epoch function
def train_1epoch(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=1,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    avg_loss = 0
    avg_psnr = 0
    avg_ssim = 0

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # apply model and calculate loss
        prediction = model(x)  # placeholder since we use prediction later
        #print(prediction.shape)
        #print(y.shape)
        loss = loss_function(prediction,y)  # placeholder since we use the loss later
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss
        avg_psnr += psnr(y[0,0,...].detach().cpu().numpy(),prediction[0,0,...].detach().cpu().numpy(),data_range=100)
        avg_ssim += ssim(y[0,0,...].detach().cpu().numpy(),prediction[0,0,...].detach().cpu().numpy(),data_range=100)

        """
        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )
        """
        #print("Epoch: ", epoch, " - batch: ", batch_id, "- loss: ", loss.item())

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break
    avg_loss = avg_loss.item()/len(loader)
    avg_psnr = avg_psnr / len(loader)
    avg_ssim = avg_ssim / len(loader)
    print("avg_loss: ", avg_loss)
    
    return avg_loss, avg_psnr, avg_ssim

# train for n_epochs function
def train_loop(
        n_epochs,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_function,
        metric,
        log_interval=100,
        device=None,
        early_stop=False,
        validate_param=False,
):
    best_val_loss = 100
    for epoch in range(n_epochs):
        avg_loss, avg_psnr, avg_ssim = train_1epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                epoch,
                log_interval=100,
                device=None,
                early_stop=False,
        )
        
        if validate_param:
            avg_val_loss, avg_val_psnr, avg_val_ssim = validate(model, val_loader, loss_function, metric,epoch=epoch)
            if avg_val_loss < best_val_loss:
                print(f"Epoch {epoch} validation loss ({avg_val_loss}) is better than previous ({best_val_loss}) so saving!")
                best_val_loss = avg_val_loss
                torch.save(model, "model_L2_depth5.pt")
            else:
                print(f"Epoch {epoch} validation loss ({avg_val_loss}) is worse than previous ({best_val_loss}) so no saving!")

        writer.add_scalar('Avg_loss_train',avg_loss,epoch)
        writer.add_scalar('Avg_loss_val',avg_val_loss,epoch)
        writer.add_scalar('Avg_psnr_train',avg_psnr,epoch)
        writer.add_scalar('Avg_psnr_val',avg_val_psnr,epoch)
        writer.add_scalar('Avg_ssim_train',avg_ssim,epoch)
        writer.add_scalar('Avg_ssim_val',avg_val_ssim,epoch)
    torch.save(model, "last_epoch.pt")





if __name__ == "__main__":
    assert torch.cuda.is_available()

    print(len(train_data))
    train_loop(
        n_epochs,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_function,
        metric,
        log_interval=100,
        device=None,
        early_stop=False,
        validate_param=True,
)


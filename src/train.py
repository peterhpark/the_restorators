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

#import custom method
from Data import NucleiDataset
from Data import SimpleMonalisaDataset


### simple monalisa data loading ###

#paths for train, val, test

input_dir_train = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/input"
gt_dir_train = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/gt"

input_dir_val = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/val/input"
gt_dir_val = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/val/gt"

input_dir_test = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/input"
gt_dir_test = "/mnt/efs/shared_data/restorators/monalisa_data/Actin_20nmScanStep/train/gt"


list_transforms = transforms.Compose(
    [
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]
)

list_transforms = transforms.RandomCrop(256)

avg_input = 56.2473
std_input = 144.2397
avg_gt = 67.9836
std_gt = 179.4052

#creating loaders train,val, test
train_data = SimpleMonalisaDataset(input_dir_train,gt_dir_train,list_transforms)
train_loader = DataLoader(train_data,batch_size=5,shuffle=True)

val_data = SimpleMonalisaDataset(input_dir_val,gt_dir_val, mean_input = avg_input, std_input = std_input, mean_gt = avg_gt, std_gt = std_gt)
val_loader = DataLoader(train_data,batch_size=5)

test_data = SimpleMonalisaDataset(input_dir_test,gt_dir_test, mean_input = avg_input, std_input = std_input, mean_gt = avg_gt, std_gt = std_gt)
test_loader = DataLoader(train_data,batch_size=5)


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
depth=4
final_activation=None

#building model
model = UNet(in_channels, out_channels, depth, final_activation=None)
simple_net = UNet(1,1,depth=1,final_activation=torch.nn.Sigmoid())


# training parameters
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.MSELoss()
metric = None
n_epochs = 100


#validate function
def validate(model,
    loader,
    loss_function,
    metric,
    device=None,
):
    if metric is None:
        print("WARNING: NO METRIC FOR VALIDATION")
        
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # TODO: evaluate this example with the given loss and metric
            prediction = model(x)
            val_loss += loss_function(prediction,y).item()
            if metric is not None:
                val_metric += metric(prediction,y).item()

    val_loss = val_loss / len(loader)

    print("Val_loss: ", val_loss, "Val_metric: ", val_metric)

    return val_loss

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

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # apply model and calculate loss
        prediction = model(x)  # placeholder since we use prediction later
        loss = loss_function(prediction,y)  # placeholder since we use the loss later
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss

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
        print("Epoch: ", epoch, " - batch: ", batch_id, "- loss: ", loss.item())

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break
    avg_loss = avg_loss.item()/len(loader)
    print("avg_loss: ", avg_loss)
    
    return avg_loss

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
        avg_loss = train_1epoch(
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
            val_loss = validate(model, val_loader, loss_function, metric)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, "model.pt")




if __name__ == "__main__":
    assert torch.cuda.is_available()


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


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


#data
TRAIN_DATA_PATH = "src/nuclei_train_data"
train_data = NucleiDataset(TRAIN_DATA_PATH, transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

VAL_DATA_PATH = "src/nuclei_val_data"
val_data = NucleiDataset(VAL_DATA_PATH, transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=5)

# model parameters
in_channels=1
out_channels=1
depth=2
final_activation = None
final_activation=torch.nn.ReLU()

#building model
model = UNet(in_channels, out_channels, depth, final_activation)
simple_net = UNet(1,1,depth=1,final_activation=torch.nn.Sigmoid())


# training parameters
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.MSELoss()
metric = None
n_epochs = 1


#validate function
def validate():
    #TODO
    return

# train for one epoch function
def train_1epoch(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
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

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


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
    
    for epoch in range(n_epochs):
        train_1epoch(
            model,
            train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch=epoch,
            log_interval=log_interval,
        )
        
        if validate_param:
            validate(model, val_loader, loss_function, metric)



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
        validate_param=False,
)

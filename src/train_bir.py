'''Training script to train the model defined in model_bir.py'''
import os
import numpy as np
import torch
from model_bir import BirNetwork
from Data import BirefringenceDataset
import torch.nn as nn
# import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# load data
DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
train_data = BirefringenceDataset(DATA_PATH, split='train')
val_data = BirefringenceDataset(DATA_PATH, split='val')
test_data = BirefringenceDataset(DATA_PATH, split='test')
batch_size = 1
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device")

# establish the model
net = BirNetwork().to(device)
print(summary(net, (512, 16, 16)))
print("model instantiated")
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train_one_epoch(network, trainloader, loss_function, optimizer, tfwriter, epoch):
    '''This function updates the weights of the network as it loops
    through all the data in the dataset.'''
    running_loss = 0.0
    running_loss_per_epoch = 0.0
    for i, data in enumerate(trainloader, start=0):
        network.train()
        source, target = data
        source = source.to(device)
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(source)
        train_loss = loss_function(outputs, target)
        train_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += train_loss.item()
        running_loss_per_epoch += train_loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            tfwriter.add_scalar('Loss/train', running_loss / 50, i)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            running_loss = 0.0
            
        data_idx = i
    avg_sum_per_epoch = running_loss_per_epoch/data_idx
    return network, avg_sum_per_epoch

def validate(network, valloader, loss_function, optimizer, tfwriter, epoch):
    '''This function validates network parameter optimizations'''
    running_loss = 0.0
    val_loss_per_batch = []
    network.eval()
    print('validating...')
    #  iterating through batches
    for i, data in enumerate(valloader, start=0):
        source, target = data
        source = source.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        output = network(source)
        val_loss = loss_function(output, target)
        val_loss_per_batch.append(val_loss.item())
        # print statistics
        running_loss += val_loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            tfwriter.add_scalar('Loss/eval', running_loss / 50, i)
            print(f'[{epoch + 1}, {i + 1:5d}] val loss: {running_loss:.3f}')
            running_loss = 0.0
    print('all done!')

    return val_loss_per_batch

if __name__ == '__main__':
    writer = SummaryWriter('runs/spheres')
    # to view training results: tensorboard --logdir runs
    min_val_loss = 1000
    for epoch in range(300):  # loop over the dataset multiple times
        print(f"starting training epoch {epoch}")
        trained_net, train_loss_per_batch = train_one_epoch(net, trainloader, criterion, optimizer, writer, epoch)
        writer.add_scalar('Loss/train per epoch', train_loss_per_batch, epoch)

        # validate after each epoch
        val_loss_per_batch = validate(trained_net, valloader, criterion, optimizer, writer, epoch)
        writer.add_scalar('Loss/validate per epoch', np.mean(val_loss_per_batch), epoch)
        
        if np.mean(val_loss_per_batch) < min_val_loss:
            # save model
            save_dir = "/mnt/efs/shared_data/restorators/models_bir/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_filename = save_dir + f'sphere_9_3_epoch{epoch}.pt'
            torch.save(trained_net.state_dict(), model_filename)
            print(f'saved model as {model_filename}')
            min_val_loss = np.mean(val_loss_per_batch)
        
    writer.close()
    print('finished training')

    # save model
    save_dir = "/mnt/efs/shared_data/restorators/models_bir/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_filename = save_dir + 'sphere_9_3_final.pt'
    torch.save(trained_net.state_dict(), model_filename)
    print(f'saved model as {model_filename}')

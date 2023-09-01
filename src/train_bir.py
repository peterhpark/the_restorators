'''Training script to train the model defined in model_bir.py'''
import torch
from model_bir import BirNetwork
from Data import BirefringenceDataset
import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary

TRAIN_DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
train_data = BirefringenceDataset(TRAIN_DATA_PATH, split='train')
test_data = BirefringenceDataset(TRAIN_DATA_PATH, split='test')

batch_size = 1
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

import torch.optim as optim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

net = BirNetwork().to(device)
# print(summary(net, (512, 16, 16)))
print("model intanciated")
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/spheres_experiment_1')

for epoch in range(2):  # loop over the dataset multiple times
    print("starting training")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        source, target = data
        source = source.to(device)
        target = target.to(device)
        # source = source.to(torch.float32)
        # target = target.to(torch.float32)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(source)
        loss = criterion(outputs, target)
        print(loss)
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss train', loss.item())
        # print statistics
        running_loss += loss.item()
        if i % 5 == 0:    # print every 50 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    # eval after each epoch
print('Finished Training')

# save model
# new script to test the trained model
'''Model script where the network architecture is defined for the
polarized light field images. Training script is train_bir.py'''
import torch
from torch import nn
from Data import BirefringenceDataset

class BirNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        conv1_output_channels = 1
        conv1_output_dim = 10
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, conv1_output_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(conv1_output_channels, conv1_output_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(conv1_output_channels, conv1_output_channels, kernel_size=3),
            nn.ReLU(),
        )
        # after the convolutions, the HxW will shrink
        volume_shape = 4 * 8 * 32 * 32
        self.fully_connected = nn.Sequential(
            # self.flatten,
            nn.Linear(conv1_output_dim*conv1_output_dim*conv1_output_channels, volume_shape),
            nn.ReLU(),
            # nn.Linear(volume_shape, volume_shape),
            # nn.ReLU()
        )
        # convolutions layers within volume domain
        self.conv2a = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=1),
            nn.ReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=1),
            nn.ReLU(),
        )
        self.conv_final = nn.Conv3d(4, 4, kernel_size=1)

    def forward(self, x):
        # x = self.flatten(x)
        batch_size = x.shape[0]
        step1 = self.conv1(x)
        step1 = self.flatten(step1)
        step2 = self.fully_connected(step1)
        step3 = step2.view(batch_size, 4, 8, 32, 32)
        step3 = self.conv2a(step3)
        step4 = self.conv2b(step3)
        # add a skip connection
        step5 = self.conv_final(step3 + step4)
        output = step5
        # output = step3
        return output
    
if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = BirNetwork().to(device)
    print(model)

    TRAIN_DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
    train_data = BirefringenceDataset(TRAIN_DATA_PATH, split='test')
    X = train_data[0][0].to(device).to(torch.float32).unsqueeze(dim=0)
    y_pred = model(X)
    print(f"Predicted values: {y_pred}")

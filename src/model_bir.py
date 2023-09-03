'''Model script where the network architecture is defined for the
polarized light field images. Training script is train_bir.py'''
import torch
from torch import nn
from torchsummary import summary
from Data import BirefringenceDataset

class BirNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        conv1_out_chs = 64
        # after the convolutions, the HxW will shrink
        conv1_out_dim = 10 # calc from kernel size
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, conv1_out_chs, kernel_size=3),
            nn.LeakyReLU(),
        )
        linear_input_size = conv1_out_dim * conv1_out_dim * conv1_out_chs
        target_size = 4 * 8 * 32 * 32
        combat_conv_size = 4 ** 3
        target_size_expand = 4 * (8+6) * (32+6) * (32+6)
        # target_size_expand = target_size * combat_conv_size
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_input_size, target_size_expand),
            nn.LeakyReLU(),
            # nn.Linear(target_size_expand, target_size_expand),
            # nn.ReLU()
        )
        # convolutions layers within target domain
        self.conv2a = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
        )
        self.conv2b = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
        )
        self.conv_final = nn.Conv3d(4, 4, kernel_size=3)

    def forward(self, x):
        batch_size = x.shape[0]
        step1 = self.conv1(x)
        step1 = self.flatten(step1)
        step2 = self.fully_connected(step1)
        step3 = step2.view(batch_size, 4, 8+6, 32+6, 32+6)
        # step3 = step2.view(batch_size, 4, 8, 32, 32)
        step3 = self.conv2a(step3)
        step4 = self.conv2b(step3)
        # add a skip connection
        step3_crop = step3[:, :, 1:-1, 1:-1, 1:-1]
        step5 = self.conv_final(step3_crop + step4)
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
    print(summary(model, (512, 16, 16)))

    TRAIN_DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
    train_data = BirefringenceDataset(TRAIN_DATA_PATH, split='test')
    X = train_data[0][0].to(device).to(torch.float32).unsqueeze(dim=0)
    y_pred = model(X)
    print(f"Predicted values: {y_pred}")

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
            nn.Conv2d(conv1_output_channels, conv1_output_channels, kernel_size=3),
            nn.Conv2d(conv1_output_channels, conv1_output_channels, kernel_size=3),
        )
        # 14 by 14
        volume_shape = 4 * 8 * 32 * 32
        self.fully_connected = nn.Sequential(
            # self.flatten,
            nn.Linear(conv1_output_dim*conv1_output_dim*conv1_output_channels, volume_shape),
            nn.ReLU(),
            # nn.Linear(volume_shape, volume_shape),
            # nn.ReLU()
        )
        self.conv2 = nn.Conv3d(4, 4, kernel_size=1)

    def forward(self, x):
        # x = self.flatten(x)
        batch_size = x.shape[0]
        step1 = self.conv1(x)
        step1 = self.flatten(step1)
        step2 = self.fully_connected(step1)
        step3 = step2.view(batch_size, 4, 8, 32, 32)
        step3 = self.conv2(step3)
        output = step3
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
    # X = torch.rand(1, 28, 28, device=device)

    y_pred = model(X)
    x = 5
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    print(f"Predicted values: {y_pred}")
'''Script to test a trained model on the set of birefringence data'''
import torch
from Data import BirefringenceDataset
from model_bir import BirNetwork

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device")

saved_model_dir = "/mnt/efs/shared_data/restorators/models_bir/"

DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
test_data = BirefringenceDataset(DATA_PATH, split='test')
testloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                         shuffle=False, num_workers=2)

model = BirNetwork().to(device)
model.eval()
checkpoint = torch.load(saved_model_dir + 'sphere128.pt')
model.load_state_dict(checkpoint)

data_pair = test_data[0]
source = data_pair[0]
source = source.unsqueeze(axis=0).to(device)

with torch.no_grad():
    target_pred = model(source)

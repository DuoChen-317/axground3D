from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
from ignite.engine import Engine
from ignite.metrics import FID
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.clustering import *
from ignite.metrics.regression import *
from ignite.utils import *
import torch.nn as nn
from pytorch_fid.inception import InceptionV3

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# wrapper class as feature_extractor
class WrapperInceptionV3(nn.Module):

    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y

def calculate_fid(real_path="frames/real", fake_path="frames/fake"):
    
    # check if the paths exist and images are present
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        raise FileNotFoundError("Real or fake image directory does not exist.")
    if not os.listdir(real_path) or not os.listdir(fake_path):
        raise FileNotFoundError("Real or fake image directory is empty.")
    
    manual_seed(12345)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    real_ds = ImageFolderDataset(real_path, transform)
    fake_ds = ImageFolderDataset(fake_path, transform)

    real_loader = DataLoader(real_ds, batch_size=32, shuffle=False, num_workers=4)
    fake_loader = DataLoader(fake_ds, batch_size=32, shuffle=False, num_workers=4)

    device = "cpu"

    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    def eval_step(engine, batch):
        fake_batch, real_batch = batch
        return fake_batch, real_batch
    
    # wrapper model to pytorch_fid model
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.eval()

    evaluator = Engine(eval_step)

    fid_metric = FID(num_features=2048, device=device,feature_extractor=wrapper_model)
    fid_metric.attach(evaluator, "fid")  

    print("Calculating FID...")
    zipped_loader = zip(fake_loader, real_loader)
    
    evaluator.run(zipped_loader)

    fid_value = evaluator.state.metrics["fid"]
    print(f"FID between real and fake: {fid_value:.4f}")
    return fid_value

if __name__ == "__main__":

    calculate_fid()
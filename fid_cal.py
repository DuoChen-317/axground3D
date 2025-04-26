from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os

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


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

real_ds = ImageFolderDataset("/path/to/real_images", transform)
fake_ds = ImageFolderDataset("/path/to/fake_images", transform)

real_loader = DataLoader(real_ds, batch_size=32, shuffle=False, num_workers=4)
fake_loader = DataLoader(fake_ds, batch_size=32, shuffle=False, num_workers=4)



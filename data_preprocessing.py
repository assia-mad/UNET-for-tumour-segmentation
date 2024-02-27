from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class BreastCancerDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_dir) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
        self.labels = [os.path.join(dp, f) for dp, dn, filenames in os.walk(labels_dir) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = self.labels[idx]
        image = Image.open(img_name).convert('RGB')  # Convert to grayscale
        label = Image.open(label_name).convert('L')
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

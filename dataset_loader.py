from torchvision import transforms
from .data_preprocessing import BreastCancerDataset
from torch.utils.data import DataLoader

images_dir = '/kaggle/input/breastdm/BC dataset/seg/train/images'
labels_dir = '/kaggle/input/breastdm/BC dataset/seg/train/labels'
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])
seg_dataset = BreastCancerDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)
seg_loader = DataLoader(seg_dataset, batch_size=1, shuffle=True)
images_dir_val = '/kaggle/input/breastdm/BC dataset/seg/val/images' 
labels_dir_val = '/kaggle/input/breastdm/BC dataset/seg/val/labels'
seg_dataset_val = BreastCancerDataset(images_dir=images_dir_val, labels_dir=labels_dir_val, transform=transform)
seg_loader_val = DataLoader(seg_dataset_val, batch_size=1, shuffle=True)

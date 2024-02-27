import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from data_preprocessing import BreastCancerDataset  
from model import UNET  
from train import train  
from validate import validate
from torchvision import transforms 

def main(epochs, learning_rate, batch_size):
    print('we staaaaaaarted')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader setup
    train_images_dir = 'C:/Users/windows/Desktop/Datasets/BC dataset/seg/train/images'
    train_labels_dir = 'C:/Users/windows/Desktop/Datasets/BC dataset/seg/train/labels'
    val_images_dir = 'C:/Users/windows/Desktop/Datasets/BC dataset/seg/val/images' 
    val_labels_dir = 'C:/Users/windows/Desktop/Datasets/BC dataset/seg/val/labels'

    train_dataset = BreastCancerDataset(images_dir=train_images_dir, labels_dir=train_labels_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = BreastCancerDataset(images_dir=val_images_dir, labels_dir=val_labels_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    inp = torch.rand(1,3,96,96)
    model = UNET(inp.shape)  
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Validation loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(model, train_loader, optimizer, device, epoch)
        # validate(model, val_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for Breast Cancer Segmentation')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    args = parser.parse_args()

    main(args.epochs, args.lr, args.batch_size)

import torch
import matplotlib.pyplot as plt
from loss import dice_loss

def validate(model, seg_loader_val, device):
    val_losses = []  

    model.eval()  
    with torch.no_grad():  
        val_loss = 0
        for images, labels in seg_loader_val:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            output = torch.sigmoid(output) 

            dice_loss_value = dice_loss(output, labels)
            val_loss += dice_loss_value.item()

        mean_val_loss = val_loss / len(seg_loader_val)
        val_losses.append(mean_val_loss)

        print(f"Validation Loss: {mean_val_loss:.4f}")

    plot_validation_results(images, output, labels)

    return val_losses

def plot_validation_results(images, output, labels):
    images_np = images.squeeze().cpu().numpy()
    output_np = output.squeeze().detach().cpu().numpy()
    output_np = (output_np > 0.5).astype(float)  
    labels_np = labels.squeeze().cpu().numpy()


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].imshow(images_np[0], cmap='gray')  
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(output_np, cmap='gray')  
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    axes[2].imshow(labels_np, cmap='gray')
    axes[2].set_title('True Label Mask')
    axes[2].axis('off')

    plt.show()

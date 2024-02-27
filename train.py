import datetime
import torch
from loss import dice_loss

def train(model, seg_loader, optimizer, device, epochs=11):
    trlosses = []  

    for epoch in range(epochs):
        print('hi training here')
        startepoch = datetime.datetime.now()
        training_loss = 0
        model.train()  

        for images, labels in seg_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() 
            output = model(images)
            output = torch.sigmoid(output) 

            dice_loss_value = dice_loss(output, labels)
            # print("loss",dice_loss_value)
            dice_loss_value.backward()
            optimizer.step()

            training_loss += dice_loss_value.item()

        mean_tloss = training_loss / len(seg_loader)
        trlosses.append(mean_tloss)

        print(f"Epoch: {epoch + 1} ... Training Loss: {trlosses[-1]:.4f} ...")

        endepoch = datetime.datetime.now()
        print("Epoch time:", str(endepoch - startepoch), "\n")

    return trlosses

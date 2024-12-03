import torch
from torch.utils.data import DataLoader

def train_model(model, train_dataset, device, num_epochs=3, lr=0.001):
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            # Move to GPU
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
        # Save weights at the end of each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        print(f"Model weights saved for epoch {epoch+1}!")

        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete!")




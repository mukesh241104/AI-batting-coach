import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataloader import CricketShotDataset
from i3d_model import Simple3DCNN

# Settings
EPOCHS = 10
BATCH_SIZE = 2
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Dataset
dataset = CricketShotDataset("frames_dataset", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = Simple3DCNN(num_classes=len(dataset.label_map)).to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    running_loss = 0.0
    for clips, labels in loader:
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(loader):.4f}")

# ✅ Save the model
torch.save(model.state_dict(), "shot_classifier.pth")
print("Model saved as shot_classifier.pth")

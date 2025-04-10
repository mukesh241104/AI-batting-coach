import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from i3d_model import Simple3DCNN
from dataloader import CricketShotDataset

# Define the video loading function
def load_video(path, transform, max_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = transforms.ToPILImage()(frame)
        frame = transform(pil_img)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from the video.")

    # Pad if not enough frames
    while len(frames) < max_frames:
        frames.append(frames[-1])
    clip = torch.stack(frames)  # [T, C, H, W]
    clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W]
    return clip.unsqueeze(0)  # [1, C, T, H, W]

# Device and transform
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load label map from dataset
dummy_dataset = CricketShotDataset("frames_dataset", transform=transform)
label_map = dummy_dataset.label_map
idx_to_label = {v: k for k, v in label_map.items()}

# Load the trained model
model = Simple3DCNN(num_classes=len(label_map)).to(DEVICE)
model.load_state_dict(torch.load("shot_classifier.pth", map_location=DEVICE))
model.eval()

# Load the test clip (replace "testy.mp4" with your test file)
clip = load_video("testy.mp4", transform)
clip = clip.to(DEVICE)

# Predict
with torch.no_grad():
    output = model(clip)
    _, predicted = torch.max(output, 1)
    predicted_label = idx_to_label[predicted.item()]

print(f"Predicted shot: {predicted_label}")

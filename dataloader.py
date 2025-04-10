# dataloader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CricketShotDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples = []

        shot_types = sorted(os.listdir(root_dir))
        self.label_map = {shot: idx for idx, shot in enumerate(shot_types)}

        for shot in shot_types:
            shot_path = os.path.join(root_dir, shot)
            for clip in os.listdir(shot_path):
                clip_path = os.path.join(shot_path, clip)
                self.samples.append((clip_path, self.label_map[shot]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        frames = []

        for i in range(self.frames_per_clip):
            frame_path = os.path.join(clip_path, f"frame_{i:03d}.jpg")
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        clip = torch.stack(frames)  # Shape: [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W] for 3D CNN

        return clip, label

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = CricketShotDataset("frames_dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for clips, labels in loader:
        print("Batch:", clips.shape, labels)
        break

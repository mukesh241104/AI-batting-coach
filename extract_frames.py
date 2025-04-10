import os
import cv2

# Source: videos | Destination: extracted frames
video_root = "cricket_dataset"
frame_root = "frames_dataset"

# Number of frames to extract per video
TARGET_FRAMES = 16

def extract_frames_from_video(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(1, total_frames // TARGET_FRAMES)
    count, extracted = 0, 0

    while cap.isOpened() and extracted < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(f"{frame_dir}/frame_{extracted:03d}.jpg", frame)
            extracted += 1
        count += 1

    cap.release()

# Loop through videos by class
for shot_type in os.listdir(video_root):
    shot_folder = os.path.join(video_root, shot_type)
    for video_file in os.listdir(shot_folder):
        video_path = os.path.join(shot_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        frame_dir = os.path.join(frame_root, shot_type, video_name)
        extract_frames_from_video(video_path, frame_dir)
        print(f"✅ {shot_type}/{video_name} - Frames extracted")

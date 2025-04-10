import os
import shutil
import stat

base_path = "frames_dataset"
min_required_frames = 16

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

for shot_type in os.listdir(base_path):
    shot_folder = os.path.join(base_path, shot_type)
    for video_folder in os.listdir(shot_folder):
        video_path = os.path.join(shot_folder, video_folder)
        if not os.path.isdir(video_path):
            continue
        frame_files = [f for f in os.listdir(video_path) if f.endswith(".jpg")]
        if len(frame_files) < min_required_frames:
            print(f"Deleting {video_path} — only {len(frame_files)} frames")
            shutil.rmtree(video_path, onerror=on_rm_error)

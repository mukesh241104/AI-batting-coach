import os
import yt_dlp

# Define shot classes and YouTube search queries
shots = {
    "cover_drive": "virat kohli cover drive slow motion",
    "pull_shot": "rohit sharma pull shot slow motion",
    "sweep": "sweep shot cricket slow motion",
    "straight_drive": "sachin tendulkar straight drive slow motion",
    "cut_shot": "cut shot cricket slow motion"
}

# Base directory where videos will be stored
base_dir = "cricket_dataset"

# Common download options
common_opts = {
    'format': 'mp4',
    'noplaylist': True,
    'quiet': False
}

# Loop through each shot type
for shot, query in shots.items():
    print(f"\n📥 Downloading: {shot}")
    folder = os.path.join(base_dir, shot)
    os.makedirs(folder, exist_ok=True)

    # Set download location for this shot type
    ydl_opts = common_opts.copy()
    ydl_opts['outtmpl'] = f'{folder}/%(title).50s.%(ext)s'

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"ytsearch5:{query}"])

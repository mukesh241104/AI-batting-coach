from yt_dlp import YoutubeDL

query = "pull shot cricket"

ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'test_clip.%(ext)s',
    'noplaylist': True,
    'quiet': False,
    'max_downloads': 1,  # important
    'ignoreerrors': True
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([f"ytsearch:{query}"])

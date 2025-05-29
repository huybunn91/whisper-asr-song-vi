import yt_dlp
import ffmpeg
import os
import json

# Tạo thư mục lưu audio và manifest
os.makedirs("/content/drive/MyDrive/ASR music vi", exist_ok=True)
manifest_path = "/content/drive/MyDrive/ASR music vi/train.jsonl"

def download_and_convert_audio(youtube_url, out_dir="/content/drive/MyDrive/ASR music vi/downloads"):
    os.makedirs(out_dir, exist_ok=True)

    # Tải file m4a (không convert luôn)
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': f'{out_dir}/%(id)s.%(ext)s',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info["id"]
        title = info["title"]
        input_path = os.path.join(out_dir, f"{video_id}.m4a")
        output_path = os.path.join(out_dir, f"{video_id}_16k.wav")

        # Chuyển sang 16kHz, mono
        ffmpeg.input(input_path).output(output_path, ar=16000, ac=1).run(overwrite_output=True, quiet=True)

        return output_path, title

def add_entry_to_manifest(audio_path, title, lyrics, manifest_file):
    entry = {"audio_filepath": audio_path, "song": title.strip(), "text": lyrics.strip()}
    with open(manifest_file, 'a', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

while True:
    youtube_url = input("🔗 Nhập link YouTube (hoặc để trống để dừng): ").strip()
    if youtube_url == "":
        print("✅ Đã kết thúc.")
        break

    audio_path, title = download_and_convert_audio(youtube_url)
    print(f"🎵 Đã tải: {title}")
    print(f"📂 File lưu tại: {audio_path}")

    lyrics = input("📝 Nhập lời bài hát (paste full lyrics):\n")
    add_entry_to_manifest(audio_path, title, lyrics, manifest_path)
    print("✅ Đã lưu vào train.jsonl\n")
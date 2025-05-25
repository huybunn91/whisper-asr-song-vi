import os
import json
import re
from tqdm import tqdm

# Hàm normalize text giống như bạn đã dùng
def normalize_text(text):
    """
    Normalize text by removing punctuation, converting to lowercase, and stripping extra spaces.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Đường dẫn tới thư mục chính chứa các thư mục con
data_dir = "/content/drive/MyDrive/ASR music vi/data_cleaned"
output_json_path = "/content/drive/MyDrive/ASR music vi/dataset_cleaned.json"

# Khởi tạo danh sách để lưu dữ liệu
dataset = []

# Hàm đọc transcript từ file JSON
def load_transcript(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Duyệt qua các thư mục con
print("Đang tạo dataset_cleaned.json...")
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

for folder_name in tqdm(folders, desc="Xử lý các thư mục"):
    folder_path = os.path.join(data_dir, folder_name)

    # Đường dẫn file JSON transcript
    json_path = os.path.join(folder_path, f"{folder_name}_lyric.json")
    if not os.path.exists(json_path):
        print(f"Không tìm thấy file transcript: {json_path}")
        continue

    # Load transcript
    transcripts = load_transcript(json_path)

    # Xử lý từng file .wav
    for key, text in transcripts.items():
        wav_path = os.path.join(folder_path, f"{key}.wav")
        if not os.path.exists(wav_path):
            print(f"Không tìm thấy file audio: {wav_path}")
            continue

        # Tạo một mục trong dataset
        item = {
            "audio_filepath": wav_path,
            "text": text,  # Sử dụng văn bản gốc thay vì normalized
            "title": folder_name,  # Sử dụng tên thư mục làm title
            "id": key  # Sử dụng key làm id
        }

        dataset.append(item)

# Lưu dataset vào file JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Đã tạo dataset với {len(dataset)} mẫu và lưu vào: {output_json_path}")

# Hiển thị một số thống kê về dataset
total_duration = 0  # Sẽ cần tính toán thời lượng nếu có thông tin

print("\nPhân bố số lượng mẫu theo bài hát:")
titles_count = {}
for item in dataset:
    title = item["title"]
    if title in titles_count:
        titles_count[title] += 1
    else:
        titles_count[title] = 1

for title, count in titles_count.items():
    print(f"  - {title}: {count} đoạn")

print(f"\nTổng số bài hát: {len(titles_count)}")
print(f"Tổng số đoạn audio: {len(dataset)}")

# Kiểm tra một vài mẫu
print("\nMột vài mẫu trong dataset:")
for i in range(min(3, len(dataset))):
    print(f"\nMẫu {i+1}:")
    print(f"  - Audio: {os.path.basename(dataset[i]['audio_filepath'])}")
    print(f"  - Text: {dataset[i]['text'][:100]}..." if len(dataset[i]['text']) > 100 else dataset[i]['text'])
    print(f"  - Title: {dataset[i]['title']}")
    print(f"  - ID: {dataset[i]['id']}")
"""
Check xem nếu folder đấy đã có trong thư mục data_cleaned r thì chuyển sang folder khác
Check xem trong file lyric có file nào lyric là rỗng thì bỏ qua không chuyển
"""

import os
import json
import re
import unicodedata
import shutil

# Đường dẫn thư mục chứa dữ liệu gốc
source_dir = '/content/drive/MyDrive/ASR music vi/data_chunk'
# Đường dẫn thư mục đích để lưu dữ liệu đã được chuẩn hóa
target_dir = '/content/drive/MyDrive/ASR music vi/data_cleaned'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(target_dir, exist_ok=True)

def normalize_text(text):
    """Hàm chuẩn hóa văn bản"""
    # Đưa về chữ thường
    text = text.lower()

    # Chuẩn hóa Unicode
    text = unicodedata.normalize('NFC', text)

    # Xóa các ký tự đặc biệt không cần thiết (giữ lại dấu câu và khoảng trắng cơ bản)
    text = re.sub(r'[^\w\s.,!?;:"-]', '', text)

    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)

    # Cắt khoảng trắng đầu và cuối
    text = text.strip()

    return text

def process_folder(folder_name):
    """Xử lý một folder và chuẩn hóa lyrics"""
    source_folder = os.path.join(source_dir, folder_name)
    target_folder = os.path.join(target_dir, folder_name)

    # Kiểm tra xem folder đã tồn tại trong thư mục đích chưa
    if os.path.exists(target_folder):
        print(f"Folder {folder_name} đã tồn tại trong {target_dir}, bỏ qua...")
        return False, 0

    # Tìm file lyrics trong folder nguồn
    lyrics_file = os.path.join(source_folder, f"{folder_name}_lyric.json")

    if not os.path.exists(lyrics_file):
        print(f"Không tìm thấy file lyrics trong folder {folder_name}, bỏ qua...")
        return False, 0

    # Đọc file lyrics
    try:
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            lyrics_data = json.load(f)

        # Chuẩn hóa lyrics và bỏ qua những file có lyric rỗng
        normalized_lyrics = {}
        files_with_lyrics = []
        empty_lyric_count = 0

        for key, text in lyrics_data.items():
            normalized_text = normalize_text(text)
            if normalized_text and normalized_text.strip():  # Kiểm tra nếu lyric không rỗng sau khi chuẩn hóa
                normalized_lyrics[key] = normalized_text
                files_with_lyrics.append(key)
            else:
                empty_lyric_count += 1

        # Kiểm tra nếu không có lyric nào hợp lệ
        if not normalized_lyrics:
            print(f"Không có lyric hợp lệ nào trong folder {folder_name}, bỏ qua...")
            return False, empty_lyric_count

        # Tạo folder đích
        os.makedirs(target_folder, exist_ok=True)

        # Ghi lyrics đã chuẩn hóa vào thư mục đích
        target_lyrics_file = os.path.join(target_folder, f"{folder_name}_lyric.json")
        with open(target_lyrics_file, 'w', encoding='utf-8') as f:
            json.dump(normalized_lyrics, f, ensure_ascii=False, indent=4)

        # Copy chỉ những file WAV có lyric không rỗng sang thư mục đích
        copied_files = 0
        for file in os.listdir(source_folder):
            if file.endswith('.wav'):
                file_base_name = os.path.splitext(file)[0]
                if file_base_name in files_with_lyrics:
                    source_file = os.path.join(source_folder, file)
                    target_file = os.path.join(target_folder, file)
                    shutil.copy2(source_file, target_file)
                    copied_files += 1

        print(f"Đã xử lý và chuẩn hóa lyrics cho folder {folder_name}")
        print(f"  - Đã bỏ qua {empty_lyric_count} file có lyric rỗng")
        print(f"  - Đã copy {copied_files} file WAV có lyric hợp lệ")

        return True, empty_lyric_count

    except Exception as e:
        print(f"Lỗi khi xử lý folder {folder_name}: {str(e)}")
        # Nếu thư mục đích đã được tạo nhưng gặp lỗi, xóa nó đi
        if os.path.exists(target_folder):
            try:
                shutil.rmtree(target_folder)
            except:
                pass
        return False, 0

# Xử lý tất cả các folder trong thư mục nguồn
processed_count = 0
skipped_count = 0
empty_lyric_total = 0

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)

    # Kiểm tra nếu đây là thư mục
    if os.path.isdir(folder_path):
        print(f"Đang xử lý folder: {folder_name}")
        result, empty_count = process_folder(folder_name)

        if result:
            processed_count += 1
        else:
            skipped_count += 1

        empty_lyric_total += empty_count

print(f"\nTổng kết:")
print(f"- Số folder đã xử lý: {processed_count}")
print(f"- Số folder đã bỏ qua: {skipped_count}")
print(f"- Tổng số file có lyric rỗng đã bỏ qua: {empty_lyric_total}")
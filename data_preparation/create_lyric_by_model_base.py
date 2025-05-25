import os
import librosa
import torch
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Thư mục gốc chứa các folder con đã chia đoạn audio
base_dir = '/content/drive/MyDrive/ASR music vi/data_chunk'

# Khởi tạo model Whisper
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name, language="vi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Duyệt qua từng folder trong thư mục gốc
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    # Kiểm tra nếu đây là folder
    if os.path.isdir(folder_path):
        # Đường dẫn file lyrics
        lyrics_file = os.path.join(folder_path, f"{folder_name}_lyric.json")

        # Kiểm tra file lyrics đã tồn tại chưa
        if os.path.exists(lyrics_file):
            print(f"File lyrics đã tồn tại: {lyrics_file}, bỏ qua...")
            continue

        # Khởi tạo dictionary để lưu kết quả
        results = {}

        print(f"Đang xử lý folder: {folder_name}")

        # Duyệt qua tất cả các file wav trong folder
        wav_files_found = False
        for audio in os.listdir(folder_path):
            # Chỉ xử lý các file có đuôi .wav
            if audio.endswith('.wav'):
                wav_files_found = True
                # Tạo đường dẫn đầy đủ đến file âm thanh
                audio_path = os.path.join(folder_path, audio)

                # Lấy tên file không bao gồm phần mở rộng .wav
                file_name = os.path.splitext(audio)[0]

                print(f"  Đang xử lý file: {audio}")

                try:
                    # Load audio file
                    waveform, sample_rate = librosa.load(audio_path, sr=16000)

                    # Process audio
                    input_features = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features

                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = model.generate(input_features)

                    # Decode the transcription
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                    # Thêm kết quả vào dictionary
                    results[file_name] = transcription

                except Exception as e:
                    print(f"  Lỗi khi xử lý file {audio}: {str(e)}")

        # Kiểm tra xem có file wav nào được xử lý không
        if not wav_files_found:
            print(f"  Không tìm thấy file wav nào trong folder {folder_name}")
            continue

        # Ghi kết quả vào file JSON
        with open(lyrics_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Đã tạo file lyrics với {len(results)} đoạn: {lyrics_file}")
import os
from pydub import AudioSegment

# Đường dẫn thư mục chứa các file âm thanh gốc
input_dir = '/content/drive/MyDrive/ASR music vi/downloads'
# Đường dẫn thư mục lưu các đoạn âm thanh
output_base_dir = '/content/drive/MyDrive/ASR music vi/data_chunk'

# Thời gian chia nhỏ (30 giây)
segment_duration = 30 * 1000  # 30 giây

# Lặp qua tất cả các file âm thanh trong thư mục input
for audio_file in os.listdir(input_dir):
    # Chỉ xử lý các file âm thanh (ví dụ: .wav, .m4a)
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(input_dir, audio_file)

        # Tên thư mục con để lưu các đoạn âm thanh (dựa trên tên file gốc)
        base_name = os.path.splitext(audio_file)[0]
        output_dir = os.path.join(output_base_dir, base_name)

        # Kiểm tra xem thư mục đã tồn tại chưa
        if os.path.exists(output_dir):
            print(f"Thư mục {output_dir} đã tồn tại, bỏ qua file {audio_file}")
            continue

        # Nếu chưa tồn tại, tạo thư mục và xử lý file
        os.makedirs(output_dir, exist_ok=True)
        print(f"Đang xử lý file {audio_file}...")

        # Đọc file âm thanh
        audio = AudioSegment.from_file(audio_path)

        # Tính số đoạn cần chia
        num_segments = len(audio) // segment_duration + 1

        # Chia và lưu từng đoạn
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min(start_time + segment_duration, len(audio))  # Không vượt quá độ dài file
            segment = audio[start_time:end_time]

            # Chuyển sang mono nếu cần
            if segment.channels > 1:
                segment = segment.set_channels(1)

            # Đường dẫn file lưu từng đoạn
            output_path = os.path.join(output_dir, f"{base_name}_{i}.wav")

            # Lưu đoạn âm thanh
            segment.export(output_path, format="wav")
            print(f"Đã lưu đoạn {i}: {output_path}")
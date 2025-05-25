from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio

# Tải processor và model đã fine-tune từ Hugging Face Hub
model_id = "huybunn/whisper-small-vietnamese-lyrics-transcription"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.eval()

# Tải và chuẩn hóa âm thanh (mono, 16kHz)
audio_path = "/content/drive/MyDrive/ASR music vi/downloads/0HZ9UO7pLfo_16k.wav"
waveform, sr = torchaudio.load(audio_path)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)

# Nếu stereo, chọn 1 kênh
if waveform.shape[0] > 1:
    waveform = waveform[0:1, :]

# Các thông số chia đoạn
segment_length_sec = 30  # 30 giây
segment_samples = segment_length_sec * 16000
total_samples = waveform.shape[1]
segments = []

# Chia waveform thành các đoạn 30s
for start in range(0, total_samples, segment_samples):
    end = min(start + segment_samples, total_samples)
    segment = waveform[:, start:end]
    segments.append(segment)

# Nhận diện từng đoạn và ghép lại
final_transcript = ""
for i, segment in enumerate(segments):
    # Nếu đoạn cuối ngắn hơn 30s, không padding cũng được
    inputs = processor(segment.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            num_beams=5,
            length_penalty=1.0
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    final_transcript += f"{transcription} "  # hoặc thêm dấu xuống dòng nếu muốn

print(final_transcript.strip())
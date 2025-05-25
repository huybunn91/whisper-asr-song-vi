from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio

# Tải processor và model đã fine-tune từ Hugging Face Hub
model_id = "huybunn/whisper-small-vietnamese-lyrics-transcription"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.eval()

# Tải và chuẩn hóa âm thanh (mono, 16kHz)
audio_path = "/content/drive/MyDrive/ASR music vi/data_cleaned/717blSmq7s8_16k/717blSmq7s8_16k_6.wav"
waveform, sr = torchaudio.load(audio_path)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)

# Nếu stereo, chọn 1 kênh
if waveform.shape[0] > 1:
    waveform = waveform[0:1, :]

# Tiền xử lý
inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

# Inference với beam search
with torch.no_grad():
    predicted_ids = model.generate(
        inputs["input_features"],
        num_beams=5,
        length_penalty=1.0
    )

# Giải mã
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)

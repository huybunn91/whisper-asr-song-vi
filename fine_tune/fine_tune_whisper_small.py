import torch
import json
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import soundfile as sf
import numpy as np
from datasets import Dataset, DatasetDict
from jiwer import wer as calculate_wer

# 1. Tải mô hình và processor Whisper
MODEL_PATH = "openai/whisper-small"  # Sử dụng whisper-small
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# Cấu hình cho tiếng Việt
processor.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="vietnamese", task="transcribe")

# Đọc file JSON
with open('/content/drive/MyDrive/ASR music vi/dataset_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Tạo dataset từ dữ liệu đã đọc
if isinstance(data, list):
    # Nếu dữ liệu là danh sách các bản ghi
    dataset = Dataset.from_pandas(pd.DataFrame(data))
elif isinstance(data, dict):
    # Nếu dữ liệu là từ điển với các trường
    if "data" in data:
        dataset = Dataset.from_pandas(pd.DataFrame(data["data"]))
    else:
        dataset = Dataset.from_dict(data)

print(dataset)

# Hàm đọc và xử lý dữ liệu âm thanh
def load_audio(example):
    audio_array, sampling_rate = sf.read(example["audio_filepath"])
    example["audio"] = {"array": audio_array, "sampling_rate": sampling_rate}
    return example

dataset = dataset.map(load_audio)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_test_split = dataset.train_test_split(test_size=0.1)

# Tạo DatasetDict từ kết quả chia
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

print(dataset)

# Chuẩn bị dữ liệu đầu vào (input_values và labels)
def prepare_dataset(example):
    # Xử lý dữ liệu âm thanh
    audio = example["audio"]

    # Chuyển đổi âm thanh thành features - quan trọng: trích xuất input_features đúng cách
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features
    # Đảm bảo input_features có đúng shape (loại bỏ kích thước batch nếu cần)
    if len(input_features.shape) > 3:  # Nếu input_features có dạng [1, channels, height, width]
        input_features = input_features.squeeze(0)  # Loại bỏ chiều batch để có dạng [channels, height, width]

    example["input_features"] = input_features
    example["labels"] = processor.tokenizer(example["text"]).input_ids

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

processed_dataset = dataset.map(prepare_dataset, remove_columns=["audio_filepath", "audio", "text", "title", "id"])

# 4. Tạo data collator cho Whisper - ĐÃ CHỈNH SỬA
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Chú ý: xử lý input_features đúng cách
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # Kiểm tra shape trước khi trả về
        if "input_features" in batch and len(batch["input_features"].shape) == 4 and batch["input_features"].shape[1] == 1:
            # Nếu shape là [batch_size, 1, height, width], reshape thành [batch_size, height, width]
            batch["input_features"] = batch["input_features"].squeeze(1)

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import re
from jiwer import wer

def normalize_text(text):
    # Chuyển về chữ thường
    text = text.lower()

    # Loại bỏ dấu câu và ký tự đặc biệt (giữ lại dấu thanh nếu là tiếng Việt)
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', '', text)

    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Loại bỏ padding (-100)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Giải mã dự đoán và nhãn
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Áp dụng normalize cho cả nhãn và dự đoán
    norm_pred_str = [normalize_text(text) for text in pred_str]
    norm_label_str = [normalize_text(text) for text in label_str]

    # Tính WER sử dụng văn bản đã được normalize
    error_rate = wer(truth=norm_label_str, hypothesis=norm_pred_str)

    return {"wer": error_rate}

# 6. Thiết lập các tham số huấn luyện
from transformers import EarlyStoppingCallback

# Cập nhật tham số huấn luyện với early stopping
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/ASR music vi/model-whisper/whisper-song-asr",
    eval_strategy="epoch",  # Đánh giá sau mỗi epoch
    save_strategy="epoch",  # Lưu checkpoint sau mỗi epoch
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=50,
    weight_decay=0.01,
    warmup_steps=1000,
    logging_dir="/content/drive/MyDrive/ASR music vi/model-whisper/logs",
    logging_steps=50,
    fp16=True,
    save_total_limit=2,  # Giới hạn số lượng checkpoint lưu trữ
    load_best_model_at_end=True,
    metric_for_best_model="wer",  # Theo dõi WER để đánh giá mô hình tốt nhất
    greater_is_better=False,  # WER càng thấp càng tốt
    predict_with_generate=True,
    generation_max_length=225,
    report_to=["none"],  # Tắt báo cáo wandb
)

# Thêm EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=5,  # Số epoch không cải thiện để dừng huấn luyện
)


# 7. Thêm debug để kiểm tra shape của dữ liệu
print("Checking batch shape...")
debug_batch = data_collator([processed_dataset["train"][0]])
for k, v in debug_batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k} shape: {v.shape}")

# 8. Tạo trainer và chạy huấn luyện
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    callbacks=[early_stopping],  # Thêm callback vào
)

# Thêm debug để theo dõi quá trình
print("Starting training...")
print(f"Device: {model.device}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 9. Kiểm tra forward pass trước khi huấn luyện
try:
    print("Testing forward pass...")
    batch = data_collator([processed_dataset["train"][0]])
    # Kiểm tra shape
    print(f"Input features shape: {batch['input_features'].shape}")

    # Thay đổi shape nếu cần
    if len(batch["input_features"].shape) == 4:
        batch["input_features"] = batch["input_features"].squeeze(1)
        print(f"Updated input features shape: {batch['input_features'].shape}")

    batch = {k: v.to(model.device) for k, v in batch.items()}
    outputs = model(**batch)
    print(f"Forward pass successful! Loss: {outputs.loss.item()}")
except Exception as e:
    print(f"Error during forward pass: {str(e)}")
    import traceback
    traceback.print_exc()

# 10. Chạy huấn luyện
try:
    print("Starting training loop...")
    # Set decoder_input_ids for generation
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {str(e)}")
    import traceback
    traceback.print_exc()

# 11. Lưu mô hình (nếu huấn luyện thành công)
model.save_pretrained("/content/drive/MyDrive/ASR music vi/model-whisper/whisper-vi-finetune-final")
processor.save_pretrained("/content/drive/MyDrive/ASR music vi/model-whisper/whisper-vi-finetune-final")

# 12. Kiểm tra inference sau khi huấn luyện
print("\nTesting inference after fine-tuning...")
# Lấy một mẫu âm thanh từ tập test
test_example = dataset["test"][0]
test_audio = test_example["audio"]["array"]
sampling_rate = test_example["audio"]["sampling_rate"]
original_text = test_example["text"]

# Đảm bảo input_features có đúng định dạng cho mô hình
input_features = processor(test_audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
if len(input_features.shape) == 4:
    input_features = input_features.squeeze(1)

# Tạo prompt cho tiếng Việt
forced_decoder_ids = processor.get_decoder_prompt_ids(language="vietnamese", task="transcribe")

with torch.no_grad():
    generated_ids = model.generate(
        input_features.to(model.device),
        forced_decoder_ids=forced_decoder_ids,
        max_length=225
    )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Original text: {original_text}")
print(f"Transcription: {transcription}")

from huggingface_hub import notebook_login
notebook_login()

model.push_to_hub("huybunn/whisper-small-vietnamese-lyrics-transcription")
processor.push_to_hub("huybunn/whisper-small-vietnamese-lyrics-transcription")
---
library_name: transformers
tags:
- automatic-speech-recognition
- vietnames_songs_crawl_on_youtube
language:
- vi
datasets:
- vietnames_songs_on_youtube
model-index:
- name: whisper-small-vietnamese-lyrics-transcription
  results:
    - task:
        type: automatic-speech-recognition
        name: Automatic Speech Recognition
      dataset:
        name: vietnames_songs_on_youtube
        type: vietnames_songs_crawl_on_youtube
        split: test
      metrics:
        - name: WER
          type: wer
          value: 0.2052
---

# Whisper Small Vietnamese Lyrics Transcription

Fine-tuned from openai/whisper-small using 4.583 hours of Vietnamese song lyrics for training and 0.516 hours for testing.

It achieves the following results on the evaluation set:
- Loss: 0.514788 
- Wer: 0.205209

## Training & Evaluation Results

| Epoch | Training Loss | Validation Loss | WER      |
|-------|--------------|----------------|----------|
| 1     | No log       | 1.653314       | 0.322856 |
| 2     | No log       | 1.048465       | 1.033229 |
| 3     | 1.336200     | 0.820843       | 0.997306 |
| 4     | 1.336200     | 0.646260       | 0.620117 |
| 5     | 1.336200     | 0.514788       | 0.205209 |
| 6     | 0.256200     | 0.537218       | 0.312977 |
| 7     | 0.256200     | 0.543139       | 0.249663 |
| 8     | 0.256200     | 0.549626       | 0.212393 |
| 9     | 0.047100     | 0.562417       | 0.240233 |
| 10    | 0.047100     | 0.590277       | 0.226762 |


import os
import json
import glob
from pathlib import Path

# Define the paths
downloads_path = "/content/drive/MyDrive/ASR music vi/downloads"
train_file = "/content/drive/MyDrive/ASR music vi/train.jsonl"

# Get the list of WAV files in downloads directory
existing_wav_files = []
if os.path.exists(downloads_path):
    existing_wav_files = [os.path.basename(file) for file in glob.glob(os.path.join(downloads_path, "*.wav"))]
    print(f"Found {len(existing_wav_files)} WAV files in downloads directory")
else:
    print(f"Downloads directory not found at {downloads_path}")

# Get the list of WAV files referenced in train.jsonl
referenced_wav_files = []
references_with_data = []  # Store full references for missing files
if os.path.exists(train_file):
    with open(train_file, "r", encoding="utf-8") as f:
        line_number = 0
        for line in f:
            line_number += 1
            try:
                record = json.loads(line)
                if "audio_filepath" in record:
                    # Extract just the filename from the full path
                    wav_filename = os.path.basename(record["audio_filepath"])
                    referenced_wav_files.append(wav_filename)
                    # Store the full record for missing files
                    references_with_data.append((wav_filename, record, line_number))
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {line_number}: {line[:100]}...")
    print(f"Found {len(referenced_wav_files)} WAV file references in train.jsonl")
else:
    print(f"Train file not found at {train_file}")

# Find WAV files that are referenced in train.jsonl but don't exist in downloads folder
missing_files = []
for wav_file, record, line_number in references_with_data:
    if wav_file not in existing_wav_files:
        missing_files.append((wav_file, record["song"], line_number))

print(f"\nSUMMARY:")
print(f"- Total WAV files referenced in train.jsonl: {len(referenced_wav_files)}")
print(f"- WAV files in downloads directory: {len(existing_wav_files)}")
print(f"- Referenced files missing from downloads: {len(missing_files)} ({(len(missing_files)/len(referenced_wav_files))*100:.1f}%)")

if missing_files:
    print("\nFirst 20 missing files:")
    for i, (file, song, line) in enumerate(missing_files[:20], 1):
        print(f"{i}. Line {line}: {file} - {song}")

    # Save all missing files to a text file
    with open("missing_wav_files.txt", "w", encoding="utf-8") as f:
        f.write("Line\tFilename\tSong Title\n")
        for file, song, line in missing_files:
            f.write(f"{line}\t{file}\t{song}\n")
    print(f"\nAll {len(missing_files)} missing files have been saved to 'missing_wav_files.txt'")
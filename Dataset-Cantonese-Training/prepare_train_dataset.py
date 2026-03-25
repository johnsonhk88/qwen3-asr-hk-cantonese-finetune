# prepare_cantonese_dataset.py
import os
from pathlib import Path
from datasets import load_dataset, Audio
import soundfile as sf
from tqdm import tqdm

DATASET_NAME = "AlienKevin/mixed_cantonese_and_english_speech"
OUTPUT_DIR = "./"
TRAIN_JSONL = os.path.join(OUTPUT_DIR, "train.jsonl")
EVAL_JSONL = os.path.join(OUTPUT_DIR, "eval.jsonl")
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")

os.makedirs(WAV_DIR, exist_ok=True)

print("Loading dataset...")
ds = load_dataset(DATASET_NAME, split=["train", "test"])  # already has 9:1 split
ds = {"train": ds[0], "test": ds[1]}

# Force audio loading
for split in ds:
    ds[split] = ds[split].cast_column("audio", Audio(sampling_rate=16000))

def save_split(split_name, output_jsonl):
    print(f"Processing {split_name} split...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(ds[split_name])):
            audio = example["audio"]
            sentence = example["sentence"].strip()
            
            # Use "language Cantonese<asr_text>" because the dataset is HK Cantonese
            text_field = f"language Cantonese<asr_text>{sentence}"
            
            # Save as WAV
            wav_path = os.path.join(WAV_DIR, f"{split_name}_{i:06d}.wav")
            sf.write(wav_path, audio["array"], audio["sampling_rate"], subtype="PCM_16")
            
            # Write JSONL line
            json_line = f'{{"audio":"{os.path.abspath(wav_path)}","text":"{text_field}"}}\n'
            f.write(json_line)

save_split("train", TRAIN_JSONL)
save_split("test", EVAL_JSONL)

print(f"✅ Done! Files saved to {OUTPUT_DIR}")
print(f"   Train: {TRAIN_JSONL}  ({len(ds['train'])} samples)")
print(f"   Eval : {EVAL_JSONL}   ({len(ds['test'])} samples)")
print(f"   WAVs : {WAV_DIR} ({len(list(Path(WAV_DIR).glob('*.wav')))} files)")
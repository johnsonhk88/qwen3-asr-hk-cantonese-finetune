# Qwen3-ASR-HK-Cantonese

Fine-tuning Alibaba's Qwen3-ASR model for improved Automatic Speech Recognition (ASR) performance on Hong Kong Cantonese (yue-HK). This project provides scripts, configurations, and guidelines to adapt the base model using Hong Kong-specific datasets, enhancing accuracy for local accents, slang, and code-switching with English.

## Overview

Qwen3-ASR is a multilingual ASR model supporting over 190 languages, including Cantonese. However, fine-tuning on region-specific data like Hong Kong Cantonese can reduce Word Error Rate (WER) in real-world scenarios, such as conversational speech, noisy environments, or mixed-language inputs common in Hong Kong.

This repository includes:

- Data preparation scripts for Hong Kong Cantonese datasets.
- Fine-tuning scripts based on the official Qwen3-ASR repository.
- Example configurations for single/multi-GPU training.
- Inference and evaluation tools.

## Features

- Hong Kong Cantonese Focus: Optimized for yue-HK accents using datasets like Common Voice zh-HK and MDCC.
- Efficient Training: Supports FlashAttention for faster, memory-efficient fine-tuning.
- Easy Setup: Docker support and pip requirements for reproducibility.
- Evaluation Metrics: Built-in WER/CER calculation using jiwer.
- Deployment Ready: Export to Hugging Face or ONNX for production use.

## Prerequisites

- Python 3.8+
- GPU with CUDA 11.8+ (e.g., NVIDIA A100 or RTX series with 16GB+ VRAM)
- Recommended: Install FlashAttention for performance gains.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/johnsonhk88/qwen3-asr-hk-cantonese.git
cd qwen3-asr-hk-cantonese
```

2. Install dependencies:

```bash
Install dependencies:

```

(Includes qwen-asr, datasets, torch, flash-attn, jiwer, etc.)
3. Optional: For Docker setup:

```bash
docker build -t qwen3-asr-hk .
docker run -it --gpus all -v $(pwd):/app qwen3-asr-hkå
```

## Datasets

This project uses publicly available Hong Kong Cantonese datasets. Download and prepare them as follows:

- Common Voice zh-HK: ~108 hours of validated speech. Download from Mozilla Common Voice.
- MDCC (Multi-Domain Cantonese Corpus): 73.6 hours of audiobook speech. Clone from HLTCHKUST/cantonese-asr.
- Others: HKCAC (conversational, 8.1 hours) and CantoMap (dialogues, ~12.8 hours) for diversity.

## Data Preparation
Convert data to JSONL format with audio paths and transcripts prefixed by "language Cantonese<asr_text>":

```bash
python scripts/prepare_data.py --dataset common_voice_zh_hk --output_dir data/
```

This script handles MP3-to-WAV conversion (16kHz), splitting into train/eval sets (80/20), and augmentation (noise/speed perturbations using audiomentations).
Aim for 50-100+ hours of data for optimal results.

## Fine-Tuning
Use the provided script adapted from Qwen3-ASR's qwen3_asr_sft.py.
## Single GPU Example

```bash
python finetune.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file data/train.jsonl \
  --eval_file data/eval.jsonl \
  --output_dir outputs/ \
  --batch_size 16 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 2 \
  --save_steps 500
  ```

## Multi-GPU (e.g., 2 GPUs)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 finetune.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file data/train.jsonl \
  --eval_file data/eval.jsonl \
  --output_dir outputs/ \
  --batch_size 16 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 2 \
  --save_steps 500
  ```

### Monitor training with TensorBoard

```bash
tensorboard --logdir outputs/
```

## Evaluation
Evaluate the fine-tuned model on a test set:
```bash
python evaluate.py \
  --model_path outputs/checkpoint-1000 \
  --test_file data/test.jsonl \
  --metric wer  # or cer
```

Expected improvements: Base WER ~15-20% on HK data → Fine-tuned ~8-12% (depending on dataset size).

## Inference
Transcribe audio files or URLs:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained("outputs/final_model", dtype=torch.bfloat16, device_map="cuda:0")
results = model.transcribe(audio="path/to/audio.wav")
print(results[0].text)  # e.g., "呢個係香港粵語測試。"
```

For batch inference or API serving, use inference_server.py with FastAPI.

## Contributing
Contributions are welcome! Please fork the repo and submit a pull request. Focus areas:

- Adding more HK-specific datasets.
- Hyperparameter tuning experiments.
- Support for smaller models (e.g., Qwen3-ASR-0.6B).
- Integration with other tools (e.g., Whisper for comparison).

## License
This project is licensed under the Apache 2.0 License, aligning with Qwen3-ASR's license. Datasets may have their own licenses (e.g., CC-BY for Common Voice).

## Acknowledgments
- Based on Qwen3-ASR by Alibaba.
- Datasets from Mozilla, HLTCHKUST, and academic sources.

For questions, open an issue or contact @johnsonhk88 on X.
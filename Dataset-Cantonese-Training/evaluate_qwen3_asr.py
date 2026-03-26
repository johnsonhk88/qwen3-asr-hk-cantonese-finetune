# evaluate_qwen3_asr.py
# FIXED + RTF VERSION — Qwen3-ASR Evaluation (CER + WER + Real-Time Factor)
# Real-Time Factor (RTF) = transcription_time / audio_duration
# RTF < 1.0 → faster than real-time (good!)

import argparse
import json
import os
import time
import torch
import pandas as pd
import soundfile as sf
import warnings
from tqdm.auto import tqdm
import evaluate
from qwen_asr import Qwen3ASRModel

warnings.filterwarnings("ignore")

# ====================== Metrics ======================
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def clean_reference_text(text_field: str) -> str:
    """Remove training prefix 'language XXX<asr_text>...' if present"""
    if "<asr_text>" in text_field:
        return text_field.split("<asr_text>", 1)[1].strip()
    return text_field.strip()


def evaluate_checkpoint(checkpoint_dir: str, test_jsonl: str, output_dir: str,
                        language: str = "Cantonese", max_new_tokens: int = 2048,
                        max_samples: int = None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Qwen3-ASR from: {checkpoint_dir}")
    model = Qwen3ASRModel.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        max_new_tokens=max_new_tokens,
    )

    with open(test_jsonl, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f.readlines()]

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"✅ Loaded {len(test_data)} test samples")

    print("🚀 Starting ASR evaluation...\n")

    results = []
    total_cer = total_wer = total_rtf = 0.0
    total_duration = 0.0   # for optional RTF sanity check

    for i, sample in tqdm(enumerate(test_data), total=len(test_data),
                          desc="Transcribing & Evaluating", unit="sample"):
        audio_path = sample["audio"]
        ref_text_raw = sample["text"]
        ref_text = clean_reference_text(ref_text_raw)

        # === Get audio duration (needed for RTF) ===
        try:
            audio_data, sr = sf.read(audio_path)
            # Works for both mono (1D) and stereo (2D)
            duration = len(audio_data) / sr if audio_data.ndim == 1 else len(audio_data[:, 0]) / sr
        except Exception as e:
            print(f"⚠️ Could not read audio duration for sample {i}: {e}")
            duration = 1.0  # fallback

        # === Timed transcription ===
        start_time = time.perf_counter()
        try:
            asr_results = model.transcribe(
                audio=audio_path,
                language=language,          # "Cantonese" or "yue"
            )
            pred_text = asr_results[0].text.strip() if asr_results else ""
        except Exception as e:
            print(f"⚠️ Transcription failed for sample {i}: {e}")
            pred_text = ""
        transcribe_time = time.perf_counter() - start_time

        # === RTF ===
        rtf = transcribe_time / duration if duration > 0 else 0.0

        # === CER + WER ===
        cer = cer_metric.compute(predictions=[pred_text], references=[ref_text])
        wer = wer_metric.compute(predictions=[pred_text], references=[ref_text])

        results.append({
            "index": i,
            "ref_text": ref_text,
            "pred_text": pred_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "rtf": round(rtf, 4),
            "audio_duration_sec": round(duration, 3),
            "inference_time_sec": round(transcribe_time, 4),
            "audio": audio_path,
        })

        total_cer += cer
        total_wer += wer
        total_rtf += rtf
        total_duration += duration

    # === Summary ===
    n = len(results)
    avg_cer = total_cer / n if n > 0 else 0
    avg_wer = total_wer / n if n > 0 else 0
    avg_rtf = total_rtf / n if n > 0 else 0

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False, encoding="utf-8")

    summary = {
        "checkpoint": checkpoint_dir,
        "test_samples": n,
        "avg_cer": round(avg_cer, 4),
        "avg_wer": round(avg_wer, 4),
        "avg_rtf": round(avg_rtf, 4),
        "language": language,
        "max_new_tokens": max_new_tokens,
        "asr_model": "Qwen3-ASR (fine-tuned or pretrained)",
        "total_audio_hours": round(total_duration / 3600, 3)
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ QWEN3-ASR EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Checkpoint       : {checkpoint_dir}")
    print(f"Test samples     : {n}")
    print(f"Avg CER          : {avg_cer:.4f}  (lower = better)")
    print(f"Avg WER          : {avg_wer:.4f}  (lower = better)")
    print(f"Avg RTF          : {avg_rtf:.4f}  (lower = better, <1.0 = real-time)")
    print(f"Total audio      : {total_duration/3600:.2f} hours")
    print(f"Results saved to : {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-ASR Evaluation (CER + WER + RTF)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path or HF repo ID (e.g. Qwen/Qwen3-ASR-0.6B or ./your-checkpoint)")
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="asr_evaluation_results")
    parser.add_argument("--language", type=str, default="Cantonese",
                        help="Cantonese / yue / English / None")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens for generation")
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    evaluate_checkpoint(
        args.checkpoint_dir,
        args.test_jsonl,
        args.output_dir,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples
    )
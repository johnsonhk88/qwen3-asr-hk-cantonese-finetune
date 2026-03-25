# evaluate_qwen3_asr.py
# FINAL STABLE VERSION — Qwen3-ASR Evaluation (CER + WER only)
# Modeled exactly after evaluate_qwen3_tts.py but adapted for ASR
# No speaker similarity, no UTMOS, no external Whisper needed.

import argparse
import json
import os
import torch
import pandas as pd
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
                        language: str = "Cantonese", max_samples: int = None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading fine-tuned Qwen3-ASR from: {checkpoint_dir}")
    model = Qwen3ASRModel.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else None,
    )

    with open(test_jsonl, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f.readlines()]

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"✅ Loaded {len(test_data)} test samples")

    print("🚀 Starting ASR evaluation...\n")

    results = []
    total_cer = total_wer = 0.0

    for i, sample in tqdm(enumerate(test_data), total=len(test_data),
                          desc="Transcribing & Evaluating", unit="sample"):
        audio_path = sample["audio"]
        ref_text_raw = sample["text"]
        ref_text = clean_reference_text(ref_text_raw)

        # Transcribe with the fine-tuned model
        try:
            asr_results = model.transcribe(
                audio=audio_path,
                language=language,          # "Cantonese" for HK style
                max_new_tokens=2048
            )
            pred_text = asr_results[0].text.strip() if asr_results else ""
        except Exception as e:
            print(f"⚠️ Transcription failed for sample {i}: {e}")
            pred_text = ""

        # Compute CER + WER
        cer = cer_metric.compute(predictions=[pred_text], references=[ref_text])
        wer = wer_metric.compute(predictions=[pred_text], references=[ref_text])

        results.append({
            "index": i,
            "ref_text": ref_text,
            "pred_text": pred_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "audio": audio_path,
        })

        total_cer += cer
        total_wer += wer

    # === Summary ===
    n = len(results)
    avg_cer = total_cer / n
    avg_wer = total_wer / n

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False, encoding="utf-8")

    summary = {
        "checkpoint": checkpoint_dir,
        "test_samples": n,
        "avg_cer": round(avg_cer, 4),
        "avg_wer": round(avg_wer, 4),
        "language": language,
        "asr_model": "Qwen3-ASR-1.7B (fine-tuned)"
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
    print(f"Results saved to : {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-ASR Evaluation (CER + WER)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to fine-tuned checkpoint (e.g. ./qwen3-asr-cantonese-hk/checkpoint-200)")
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test.jsonl (same format as training)")
    parser.add_argument("--output_dir", type=str, default="asr_evaluation_results")
    parser.add_argument("--language", type=str, default="Cantonese",
                        help="Force language for transcription (Cantonese / yue / None)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (for quick testing)")

    args = parser.parse_args()

    evaluate_checkpoint(
        args.checkpoint_dir,
        args.test_jsonl,
        args.output_dir,
        language=args.language,
        max_samples=args.max_samples
    )
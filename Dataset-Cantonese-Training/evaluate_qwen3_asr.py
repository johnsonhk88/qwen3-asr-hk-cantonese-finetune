# evaluate_qwen3_asr.py
# FINAL MANUAL CALCULATION VERSION — Accurate WER/CER for Cantonese + English code-mixing

import argparse
import json
import os
import time
import torch
import pandas as pd
import soundfile as sf
import warnings
import re
from tqdm.auto import tqdm
import jiwer
from qwen_asr import Qwen3ASRModel

warnings.filterwarnings("ignore")


def clean_reference_text(text_field: str) -> str:
    """Remove training prefix if present"""
    if "<asr_text>" in text_field:
        return text_field.split("<asr_text>", 1)[1].strip()
    return text_field.strip()


def tokenize_mixed_text(text: str):
    """Custom tokenizer for Cantonese + English code-mixing"""
    text = re.sub(r'([a-zA-Z0-9\'\-]+)', r' \1 ', text)   # space around English words
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


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
    print("🚀 Starting evaluation with MANUAL WER/CER (custom tokenizer)...\n")

    results = []
    total_cer = total_wer = total_rtf = 0.0
    total_duration = 0.0

    for i, sample in tqdm(enumerate(test_data), total=len(test_data),
                          desc="Evaluating", unit="sample"):
        audio_path = sample["audio"]
        ref_text_raw = sample["text"]
        ref_text = clean_reference_text(ref_text_raw)

        # Audio duration
        try:
            audio_data, sr = sf.read(audio_path)
            duration = len(audio_data) / sr if audio_data.ndim == 1 else len(audio_data[:, 0]) / sr
        except Exception:
            duration = 1.0

        # Transcription
        start_time = time.perf_counter()
        try:
            asr_results = model.transcribe(audio=audio_path, language=language)
            pred_text = asr_results[0].text.strip() if asr_results else ""
        except Exception as e:
            print(f"⚠️ Failed sample {i}: {e}")
            pred_text = ""
        transcribe_time = time.perf_counter() - start_time

        rtf = transcribe_time / duration if duration > 0 else 0.0

        # === WORD-LEVEL (WER) — use our custom tokenizer ===
        ref_tokens = tokenize_mixed_text(ref_text)
        pred_tokens = tokenize_mixed_text(pred_text)

        # Join for jiwer (this respects our custom split)
        ref_str = " ".join(ref_tokens)
        pred_str = " ".join(pred_tokens)

        wer_output = jiwer.process_words(ref_str, pred_str)
        wer_sub = wer_output.substitutions
        wer_del = wer_output.deletions
        wer_ins = wer_output.insertions
        wer_errors = wer_sub + wer_del + wer_ins
        ref_words = len(ref_tokens) or 1

        # Manual WER (this is the correct value now)
        wer = wer_errors / ref_words if ref_words > 0 else 0.0

        # Word arrays for debugging
        ref_words_array = ", ".join(ref_tokens)
        pred_words_array = ", ".join(pred_tokens)

        # === CHARACTER-LEVEL (CER) — manual too ===
        cer_output = jiwer.process_characters(ref_text, pred_text)
        cer_sub = cer_output.substitutions
        cer_del = cer_output.deletions
        cer_ins = cer_output.insertions
        cer_errors = cer_sub + cer_del + cer_ins
        ref_chars = len(cer_output.references[0]) if cer_output.references else 1

        # Manual CER
        cer = cer_errors / ref_chars if ref_chars > 0 else 0.0

        # Character arrays for debugging
        ref_chars_array = ", ".join(cer_output.references[0]) if cer_output.references else ""
        pred_chars_array = ", ".join(cer_output.hypotheses[0]) if cer_output.hypotheses else ""

        results.append({
            "index": i,
            "ref_text": ref_text,
            "pred_text": pred_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "rtf": round(rtf, 4),
            "audio_duration_sec": round(duration, 3),
            "inference_time_sec": round(transcribe_time, 4),
            "ref_words_array": ref_words_array,
            "pred_words_array": pred_words_array,
            "ref_words": ref_words,
            "wer_substitutions": wer_sub,
            "wer_deletions": wer_del,
            "wer_insertions": wer_ins,
            "wer_errors_(S+D+I)": wer_errors,
            "ref_chars_array": ref_chars_array,
            "pred_chars_array": pred_chars_array,
            "ref_chars": ref_chars,
            "cer_substitutions": cer_sub,
            "cer_deletions": cer_del,
            "cer_insertions": cer_ins,
            "cer_errors_(S+D+I)": cer_errors,
            "audio": audio_path,
        })

        total_cer += cer
        total_wer += wer
        total_rtf += rtf
        total_duration += duration

    n = len(results)
    avg_cer = total_cer / n if n > 0 else 0
    avg_wer = total_wer / n if n > 0 else 0
    avg_rtf = total_rtf / n if n > 0 else 0

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE — MANUAL WER/CER with custom tokenizer")
    print("=" * 80)
    print(f"Checkpoint       : {checkpoint_dir}")
    print(f"Test samples     : {n}")
    print(f"Avg CER          : {avg_cer:.4f}")
    print(f"Avg WER          : {avg_wer:.4f}   ← now accurate for code-mixing")
    print(f"Avg RTF          : {avg_rtf:.4f}")
    print(f"Results saved to : {output_dir}/results.csv")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-ASR Evaluation (manual WER/CER)")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="asr_evaluation_results")
    parser.add_argument("--language", type=str, default="Cantonese")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    evaluate_checkpoint(
        args.checkpoint_dir, args.test_jsonl, args.output_dir,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples
    )
#!/usr/bin/env python3
"""
Batch compare transcription models on audio files.
Supports local (faster-whisper) and OpenAI providers.

Usage:
  python3 compare.py --last-days 2
  python3 compare.py --provider openai --last 10
  python3 compare.py --file /data/history/audio.wav
"""

import argparse
import csv
import gc
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

LOCAL_MODELS = [
    ("tiny", "tiny"),
    ("base", "base"),
    ("small", "small"),
    ("medium", "medium"),
    ("large-v2", "large-v2"),
    ("large-v3", "large-v3"),
    ("turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    ("selimc", "/data/models/selimc-whisper-large-v3-turbo-turkish-ct2"),
]

OPENAI_MODELS = [
    ("whisper-1", "whisper-1"),
    ("gpt-4o-mini-transcribe", "gpt-4o-mini-transcribe"),
    ("gpt-4o-transcribe", "gpt-4o-transcribe"),
    ("gpt-4o-transcribe-diarize", "gpt-4o-transcribe-diarize"),
]


def get_device():
    result = subprocess.run(["nvidia-smi"], capture_output=True)
    return "cuda" if result.returncode == 0 else "cpu"


def transcribe_local(audio_path, model_instance):
    """Transcribe using local faster-whisper model."""
    segments, _ = model_instance.transcribe(audio_path, language="tr", beam_size=5)
    return " ".join([s.text.strip() for s in segments])


def transcribe_openai(audio_path, model_name):
    """Transcribe using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it before using OpenAI provider."
        )

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model_name, file=audio_file
        )

    return transcription.text


def filter_files_by_filename_date(files, days):
    """Filter files by date in filename (format: YYYYMMDD)."""
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y%m%d")
    filtered = []
    for f in files:
        try:
            date_str = f.name[:8]
            if date_str >= cutoff_str:
                filtered.append(f)
        except:
            pass
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare transcription models on audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare last 2 days with local models (default)
  python3 compare.py --last-days 2

  # Compare last 10 recordings with OpenAI
  python3 compare.py --provider openai --last 10

  # Compare specific models
  python3 compare.py --provider openai --models whisper-1,gpt-4o-transcribe --last 5

  # Compare single file
  python3 compare.py --file /data/history/audio.wav

Environment Variables:
  OPENAI_API_KEY    Required when using --provider openai
        """,
    )

    parser.add_argument(
        "--provider",
        choices=["local", "openai"],
        default="local",
        help="Transcription provider: local (faster-whisper) or openai (default: local)",
    )
    parser.add_argument(
        "--directory",
        default="/data/history",
        help="Directory containing audio files (default: /data/history)",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        metavar="DAYS",
        help="Process files from last X days (by filename date)",
    )
    parser.add_argument(
        "--last",
        type=int,
        metavar="N",
        help="Process last N files (sorted by name)",
    )
    parser.add_argument(
        "--file",
        help="Process a single file",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of models to test (default: all for provider)",
    )
    parser.add_argument(
        "--output",
        help="Output CSV file (default: directory/comparison_results.csv)",
    )
    parser.add_argument(
        "--download-root",
        default="/data/models",
        help="Directory for local model files (default: /data/models)",
    )
    parser.add_argument(
        "--language",
        default="tr",
        help="Language code for transcription (default: tr)",
    )

    args = parser.parse_args()

    # Determine which models to use based on provider
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        models_to_use = [(m, m) for m in model_names]
    else:
        models_to_use = OPENAI_MODELS if args.provider == "openai" else LOCAL_MODELS

    # Determine output file
    output_file = args.output or str(Path(args.directory) / "comparison_results.csv")

    # Get list of files
    if args.file:
        wav_files = [Path(args.file)]
        print(f"Processing single file: {args.file}")
    else:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)

        wav_files = sorted(directory.glob("*.wav"))

        if not wav_files:
            print(f"No wav files found in {directory}")
            sys.exit(1)

        # Apply filters
        if args.last_days:
            wav_files = filter_files_by_filename_date(wav_files, args.last_days)
            print(f"Filtered to files from last {args.last_days} day(s)")
        elif args.last:
            wav_files = wav_files[-args.last :]
            print(f"Filtered to last {args.last} file(s)")

    print(f"Found {len(wav_files)} audio files to process")
    print(f"Provider: {args.provider}")
    print(f"Testing {len(models_to_use)} models: {[m[0] for m in models_to_use]}")

    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\nERROR: OPENAI_API_KEY environment variable not set!")
            print("Set it with: export OPENAI_API_KEY=sk-xxx")
            sys.exit(1)
        print("OpenAI API key found")

    results = {}

    if args.provider == "local":
        device = get_device()
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Device: {device}, Compute type: {compute_type}\n")

        for col_name, model_name in models_to_use:
            print(f"{'=' * 60}")
            print(f"Loading {col_name} ({model_name})")
            print(f"{'=' * 60}")

            start = time.time()
            try:
                from faster_whisper import WhisperModel

                model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=args.download_root,
                )
                load_time = time.time() - start
                print(f"Loaded in {load_time:.2f}s\n")
            except Exception as e:
                print(f"ERROR loading model: {e}\n")
                continue

            for i, wav_file in enumerate(wav_files, 1):
                filename = wav_file.name
                print(f"  [{i}/{len(wav_files)}] {filename}...", end=" ", flush=True)
                try:
                    text = transcribe_local(str(wav_file), model)
                    results.setdefault(filename, {})[col_name] = text
                    print(f"'{text}'")
                except Exception as e:
                    results.setdefault(filename, {})[col_name] = f"ERROR: {e}"
                    print(f"ERROR: {e}")

            # Free memory
            del model
            gc.collect()
            if device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except:
                    pass

    elif args.provider == "openai":
        print(f"{'=' * 60}")
        print(f"Using OpenAI API")
        print(f"{'=' * 60}\n")

        for col_name, model_name in models_to_use:
            print(f"\nModel: {col_name}")
            print("-" * 60)

            for i, wav_file in enumerate(wav_files, 1):
                filename = wav_file.name
                print(f"  [{i}/{len(wav_files)}] {filename}...", end=" ", flush=True)
                try:
                    text = transcribe_openai(str(wav_file), model_name)
                    results.setdefault(filename, {})[col_name] = text
                    print(f"'{text}'")
                except Exception as e:
                    results.setdefault(filename, {})[col_name] = f"ERROR: {e}"
                    print(f"ERROR: {e}")

    # Save results
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename"] + [col_name for col_name, _ in models_to_use]
        writer.writerow(header)
        for wav_file in wav_files:
            filename = wav_file.name
            row = [filename]
            for col_name, _ in models_to_use:
                row.append(results.get(filename, {}).get(col_name, ""))
            writer.writerow(row)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Review saved audio files and assign them to speakers."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def play_audio(wav_path: Path) -> bool:
    """Try to play audio using available system players."""
    players = ["aplay", "paplay", "ffplay", "vlc", "open"]

    for player in players:
        if shutil.which(player):
            try:
                if player == "open":
                    subprocess.run(
                        ["open", str(wav_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif player == "vlc":
                    subprocess.run(
                        ["vlc", "--play-and-exit", "--intf", "dummy", str(wav_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif player == "ffplay":
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", str(wav_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.run(
                        [player, str(wav_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                return True
            except subprocess.CalledProcessError:
                continue
    return False


def get_speakers(voices_dir: Path) -> dict[str, int]:
    """Get dict of speaker names to sample counts from voices directory."""
    if not voices_dir.exists():
        return {}

    speakers: dict[str, int] = {}
    for wav_path in voices_dir.glob("*.wav"):
        filename = wav_path.stem.strip()
        if not filename:
            continue
        match = re.match(r"^(.+?)(?:_\d+)?$", filename)
        name = match.group(1) if match else filename
        speakers[name] = speakers.get(name, 0) + 1
    return speakers


def main():
    parser = argparse.ArgumentParser(
        description="Label saved audio files for speaker training"
    )
    parser.add_argument(
        "--saved-dir",
        default=os.environ.get("SAVE_AUDIO_DIR", "/data/saved_audio"),
        help="Directory containing saved audio files",
    )
    parser.add_argument(
        "--voices-dir",
        default=os.environ.get("VOICES_DIR", "/data/voices"),
        help="Directory for speaker reference voices",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Process all files, including already labelled ones",
    )
    args = parser.parse_args()

    saved_dir = Path(args.saved_dir)
    voices_dir = Path(args.voices_dir)

    if not saved_dir.exists():
        print(f"Saved audio directory not found: {saved_dir}")
        sys.exit(1)

    voices_dir.mkdir(parents=True, exist_ok=True)

    existing_speakers = get_speakers(voices_dir)

    wav_files = sorted(saved_dir.glob("*.wav"))

    if not wav_files:
        print("No audio files found.")
        return

    print(f"Found {len(wav_files)} audio files in {saved_dir}")
    if existing_speakers:
        speaker_list = [
            f"{name} ({count})" for name, count in sorted(existing_speakers.items())
        ]
        print(f"Existing speakers: {', '.join(speaker_list)}")
    else:
        print("Existing speakers: none")
    print()
    print("Commands: <name> = assign to speaker, 'n' = skip, 'q' = quit, 'd' = delete")
    print()

    for wav_path in wav_files:
        transcript_path = wav_path.with_suffix(".txt")
        transcript = (
            transcript_path.read_text(encoding="utf-8")
            if transcript_path.exists()
            else "(no transcript)"
        )

        print(f"\n{'=' * 60}")
        print(f"File: {wav_path.name}")
        print(f"Transcript: {transcript}")
        print(f"{'=' * 60}")

        can_play = play_audio(wav_path)
        if not can_play:
            print("(Could not play audio - no player found)")

        while True:
            prompt = "Speaker name [n=skip, q=quit, d=delete]: "
            response = input(prompt).strip()

            if response.lower() == "q":
                print("Exiting.")
                return
            elif response.lower() == "n":
                break
            elif response.lower() == "d":
                wav_path.unlink(missing_ok=True)
                transcript_path.unlink(missing_ok=True)
                print(f"Deleted: {wav_path.name}")
                break
            elif response:
                count = existing_speakers.get(response, 0)
                if count == 0:
                    dest_path = voices_dir / f"{response}.wav"
                else:
                    dest_path = voices_dir / f"{response}_{count + 1}.wav"

                shutil.copy2(wav_path, dest_path)
                existing_speakers[response] = count + 1
                print(
                    f"Added to voices: {dest_path.name} (speaker now has {count + 1} sample(s))"
                )

                keep = input("Keep original in saved? [y/N]: ").strip().lower()
                if keep != "y":
                    wav_path.unlink(missing_ok=True)
                    transcript_path.unlink(missing_ok=True)
                    print("Removed from saved.")
                break
            else:
                print("Enter a name or command.")


if __name__ == "__main__":
    main()

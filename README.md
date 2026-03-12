# Wyoming Whisper Speaker

Standalone Wyoming Faster Whisper server with speaker recognition. Available in CPU and GPU variants.

[![Docker Hub](https://img.shields.io/docker/pulls/onesvat/wyoming-whisper-speaker.svg)](https://hub.docker.com/r/onesvat/wyoming-whisper-speaker)

## Table of Contents

- [Behavior](#behavior)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Docker Images](#docker-images)
- [Environment Variables](#environment-variables)
- [Notes](#notes)

## Behavior

- **ASR**: Runs via Faster Whisper
- **Speaker recognition**: Assumes one dominant speaker per input
  - Known voice: `[name] transcript`
  - Unknown voice: `transcript` (no tag)

## Quick Start

### Using Docker Compose

**GPU** (recommended for speed):
```bash
docker compose --profile gpu up --build -d
```

**CPU** (no GPU required):
```bash
docker compose --profile cpu up --build -d
```

### Using Pre-built Images

**CPU**:
```bash
docker run -d -p 10300:10300 \
  -v ./volumes/whisper:/data \
  -v ./volumes/voices:/data/voices \
  onesvat/wyoming-whisper-speaker:cpu
```

**GPU** (requires NVIDIA Docker):
```bash
docker run -d -p 10300:10300 --gpus all \
  -v ./volumes/whisper:/data \
  -v ./volumes/voices:/data/voices \
  onesvat/wyoming-whisper-speaker:gpu
```

## Configuration

### 1. Model Selection

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Set `WHISPER_MODEL` to a Faster-Whisper/ctranslate2 model ID or valid local ctranslate2 directory.

### 2. Reference Voices

Place reference voice files in `./volumes/voices/` as `.wav` files.

- **File naming**: The filename becomes the speaker label
  - Example: `onur.wav` → `[onur]` in transcripts
- **Audio format**: Clean mono or stereo PCM WAV

## Docker Images

Pre-built images are available on [Docker Hub](https://hub.docker.com/r/onesvat/wyoming-whisper-speaker):

| Image | Description | Size |
|-------|-------------|------|
| `onesvat/wyoming-whisper-speaker:cpu` | CPU only | Smaller |
| `onesvat/wyoming-whisper-speaker:gpu` | NVIDIA CUDA 12.4 | Requires nvidia-docker |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `Systran/faster-whisper-large-v3` | Faster-Whisper/ctranslate2 model ID or local path |
| `VOICES_DIR` | `/data/voices` | Directory containing reference voice files |
| `SPEAKER_THRESHOLD` | `0.80` | Speaker recognition confidence threshold |
| `VOICE_SCAN_INTERVAL` | `10` | Seconds between voice file scans |

## Notes

- Local model paths must contain Faster-Whisper files (e.g., `model.bin`)
- Reference audio should be clean mono or stereo PCM WAV
- This is a lightweight matcher optimized for stability, not full diarization
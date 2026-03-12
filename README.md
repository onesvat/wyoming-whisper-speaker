# wyoming-whisper-speaker

Standalone Wyoming Faster Whisper server with speaker recognition. Available in CPU and GPU variants.

## Behavior

- ASR runs via Faster Whisper.
- Speaker recognition assumes one dominant speaker per input.
- Known voice: `[name] transcript`
- Unknown voice: `transcript` (no tag)

## Quick Start

### GPU (recommended for speed)
```bash
docker compose --profile gpu up --build -d
```

### CPU (no GPU required)
```bash
docker compose --profile cpu up --build -d
```

## Setup

1. Choose model in `.env`:

```bash
cp .env.example .env
# set WHISPER_MODEL to a Faster-Whisper/ctranslate2 model id or valid local ctranslate2 directory
```

2. Put reference voices in `./volumes/voices` as `.wav` files.
   - File name is the label, e.g. `onur.wav` -> `[onur]`

3. Start (see Quick Start above).

## Docker Images

Pre-built images available on Docker Hub:

- `onesvat/wyoming-whisper-speaker:cpu` - CPU only, smaller image
- `onesvat/wyoming-whisper-speaker:gpu` - NVIDIA CUDA 12.4, requires nvidia-docker

```bash
# Pull and run CPU
docker run -d -p 10300:10300 -v ./volumes/whisper:/data -v ./volumes/voices:/data/voices onesvat/wyoming-whisper-speaker:cpu

# Pull and run GPU
docker run -d -p 10300:10300 --gpus all -v ./volumes/whisper:/data -v ./volumes/voices:/data/voices onesvat/wyoming-whisper-speaker:gpu
```

## Env vars

- `WHISPER_MODEL` (default: `Systran/faster-whisper-large-v3`)
- `VOICES_DIR` (default: `/data/voices`)
- `SPEAKER_THRESHOLD` (default: `0.80`)
- `VOICE_SCAN_INTERVAL` (default: `10` seconds)

## Notes

- If using local model path, it must contain Faster-Whisper files like `model.bin`.
- Reference files should be clean mono or stereo PCM WAV.
- This is a lightweight matcher optimized for stability, not diarization.
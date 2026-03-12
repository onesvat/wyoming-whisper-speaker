#!/usr/bin/env bash
cd /app
python3 -m wyoming_faster_whisper \
    --uri 'tcp://0.0.0.0:10300' \
    --data-dir '/data' "$@"
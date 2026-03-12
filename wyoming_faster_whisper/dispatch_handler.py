"""Event handler with minimal full-utterance speaker tagging."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import Transcriber
from .models import ModelLoader
from .speaker_recognition import SpeakerConfig, SpeakerRecognizer

_LOGGER = logging.getLogger(__name__)


class DispatchEventHandler(AsyncEventHandler):
    """Dispatches ASR and optionally prefixes transcript with speaker name."""

    def __init__(
        self,
        wyoming_info: Info,
        loader: ModelLoader,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()

        self._loader = loader
        self._transcriber: Optional[Transcriber] = None
        self._transcriber_future: Optional[asyncio.Future] = None
        self._language: Optional[str] = None

        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._speaker_recognizer = self._create_speaker_recognizer()

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)

            if (self._transcriber is None) and (self._transcriber_future is None):
                self._transcriber_future = asyncio.create_task(
                    self._loader.load_transcriber(self._language)
                )

            return True

        if AudioStop.is_type(event.type):
            if self._transcriber is None:
                assert self._transcriber_future is not None
                self._transcriber = await self._transcriber_future

            assert self._transcriber is not None
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            text = await asyncio.to_thread(self._transcribe_plain)
            if text and self._speaker_recognizer is not None:
                speaker = await asyncio.to_thread(
                    self._speaker_recognizer.identify, self._wav_path
                )
                if speaker:
                    text = f"[{speaker}] {text}"

            _LOGGER.info(text)
            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            self._language = None
            self._transcriber = None
            self._transcriber_future = None

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language or self._loader.preferred_language
            _LOGGER.debug("Language set to %s", self._language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    def _transcribe_plain(self) -> str:
        assert self._transcriber is not None

        return self._transcriber.transcribe(
            self._wav_path,
            self._language,
            beam_size=self._loader.beam_size,
            initial_prompt=self._loader.initial_prompt,
        )

    def _create_speaker_recognizer(self) -> Optional[SpeakerRecognizer]:
        voices_dir = Path(os.environ.get("VOICES_DIR", "/data/voices"))
        if not voices_dir.exists():
            _LOGGER.info(
                "Speaker recognition disabled (missing voices dir: %s)", voices_dir
            )
            return None

        try:
            threshold = float(os.environ.get("SPEAKER_THRESHOLD", "0.80"))
            scan_interval_s = int(os.environ.get("VOICE_SCAN_INTERVAL", "10"))
            config = SpeakerConfig(
                voices_dir=voices_dir,
                threshold=threshold,
                scan_interval_s=scan_interval_s,
            )
            recognizer = SpeakerRecognizer(config=config)
            _LOGGER.info("Speaker recognition enabled. Voices dir: %s", voices_dir)
            return recognizer
        except Exception as err:
            _LOGGER.exception(
                "Speaker recognition disabled due to initialization error: %s", err
            )
            return None

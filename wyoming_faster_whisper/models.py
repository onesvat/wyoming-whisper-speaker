"""Logic for model selection, loading, and transcription."""

import asyncio
import logging
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .const import SttLibrary, Transcriber
from .faster_whisper_handler import FasterWhisperTranscriber

_LOGGER = logging.getLogger(__name__)

TRANSCRIBER_KEY = Tuple[SttLibrary, str]


class ModelLoader:
    """Load transcribers for models."""

    def __init__(
        self,
        preferred_stt_library: SttLibrary,
        preferred_language: Optional[str],
        download_dir: Union[str, Path],
        local_files_only: bool,
        model: Optional[str],
        compute_type: str,
        device: str,
        beam_size: int,
        cpu_threads: int,
        initial_prompt: Optional[str],
        vad_parameters: Optional[Dict[str, Any]],
    ) -> None:
        self.preferred_stt_library = preferred_stt_library
        self.preferred_language = preferred_language

        self.download_dir = Path(download_dir)
        self.local_files_only = local_files_only

        self.model = model
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters

        self._transcriber: Dict[TRANSCRIBER_KEY, Transcriber] = {}
        self._transcriber_lock: Dict[TRANSCRIBER_KEY, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        """Load or get transcriber from cache for a language."""
        language = language or self.preferred_language
        stt_library = self.preferred_stt_library

        if stt_library == SttLibrary.AUTO:
            stt_library = SttLibrary.FASTER_WHISPER

        model = self.model
        if model is None:
            machine = platform.machine().lower()
            is_arm = ("arm" in machine) or ("aarch" in machine)
            model = guess_model(stt_library, language, is_arm)

        _LOGGER.debug(
            "Selected stt-library '%s' with model '%s'", stt_library.value, model
        )

        assert stt_library != SttLibrary.AUTO
        assert model

        key = (stt_library, model)

        async with self._transcriber_lock[key]:
            transcriber = self._transcriber.get(key)
            if transcriber is not None:
                return transcriber

            transcriber = FasterWhisperTranscriber(
                model,
                cache_dir=self.download_dir,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                vad_parameters=self.vad_parameters,
            )

            self._transcriber[key] = transcriber

        return transcriber

    async def transcribe(
        self, wav_path: Union[str, Path], language: Optional[str]
    ) -> str:
        """Transcribe WAV file using appropriate transcriber.

        Assume WAV file is 16Khz 16-bit mono PCM.
        """
        transcriber = await self.load_transcriber(language)
        text = await asyncio.to_thread(
            transcriber.transcribe,
            wav_path,
            language=language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt,
        )
        _LOGGER.debug("Transcribed audio: %s", text)

        return text


def guess_model(stt_library: SttLibrary, language: Optional[str], is_arm: bool) -> str:
    """Automatically guess STT model id."""
    if is_arm:
        return "rhasspy/faster-whisper-tiny-int8"

    return "rhasspy/faster-whisper-base-int8"

"""OpenAI transcription handler."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

from .const import Transcriber

_LOGGER = logging.getLogger(__name__)


class OpenAITranscriber(Transcriber):
    """Transcriber using OpenAI API."""

    def __init__(self, model_id: str = "gpt-4o-transcribe") -> None:
        self.model_id = model_id
        self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI provider. "
                "Please set OPENAI_API_KEY before starting the service."
            )

        _LOGGER.info("Initialized OpenAI transcriber with model: %s", model_id)

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Transcribe audio file using OpenAI API.

        Args:
            wav_path: Path to WAV file (16Khz 16-bit mono PCM)
            language: Language code (e.g., "tr", "en")
            beam_size: Ignored for OpenAI (not supported)
            initial_prompt: Ignored for OpenAI (not supported)

        Returns:
            Transcribed text
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        client = OpenAI(api_key=self.api_key)

        with open(wav_path, "rb") as audio_file:
            # Note: OpenAI API doesn't support beam_size or initial_prompt
            transcription = client.audio.transcriptions.create(
                model=self.model_id,
                file=audio_file,
                language=language if language else None,
            )

        return transcription.text

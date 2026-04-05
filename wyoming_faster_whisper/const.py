"""Constants."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class SttLibrary(str, Enum):
    """Speech-to-text library."""

    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    OPENAI = "openai"


AUTO_LANGUAGE = "auto"
AUTO_MODEL = "auto"


class Transcriber(ABC):
    """Base class for transcribers."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        pass

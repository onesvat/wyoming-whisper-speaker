"""Utility helpers for speaker-tagged transcript formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class SegmentLike:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: Optional[str]


def overlap_seconds(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """Returns overlap in seconds between 2 time ranges."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def resolve_speaker(
    seg_start: float,
    seg_end: float,
    turns: Iterable[SpeakerTurn],
) -> Optional[str]:
    """Returns the speaker with max overlap for a segment."""
    best_speaker: Optional[str] = None
    best_overlap = 0.0
    for turn in turns:
        overlap = overlap_seconds(seg_start, seg_end, turn.start, turn.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

    if best_overlap <= 0:
        return None

    return best_speaker


def format_tagged_text(
    segments: Iterable[SegmentLike], turns: Iterable[SpeakerTurn]
) -> str:
    """Formats transcript as `[speaker] text` when speaker exists."""
    lines = []
    turns = list(turns)
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        speaker = resolve_speaker(segment.start, segment.end, turns)
        if speaker:
            lines.append(f"[{speaker}] {text}")
        else:
            lines.append(text)

    return "\n".join(lines)

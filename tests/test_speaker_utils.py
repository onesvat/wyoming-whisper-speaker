from wyoming_faster_whisper.speaker_utils import (
    SegmentLike,
    SpeakerTurn,
    format_tagged_text,
    overlap_seconds,
    resolve_speaker,
)


def test_overlap_seconds():
    assert overlap_seconds(0.0, 2.0, 1.0, 3.0) == 1.0
    assert overlap_seconds(0.0, 1.0, 1.0, 2.0) == 0.0


def test_resolve_speaker_by_max_overlap():
    turns = [
        SpeakerTurn(start=0.0, end=1.5, speaker="onur"),
        SpeakerTurn(start=1.5, end=3.0, speaker="gamze"),
    ]

    assert resolve_speaker(0.2, 1.0, turns) == "onur"
    assert resolve_speaker(2.1, 2.9, turns) == "gamze"


def test_format_tagged_text():
    segments = [
        SegmentLike(start=0.0, end=1.0, text="selam"),
        SegmentLike(start=1.1, end=2.0, text="nasilsin"),
        SegmentLike(start=3.0, end=4.0, text=" "),
    ]
    turns = [SpeakerTurn(start=0.0, end=1.2, speaker="onur")]

    output = format_tagged_text(segments, turns)
    assert output == "[onur] selam\nnasilsin"

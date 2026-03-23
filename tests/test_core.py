import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from core import deduplicate_texts, build_storyline, infer_mood, get_video_info, extract_keyframes


BASE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = BASE_DIR / "sample_inputs" / "sample_silent_video.mp4"


def test_deduplicate_texts():
    texts = ["A man is running", "a man is running", "A child is smiling"]
    result = deduplicate_texts(texts)
    assert result == ["A man is running", "A child is smiling"]


def test_build_storyline():
    captions = ["a man is running", "a man is running", "a crowd is cheering"]
    storyline = build_storyline(captions)
    assert "Scene 1:" in storyline
    assert "Scene 2:" in storyline
    assert storyline.count("running") == 1


def test_infer_mood_energetic():
    mood, prompt = infer_mood(["a man is running in a sports race", "a cheering crowd"], "")
    assert mood in {"energetic", "epic"}
    assert "soundtrack" in prompt


def test_get_video_info_sample():
    duration, width, height, fps = get_video_info(str(SAMPLE_VIDEO))
    assert duration > 0
    assert width > 0 and height > 0
    assert fps > 0


def test_extract_keyframes_sample():
    frames = extract_keyframes(str(SAMPLE_VIDEO), sample_every_sec=1.0, max_frames=5, diff_threshold=5.0)
    assert len(frames) >= 1
    assert hasattr(frames[0], "size")

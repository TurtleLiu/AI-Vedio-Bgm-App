import os
import math
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image
from pydub import AudioSegment


def load_caption_model():
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()
    return processor, model, device


def load_music_model():
    from audiocraft.models import MusicGen

    return MusicGen.get_pretrained("facebook/musicgen-small")


def get_video_info(video_path: str) -> Tuple[float, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, width, height, fps


def extract_keyframes(
    video_path: str,
    sample_every_sec: float = 3.0,
    max_frames: int = 12,
    diff_threshold: float = 18.0,
) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(sample_every_sec * fps), 1)

    selected_frames: List[Image.Image] = []
    last_gray = None

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keep = False
        if last_gray is None:
            keep = True
        else:
            diff = cv2.absdiff(gray, last_gray)
            mean_diff = float(diff.mean())
            if mean_diff >= diff_threshold:
                keep = True

        if keep:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected_frames.append(Image.fromarray(rgb))
            last_gray = gray

        if len(selected_frames) >= max_frames:
            break

    cap.release()

    if not selected_frames:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected_frames.append(Image.fromarray(rgb))

    return selected_frames


def caption_images(images: List[Image.Image]) -> List[str]:
    import torch

    processor, model, device = load_caption_model()
    captions: List[str] = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=25)
        text = processor.decode(output[0], skip_special_tokens=True).strip()
        captions.append(text)
    return captions


def deduplicate_texts(texts: List[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for t in texts:
        norm = t.lower().strip()
        if norm not in seen:
            seen.add(norm)
            unique.append(t)
    return unique


def build_storyline(captions: List[str]) -> str:
    caps = deduplicate_texts(captions)
    if not caps:
        return "A video with changing scenes."
    if len(caps) == 1:
        return f"The video mainly shows: {caps[0]}."

    parts = [f"Scene {i}: {c}" for i, c in enumerate(caps[:8], start=1)]
    return " ".join(parts)


def infer_mood(captions: List[str], user_hint: str = "") -> Tuple[str, str]:
    text = " ".join(captions).lower() + " " + user_hint.lower()

    mood_scores = {
        "energetic": 0,
        "uplifting": 0,
        "emotional": 0,
        "suspense": 0,
        "calm": 0,
        "epic": 0,
    }
    keyword_map = {
        "energetic": [
            "running", "sports", "race", "jump", "dance", "crowd",
            "game", "action", "fast", "moving", "playing", "fight",
        ],
        "uplifting": [
            "smile", "happy", "sun", "children", "family", "celebration",
            "success", "win", "friends", "festival",
        ],
        "emotional": [
            "alone", "sad", "crying", "night", "rain", "memory",
            "silhouette", "slow", "dramatic",
        ],
        "suspense": [
            "dark", "shadow", "danger", "mystery", "empty", "tense",
            "chase", "threat", "sudden",
        ],
        "calm": [
            "nature", "sea", "sky", "walking", "mountain", "quiet",
            "road", "sunset", "landscape",
        ],
        "epic": [
            "stadium", "large", "performance", "ceremony", "hero",
            "dramatic", "cinematic", "grand",
        ],
    }

    for mood, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text:
                mood_scores[mood] += 1

    best_mood = max(mood_scores, key=mood_scores.get)
    if max(mood_scores.values()) == 0:
        best_mood = "uplifting"

    prompt_map = {
        "energetic": "energetic cinematic instrumental soundtrack, rhythmic percussion, modern, exciting, sports montage background music",
        "uplifting": "uplifting inspirational instrumental soundtrack, warm strings and piano, hopeful, emotional but positive, cinematic background music",
        "emotional": "emotional cinematic instrumental soundtrack, soft piano and strings, touching, reflective, film background music",
        "suspense": "suspense cinematic instrumental soundtrack, dark textures, tension, subtle percussion, thriller film background music",
        "calm": "calm ambient instrumental soundtrack, soft piano, airy pads, peaceful, documentary background music",
        "epic": "epic cinematic instrumental soundtrack, grand orchestra, powerful drums, inspiring, trailer-like film background music",
    }
    return best_mood, prompt_map[best_mood]


def generate_music_wav(
    music_prompt: str,
    target_duration_sec: float,
    out_wav_path: str,
    progress_callback=None,
) -> None:
    import torchaudio

    model = load_music_model()
    gen_duration = min(max(12, int(target_duration_sec)), 30)

    if progress_callback:
        progress_callback(0.55, f"正在生成配乐（约 {gen_duration} 秒）...")

    model.set_generation_params(
        duration=gen_duration,
        use_sampling=True,
        top_k=250,
        temperature=1.0,
        cfg_coef=3.0,
    )

    wav = model.generate([music_prompt])[0].cpu()
    sample_rate = model.sample_rate

    raw_tmp = str(Path(out_wav_path).with_name("raw_music.wav"))
    torchaudio.save(raw_tmp, wav, sample_rate)

    if progress_callback:
        progress_callback(0.75, "正在扩展音乐长度并做淡入淡出...")

    seg = AudioSegment.from_wav(raw_tmp)
    target_ms = int(target_duration_sec * 1000)
    if len(seg) < target_ms:
        loops = math.ceil(target_ms / len(seg))
        seg = seg * loops

    seg = seg[:target_ms]
    seg = seg.fade_in(500).fade_out(1200)
    seg.export(out_wav_path, format="wav")

    if os.path.exists(raw_tmp):
        os.remove(raw_tmp)


def mux_video_and_audio(video_path: str, wav_path: str, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", wav_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 合成失败：\n{result.stderr}")

import os
import tempfile

import streamlit as st

from core import (
    get_video_info,
    extract_keyframes,
    caption_images,
    build_storyline,
    infer_mood,
    generate_music_wav,
    mux_video_and_audio,
)


st.set_page_config(page_title="AI影视自动配乐", page_icon="🎬", layout="wide")

st.title("🎬 AI 自动影视配乐 APP")
st.caption("上传无声视频，自动理解情节并生成背景音乐，输出带配乐视频。")

with st.sidebar:
    st.header("参数设置")
    sample_every_sec = st.slider("关键帧采样间隔（秒）", 1.0, 6.0, 3.0, 0.5)
    max_frames = st.slider("最多分析帧数", 4, 16, 10, 1)
    custom_hint = st.text_area(
        "可选：补充剧情/风格提示",
        value="",
        placeholder="例如：青春、热血、温暖、纪录片风格、科技感、悬疑……",
    )
    music_bias = st.selectbox(
        "偏好配乐风格",
        ["自动判断", "电影感", "纪录片感", "热血运动", "温暖治愈", "悬疑紧张"],
    )

uploaded_file = st.file_uploader(
    "上传无声视频（mp4 / mov / avi / mkv）",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file:
    st.video(uploaded_file)

    if st.button("开始自动配乐", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = os.path.join(tmpdir, uploaded_file.name)
            with open(input_video, "wb") as f:
                f.write(uploaded_file.read())

            progress = st.progress(0, text="准备处理中...")
            log_box = st.empty()

            def update_progress(v, msg):
                progress.progress(min(int(v * 100), 100), text=msg)
                log_box.info(msg)

            try:
                duration, width, height, fps = get_video_info(input_video)
                update_progress(0.1, f"视频时长：{duration:.1f}s，分辨率：{width}x{height}，FPS：{fps:.2f}")

                keyframes = extract_keyframes(
                    input_video,
                    sample_every_sec=sample_every_sec,
                    max_frames=max_frames,
                )
                update_progress(0.25, f"已抽取关键帧：{len(keyframes)} 张")

                captions = caption_images(keyframes)
                update_progress(0.45, "关键帧语义分析完成")

                storyline = build_storyline(captions)

                extra_hint = custom_hint.strip()
                if music_bias != "自动判断":
                    extra_hint = (extra_hint + " " + music_bias).strip()

                mood, music_prompt = infer_mood(captions, extra_hint)
                if extra_hint:
                    music_prompt = music_prompt + f", {extra_hint}"

                music_wav = os.path.join(tmpdir, "bgm.wav")
                generate_music_wav(
                    music_prompt=music_prompt,
                    target_duration_sec=duration,
                    out_wav_path=music_wav,
                    progress_callback=update_progress,
                )

                output_video = os.path.join(tmpdir, "output_scored.mp4")
                update_progress(0.9, "正在合成最终视频...")
                mux_video_and_audio(input_video, music_wav, output_video)
                update_progress(1.0, "处理完成")

                st.success("已完成自动配乐。")
                st.subheader("情节分析结果")
                st.write("**关键帧描述：**")
                for i, c in enumerate(captions, start=1):
                    st.write(f"{i}. {c}")

                st.write("**剧情摘要：**")
                st.write(storyline)
                st.write("**推断配乐情绪：**", mood)
                st.write("**音乐提示词：**", music_prompt)

                st.subheader("输出视频")
                with open(output_video, "rb") as f:
                    video_bytes = f.read()

                st.video(video_bytes)
                st.download_button(
                    label="下载带配乐视频",
                    data=video_bytes,
                    file_name="video_with_bgm.mp4",
                    mime="video/mp4",
                )

            except Exception as e:
                st.error(f"处理失败：{e}")

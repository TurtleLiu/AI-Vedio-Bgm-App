# AI 自动影视配乐 APP（Streamlit）

这是一个基于 **Streamlit + BLIP + MusicGen + ffmpeg** 的原型应用：

- 输入：无声视频
- 处理：自动抽取关键帧、分析情节、推断配乐风格、生成背景音乐
- 输出：带有 AI 配乐的视频

## 目录结构

```text
ai_video_bgm_app_package/
├── app.py                     # Streamlit 主界面
├── core.py                    # 核心处理逻辑
├── requirements.txt           # Python 依赖
├── README.md                  # 使用说明
├── sample_inputs/
│   └── sample_silent_video.mp4
└── tests/
    └── test_core.py           # 测试代码
```

---

## 功能说明

1. 上传一个无声视频（mp4 / mov / avi / mkv）
2. 自动抽取关键帧
3. 用图像描述模型生成场景描述
4. 拼接为剧情摘要，并自动推断配乐情绪
5. 用 MusicGen 生成纯背景音乐
6. 使用 ffmpeg 将音乐混入原视频，输出有声视频

> 当前版本生成的是 **背景配乐**，不包含专业级拟音（脚步、打斗、环境声）和对白。

---

## 环境要求

- Python 3.10 或 3.11
- 建议有 NVIDIA GPU（CPU 也能跑，但会很慢）
- 系统已安装 `ffmpeg`

---

## 依赖安装

### 1. 安装 ffmpeg

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y ffmpeg
```

#### macOS

```bash
brew install ffmpeg
```

#### Windows

下载安装 ffmpeg，并将 `ffmpeg.exe` 加入系统 PATH。

### 2. 安装 Python 依赖

建议先创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 编译 / 运行命令

本项目是 Python + Streamlit 应用，不需要传统“编译”。

### 启动 APP

```bash
streamlit run app.py
```

启动后，在浏览器中访问终端输出的本地地址（通常是 `http://localhost:8501`）。

---

## 测试代码

### 运行测试

```bash
pytest -q
```

### 测试覆盖内容

- 去重文本逻辑
- 剧情摘要拼接逻辑
- 配乐情绪推断逻辑
- 示例视频读取
- 关键帧抽取

> 测试代码默认不下载大模型，也不执行 MusicGen 生成，以保证测试更快。

---

## 测试输入示例视频

示例视频位于：

```text
sample_inputs/sample_silent_video.mp4
```

这是一个短的无声测试视频，可用于验证视频读取、关键帧抽取和 Streamlit 上传流程。

---

## 运行流程示例

1. 运行：

```bash
streamlit run app.py
```

2. 上传：
   - `sample_inputs/sample_silent_video.mp4`
3. 点击：**开始自动配乐**
4. 等待模型分析和音乐生成
5. 下载输出结果视频

---

## 技术栈说明

- **Streamlit**：前端界面
- **OpenCV**：视频读取与关键帧抽取
- **BLIP**：关键帧图像描述
- **MusicGen**：背景音乐生成
- **pydub**：音频拼接与淡入淡出
- **ffmpeg**：音视频合成

---

## 已知局限

- 当前更适合做 **背景配乐**，不等于完整影视声音设计
- 对剧情的理解是基于关键帧描述，尚未达到镜头级叙事理解
- 复杂长视频在 CPU 上处理会比较慢
- 影视级配乐还需要更细的情绪分段、切点控制和混音能力

---

## 下一步可扩展方向

- 支持镜头切分与分段配乐
- 增加环境音 / 拟音自动生成
- 增加对白、字幕和情绪曲线分析
- 接入更强的视频多模态模型
- 增加批量视频处理和任务队列


# Kakure (隠れ)

[English](README.en.md)

专为 ASMR 音声设计的双语翻译工具。

为音声融合其他语言的音轨。

## 功能特性

- **ASR（语音识别）**：可选 faster-whisper（多语言）或 kotoba-whisper（日语优化）
- **AI 翻译**：OpenAI GPT
- **TTS（语音合成）**：中文语音生成——可选 edge-tts（云端、预制音色）或 IndexTTS（本地 GPU、声音克隆）
- **人声分离**：可选基于 Demucs 的人声/背景分离，混合更干净
- **混合模式**：四种双语输出模式：
  - `dual`：日语左声道，中文右声道
  - `overlay`：中文语音以较低音量叠加
  - `sequential`：日语片段后跟中文翻译
  - `whisper`：中文语音极低音量（耳语感）

## Windows 一键安装（小白专用）

不想碰命令行？用这种方式：

1. 从 GitHub 页面点 **Code → Download ZIP** 下载并解压
2. 双击 `install.bat` —— 脚本会自动：
   - 检测/安装 Python 3.12（没有的话自动装，无需管理员权限）
   - 创建虚拟环境 `.venv`
   - 安装 Kakure 及全部依赖（可选用清华镜像加速）
   - 自动下载免安装版 ffmpeg 到项目 `bin\ffmpeg` 目录（不用自己配 PATH）
   - 生成 `kakure.toml` 配置文件
3. 双击 `start-kakure.bat` —— 浏览器会自动打开 Kakure 界面
4. 在 **Settings** 页填入 OpenAI API Key，然后上传音频开始使用

> **提示**
> - 首次运行会自动下载 Whisper 模型（large-v3 约 3GB），请耐心等待
> - 一键安装默认只装核心功能（faster-whisper + edge-tts，CPU 可用）
> - 需要 GPU 进阶功能（IndexTTS 声音克隆、Demucs 人声分离、kotoba-whisper）的
>   用户请见下方「手动安装」说明，在虚拟环境中 `pip install -e ".[可选组件]"`

## 手动安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e ".[dev]"

# kotoba-whisper ASR 后端（可选）：
pip install -e ".[kotoba]"

# IndexTTS TTS 后端（可选，需要 NVIDIA GPU）：
pip install -e ".[indextts]"

# Demucs 人声分离（可选，GPU 加速更快）：
pip install -e ".[demucs]"
```

需要 [ffmpeg](https://ffmpeg.org/) 处理音频。

## 快速上手

Kakure 通过本地 Web 界面使用，命令行只负责启动服务：

```bash
# 启动 Kakure（默认：http://127.0.0.1:7860，浏览器自动打开）
kakure

# 自定义端口
kakure --port 8080

# 绑定到所有网卡（局域网/远程访问）
kakure --host 0.0.0.0

# 不自动打开浏览器
kakure --no-browser
```

## 许可证

MIT

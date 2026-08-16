# Kakure (隠れ)

[English](README.en.md)

专为 ASMR 音声设计的双语翻译工具。

为音声融合其他语言的音轨。

## 功能特性

- **ASR（语音识别）**：可选 faster-whisper（多语言）或 kotoba-whisper（日语优化）
- **AI 翻译**：OpenAI GPT
- **TTS（语音合成）**：中文语音生成——基于 audiocpp（audio.cpp 引擎，本地 GPU、支持音色克隆）
- **人声分离**：可选基于 Demucs 的人声/背景分离，混合更干净
- **混合模式**：四种双语输出模式：
  - `dual`：日语左声道，中文右声道
  - `overlay`：中文语音以较低音量叠加
  - `sequential`：日语片段后跟中文翻译
  - `whisper`：中文语音极低音量（耳语感）

## 整合包（免安装，解压即用）

不想装 Python、不想跑安装脚本？直接用整合包：

1. 下载 `Kakure-整合包-vX.zip`（由作者发布，或自行用 `build_package.py` 打包）
2. 解压到任意目录（建议路径无中文无空格，如 `D:\Kakure`）
3. 双击 `start-kakure.bat` —— 内置了 Python 3.11 运行时和全部依赖，开箱即用
4. 在 **Settings** 页填入 OpenAI API Key（默认对接 DeepSeek），开始使用

整合包特点：
- 内置 Python 3.11 便携版运行时，无需安装任何环境
- 内置共享版 FFmpeg（torchcodec 兼容）
- 默认内置 whisper small/base 模型，语音识别开箱即用
- 所有模型下载到包内 `models\` 目录，不写系统目录
- 已配置 hf-mirror.com 镜像端点，国内网络下载模型更稳定
- 杀毒软件可能对未签名的内置 Python 误报，添加信任即可

自行打包（在 Windows 上）：

```bash
python build_package.py                    # CPU 版（默认，包含全部可选组件）
python build_package.py --core-only        # 只含核心功能（faster-whisper）
python build_package.py --cuda             # PyTorch 用 CUDA 版（需要 NVIDIA 显卡）
python build_package.py --mirror           # 用清华 PyPI 镜像加速下载
python build_package.py --full-models      # 把本机 HF 缓存里的模型全部打进包（含 audiocpp TTS GGUF）
python build_package.py --zip-out DIR      # zip 输出到其他目录（磁盘空间不够时用）
```

输出在 `dist/`：文件夹 `Kakure/` 和压缩包 `Kakure-整合包-vX.zip`。
打包机需要提前准备：本地 HF 模型缓存（含 Whisper / kotoba-whisper / Demucs / audiocpp TTS 的 GGUF）。

## Windows 一键安装（小白专用）

不想碰命令行？用这种方式：

1. 从 GitHub 页面点 **Code → Download ZIP** 下载并解压
2. 双击 `install.bat` —— 脚本会自动：
   - 检测/安装 Python 3.11（没有的话自动装，无需管理员权限）
   - 创建虚拟环境 `.venv`
   - 安装 Kakure 及全部依赖（可选用清华镜像加速）
   - 自动下载免安装版 ffmpeg 到项目 `bin\ffmpeg` 目录（不用自己配 PATH）
   - 生成 `kakure.toml` 配置文件
3. 双击 `start-kakure.bat` —— 浏览器会自动打开 Kakure 界面
4. 在 **Settings** 页填入 OpenAI API Key，然后上传音频开始使用

> **提示**
> - 首次运行会自动下载 Whisper 模型（large-v3 约 3GB），请耐心等待
> - 一键安装默认只装核心功能（faster-whisper，CPU 可用）
> - 需要 GPU 进阶功能（audiocpp 音色克隆、Demucs 人声分离、kotoba-whisper）的
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

# audiocpp TTS 引擎（内置，无需 pip 安装；需要 NVIDIA GPU + CUDA 运行时）
# 模型 GGUF 通过界面 Models 标签页下载，或在 kakure.toml 配置 audiocpp_model 指向本地文件

# Demucs 人声分离（可选，GPU 加速更快）：
pip install -e ".[demucs]"
```

需要 [ffmpeg](https://ffmpeg.org/) 处理音频。

## 快速上手

Kakure 通过本地 Web 界面使用，命令行只负责启动服务：

```bash
# 启动 Kakure（默认：http://127.0.0.1:7530，浏览器自动打开）
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

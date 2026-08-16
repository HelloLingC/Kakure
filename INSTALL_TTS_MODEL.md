# 如何为 Kakure 添加新的 audiocpp TTS 模型

Kakure 的 TTS 由侧边进程 `audiocpp_server.exe`（audio.cpp 引擎）驱动。要使用一个新
的 TTS 模型，核心只有两步：**把模型文件弄到本地**，然后**让 Kakure 认识它**。

本文档覆盖从「下载模型」到「让 Kakure 的界面/配置完全支持它」的全部路径。

---

## 1. 了解模型文件的位置

Kakure 支持三种摆放 GGUF 的方式，按优先级（见 `kakure/tts.py` 的
`_resolve_model_path()`）：

1. **`kakure.toml` 里显式配置 `audiocpp_model`** —— 指向任意本地路径（`.gguf`
   文件）。最高优先级。
2. **HuggingFace 缓存自动发现** —— `audiocpp_model` 为空时，按
   `AUDIOCPP_FAMILY_DOWNLOADS` 里的 `repo_id` + `file` 在 HF 缓存里查找。
3. **默认无模型** —— 找不到就只用服务端内置默认音色（无克隆能力）。

所以最直接的使用方式：把 `.gguf` 放到任意目录（惯例是
`audiocpp/models/<系列>/xxx.gguf`），然后在 `kakure.toml` 里配：

```toml
audiocpp_family = "my_family"
audiocpp_model = "audiocpp/models/MyFamily/my-model-q8_0.gguf"
```

---

## 2. 下载模型文件

### 方式 A：Kakure 的 Models 标签页（推荐，支持镜像）

Models 页的「audiocpp (TTS)」分组列出的模型来自 `kakure/models.py` 的
`MODEL_GROUPS`。点 Download 会用 `huggingface_hub`（自动遵循 `HF_ENDPOINT`）
下载到缓存目录（`model_dir` 配置下则为项目内）。

要把一个新模型加进下载按钮，在 `MODEL_GROUPS` 的 `GROUP_AUDIOCPP` 列表加一项：

```python
{
    "id": "audiocpp-my-model-q8_0",
    "name": "MyModel Q8_0 (中文)",
    "repo_id": "audio-cpp/audio.cpp-gguf",
    "file": "MyModel-GGUF/my-model-q8_0.gguf",      # 仓库内路径
    "approx_size": "~1.8 GB",
    "probe_file": "MyModel-GGUF/my-model-q8_0.gguf", # 与 file 一致用于检测已装
    "family": "my_family",                            # 选中的系列名
},
```

注意：Models 页下载落到 HF 缓存，**不会**自动设置 `audiocpp_model`。下载后仍需在
配置里指向它（或在缓存中设置 `audiocpp_model` 为空以触发自动发现，前提是该系列
在 `AUDIOCPP_FAMILY_DOWNLOADS` 中）。

### 方式 B：直接用 huggingface_hub 下载（支持 hf-mirror）

```bash
set HF_ENDPOINT=https://hf-mirror.com
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('audio-cpp/audio.cpp-gguf', 'MyModel-GGUF/my-model-q8_0.gguf', local_dir='audiocpp/models'); print(p)"
```

这会下载到 `audiocpp/models/MyModel-GGUF/xxx.gguf`，正好落在惯例目录里，配合
`audiocpp_model` 直接可用。

### 方式 C：audio.cpp 官方 Model Manager

```bash
python audiocpp/tools/model_manager_v2.py install my_package_id --models-root audiocpp/models
```

依赖 `audiocpp/model_specs/*.json` 中声明的 package，文件会落到
`audiocpp/models/<target_directory>/<file>`。注意该脚本内置 `huggingface.co`，
**不遵循 `HF_ENDPOINT`**（中国大陆网络需要方式 A/B）。

---

## 3. 让 Kakure 的界面/设置认识新系列

有多个触点需要同步修改。下面按「界面功能」分组说明。

### 3.1 系列下拉框（Settings → TTS → Family）

列表来自 `kakure/tts.py` 的 `AUDIOCPP_FAMILY_CHOICES`，显示名来自
`kakure/routes.py` 的 `_tts_family_options()`。新增系列需要：

```python
# tts.py
AUDIOCPP_FAMILY_CHOICES: list[str] = [
    "qwen3_tts",
    "my_family",      # 新系列
    ...
]
```

```python
# routes.py
def _tts_family_options() -> list[dict]:
    labels = {
        ...
        "my_family": "MyModel (中文/日英)",
    }
```

### 3.2 变体下拉框（Settings → TTS → Variant）

变体列表来自 `kakure/tts.py` 的 `audiocpp_family_packages()`，它扫描
`audiocpp/model_specs/*.json`。每个 spec 文件来自 audio.cpp 仓库，格式如下：

```json
{
  "family": "my_family",
  "display_name": "MyModel",
  "category": "tts",
  "packages": [
    {
      "id": "my_model_q8_0",
      "display_name": "MyModel Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "MyModel-GGUF",
      "files": ["MyModel-GGUF/my-model-q8_0.gguf"],
      "strip_prefix": "MyModel-GGUF"
    }
  ]
}
```

`audiocpp_family_packages()` 会为每个 GGUF package 算出本地路径
`audiocpp/models/<target_directory>/<local_file>`，并标注 `installed`。因此：
- **只有 `category == "tts"` 且 `format == "gguf"` 的 package 会显示**。
- 从 audio.cpp 仓库拉最新的 `model_specs/*.json` 放到 `audiocpp/model_specs/`
  即可让变体列表自动更新，通常无需改 Kakure 代码。

### 3.3 自动发现默认模型（`audiocpp_model` 留空时）

`kakure/tts.py` 的 `AUDIOCPP_FAMILY_DOWNLOADS` 决定留空时自动用哪个 GGUF：

```python
AUDIOCPP_FAMILY_DOWNLOADS: dict[str, dict] = {
    "qwen3_tts": {
        "repo_id": "audio-cpp/audio.cpp-gguf",
        "file": "Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf",
        "label": "Qwen3-TTS 1.7B (q8_0, 中文/日英)",
    },
    "my_family": {   # 新系列
        "repo_id": "audio-cpp/audio.cpp-gguf",
        "file": "MyModel-GGUF/my-model-q8_0.gguf",
        "label": "MyModel (q8_0)",
    },
}
```

如果新系列不需要自动发现（用户总是手动填 `audiocpp_model`），可跳过此步。

### 3.4 内存调优（session_options）

有些模型（如 IndexTTS2.5）默认的 ggml 图 arena 很大（2 GB/阶段），Windows 上
`ggml_init` 会一次性占用 commit limit，低内存机器会直接崩溃
（`GGML_ASSERT(ctx->mem_buffer != NULL)`）。`kakure/tts.py` 的
`_session_options_for()` 按系列下发缩小的 arena 与 `mem_saver`：

```python
def _session_options_for(settings: Settings) -> dict[str, str] | None:
    if settings.audiocpp_family != "index_tts2":
        return None
    return {
        "index_tts2.gpt_graph_arena_mb": "512",
        ...
        "index_tts2.mem_saver": "true",
    }
```

新系列如果有类似的内存/选项需求，在这里加分支。选项名在 audio.cpp 的
`docs/tts.md` 各模型小节里查（`--session-option <family>.<key>=<value>`）。

---

## 4. 配置示例

```toml
audiocpp_family = "my_family"
audiocpp_model = "audiocpp/models/MyModel-GGUF/my-model-q8_0.gguf"
audiocpp_language = "zh"
audiocpp_reference_audio = "references/my_voice.wav"   # 可留空
audiocpp_reference_text = ""
audiocpp_speed = 1.0
```

---

## 5. 验证

服务端会在配置变更时自动重启（`_AudioCppServer` 的签名包含 model 路径）。可直接
跑一次冒烟测试确认合成成功：

```bash
.venv\Scripts\python.exe -c "from kakure.config import load_settings; from kakure.tts import _SERVER, _REQUEST_TIMEOUT; import httpx; s = load_settings(); base = _SERVER.ensure(s); r = httpx.post(base + '/v1/audio/speech', json={'model':'kakure-tts','input':'测试','language':'zh','speed':1.0}, timeout=_REQUEST_TIMEOUT); print(r.status_code, len(r.content)); _SERVER.stop()"
```

期望输出 `200 <非零字节数>`。若崩溃，查看 `tmp/audiocpp_server.log` 尾部——常见原因：

- `GGML_ASSERT(ctx->mem_buffer != NULL)` → 内存不足，检查 3.4 节 arena 设置。
- 模型路径错误 / 缺 CUDA 运行时 DLL → 检查 `audiocpp_model` 与 `audiocpp` 目录。

---

## 汇总：新系列接入检查清单

| 触点 | 位置 | 必填? |
|---|---|---|
| 下载模型文件 | HF 缓存 / `audiocpp/models/` | ✅ 必须 |
| `audiocpp_family` / `audiocpp_model` 配置 | `kakure.toml` | ✅ 必须 |
| `AUDIOCPP_FAMILY_CHOICES` | `kakure/tts.py` | 界面显示用 |
| `_tts_family_options()` labels | `kakure/routes.py` | 界面显示用 |
| `model_specs/*.json` | `audiocpp/model_specs/` | 变体下拉框用 |
| `MODEL_GROUPS` (GROUP_AUDIOCPP) | `kakure/models.py` | Models 页下载按钮用 |
| `AUDIOCPP_FAMILY_DOWNLOADS` | `kakure/tts.py` | 留空自动发现用 |
| `_session_options_for()` | `kakure/tts.py` | 低内存/特殊选项用 |

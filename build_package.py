#!/usr/bin/env python3
"""Kakure portable package (整合包) builder.

Produces a self-contained folder + zip that runs without any install step:
    embedded Python 3.11, every Python dependency, bundled shared FFmpeg, the
Kakure app and (optionally) pre-downloaded models. Users unzip the package
and double-click ``start-kakure.bat`` - no Python, no pip, no install.

Like ``install.py``, this script is stdlib-only. Run it on the packaging
machine (Windows):

    python build_package.py [options]

Options:
    --core-only         Skip kotoba-whisper / IndexTTS / Demucs extras
                        (smaller package, faster-whisper + edge-tts only)
    --cuda              Install the CUDA build of PyTorch (default: CPU)
    --mirror            Use the Tsinghua PyPI mirror for downloads
    --no-whisper-models Do not bundle the default whisper models
    --with-indextts     Bundle IndexTTS (main checkpoint + aux models) and
                        the references/ folder; default TTS becomes indextts
    --full-models       Bundle every model in the local HF cache
                        (implies --with-indextts)
    --no-zip            Keep the folder but skip the final zip
    --out DIR           Output directory (default: <repo>/dist)

Output layout (dist/Kakure):

    start-kakure.bat        double-click to launch
    使用说明.txt             Chinese user guide
    kakure.toml             pre-filled config (model_dir="models", ...)
    python/                 embedded Python 3.11 + all site-packages
    bin/ffmpeg/bin/         shared FFmpeg 8.1.2 (torchcodec-compatible)
    models/huggingface/     pre-downloaded models (when bundled)

The zip is written as ``dist/Kakure-整合包-v<version>.zip``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# CUDA 12.8 build. PyPI's plain torch/torchaudio wheels on Windows are CPU
# builds; the CUDA builds live on the pytorch.org index as +cu128 local
# versions, which sort higher than the plain versions.
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TSINGHUA_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
# Pinned FFmpeg 8.1.2 full-shared: ships the avcodec/avformat/avutil DLLs that
# torchcodec (pulled in by recent torchaudio) needs. gyan.dev's current
# "release" builds are FFmpeg 9 static-only, which do not work with torchcodec.
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-full_build-shared.7z"
FFMPEG_DLL_PTH = "torchcodec_ffmpeg_path.pth"

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Model repos bundled by default, in preference order. Whichever of these
# exist (completely) in the local HF cache get bundled; the packaged default
# model is the first one found (large-v3 > medium > small > base).
DEFAULT_WHISPER_REPOS = [
    "models--Systran--faster-whisper-large-v3",
    "models--Systran--faster-whisper-medium",
    "models--Systran--faster-whisper-small",
    "models--Systran--faster-whisper-base",
]
DEFAULT_WHISPER_MODELS = ["large-v3", "medium", "small", "base"]

# Auxiliary models IndexTTS-2.5 needs besides the main checkpoint (all resolve
# through the HF cache via HF_ENDPOINT / apply_model_env routing). At runtime
# ``ensure_models_available`` migrates them into <indextts_model_dir>/hf_cache/.
INDEX_TTS_AUX_REPOS = [
    "models--facebook--w2v-bert-2.0",
    "models--amphion--MaskGCT",
    "models--funasr--campplus",
    "models--nvidia--bigvgan_v2_22khz_80band_256x",
]

# Python deps IndexTTS-2.5's inference path needs at runtime. The official
# repo pins a much larger set (keras, tensorboard, matplotlib, opencv, pandas,
# ...) for its WebUI/training; shipping only these keeps the package lean and
# avoids version conflicts with faster-whisper/kotoba/demucs.
INDEX_TTS_DEPS = [
    "accelerate>=1.0",
    "descript-audiotools>=0.7",
    "einops>=0.8",
    "fugashi>=1.2",
    "librosa>=0.10",
    "modelscope>=1.20",
    "munch>=4.0",
    "nemo-text-processing>=1.1",
    "omegaconf>=2.3",
    "openai-whisper>=20231117",
    "pyyaml>=6.0",
    "safetensors>=0.4",
    "scipy>=1.10",
    "sentencepiece>=0.2",
    "tiktoken>=0.5",
    "unidic-lite>=1.0",
    "wetext>=0.0.9",
    "transformers==4.52.1",
]

# Official index-tts repository (installed by cloning, not from PyPI).
INDEX_TTS_GIT = "https://github.com/index-tts/index-tts.git"
# Working copy Kakure keeps of the official repo for the portable build.
INDEX_TTS_CLONE = ROOT / "indextts-src"

# Everything Kakure's Models tab can use, in HF cache directory form.
# IndexTeam/IndexTTS-2.5 (the main checkpoint) is intentionally absent: it is
# bundled by bundle_indextts() from the dev tree (IndexTTS-2.5/) or HF cache.
FULL_MODEL_REPOS = [
    "models--Systran--faster-whisper-tiny",
    "models--Systran--faster-whisper-base",
    "models--Systran--faster-whisper-small",
    "models--Systran--faster-whisper-medium",
    "models--Systran--faster-whisper-large-v3",
    "models--Systran--faster-whisper-distil-large-v3",
    "models--kotoba-tech--kotoba-whisper-v2.0",
    "models--kotoba-tech--kotoba-whisper-v2.1",
    "models--kotoba-tech--kotoba-whisper-v2.2",
    "models--adefossez--HTDemucs",
    "models--facebook--w2v-bert-2.0",
    "models--amphion--MaskGCT",
    "models--funasr--campplus",
    "models--nvidia--bigvgan_v2_22khz_80band_256x",
]

TOTAL_STEPS = 10


def step(num: int, msg: str) -> None:
    print(f"[{num}/{TOTAL_STEPS}] {msg}", flush=True)


def run(cmd: list[str], env: dict | None = None, **kwargs) -> int:
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd, env=env, **kwargs)


def _download_resume(url: str, dest: Path) -> None:
    """Download with HTTP Range resume so flaky connections don't restart."""
    headers = {}
    if dest.exists() and dest.stat().st_size > 0:
        headers["Range"] = f"bytes={dest.stat().st_size}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, "ab") as f:
        if resp.status == 200:  # server ignored Range: restart from scratch
            f.seek(0)
            f.truncate()
        shutil.copyfileobj(resp, f)


def download(url: str, dest: Path, label: str, attempts: int = 5) -> None:
    print(f"  Downloading {label} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            _download_resume(url, dest)
            return
        except Exception as exc:
            print(f"  Download attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt == attempts:
                print(f"[ERROR] Failed to download {label}.")
                sys.exit(1)


def parse_version() -> str:
    src = (ROOT / "kakure" / "__init__.py").read_text("utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', src)
    return m.group(1) if m else "0.0.0"


# ---------------------------------------------------------------------------
# Python runtime
# ---------------------------------------------------------------------------


def setup_python(out: Path) -> Path:
    py_dir = out / "python"
    py_exe = py_dir / "python.exe"
    if py_exe.exists():
        step(1, "Embedded Python already present, reusing it.")
        return py_exe
    step(1, "Downloading embedded Python 3.11 (embeddable zip)...")
    archive = Path(os.environ.get("TEMP", str(ROOT))) / "kakure_python_embed.zip"
    download(PYTHON_EMBED_URL, archive, "Python 3.11 embeddable zip")
    py_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(py_dir)
    archive.unlink(missing_ok=True)
    return py_exe


def enable_site(py_dir: Path) -> None:
    """Uncomment ``import site`` so ``Lib/site-packages`` under the exe dir is used.

    The embeddable distro ships ``python312._pth`` with ``#import site``
    commented out, which fixes sys.path to the file contents and disables
    site-packages entirely. Uncommenting it makes pip's ``Lib/site-packages``
    work and is what makes the whole runtime relocatable (no absolute paths
    anywhere).
    """
    pth = py_dir / f"python{sys.version_info.major}{sys.version_info.minor}._pth"
    # Python 3.11.x -> python311._pth; fall back to a glob if the major/minor
    # of the *build* interpreter differs from the embedded one.
    if not pth.exists():
        matches = sorted(py_dir.glob("python3*._pth"))
        if not matches:
            print(f"[ERROR] No _pth file found in {py_dir}.")
            sys.exit(1)
        pth = matches[0]
    content = pth.read_text(encoding="utf-8")
    if re.search(r"^import site", content, flags=re.M):
        step(2, "site-packages already enabled in the embedded Python.")
        return
    content = content.replace("#import site", "import site")
    pth.write_text(content, encoding="utf-8")
    step(2, "Enabled site-packages in the embedded Python.")


def bootstrap_pip(py_exe: Path, mirror: bool) -> None:
    step(3, "Bootstrapping pip ...")
    get_pip = Path(os.environ.get("TEMP", str(ROOT))) / "kakure_get-pip.py"
    download(GET_PIP_URL, get_pip, "get-pip.py")
    env = dict(os.environ)
    if mirror:
        env["PIP_INDEX_URL"] = TSINGHUA_MIRROR
    if run([str(py_exe), str(get_pip), "--quiet"], env=env) != 0:
        print("[ERROR] Failed to bootstrap pip. Check your network.")
        sys.exit(1)
    get_pip.unlink(missing_ok=True)


def vendor_from_dev_venv(out: Path) -> None:
    """Vendor prebuilt binary packages from the dev venv into the package.

    Some dependencies ship no Windows wheel and would otherwise need a C++
    toolchain at build time:

    - pynini (nemo-text-processing dep for IndexTTS): no PyPI wheel at all.
    - cdifflib: no cp312-win wheel (sdist only, C extension).

    The dev venv holds locally built cp312 binaries; copying them lets pip
    see each requirement as satisfied instead of trying to compile it.
    """
    venv_sp = ROOT / ".venv" / "Lib" / "site-packages"
    dest = out / "python" / "Lib" / "site-packages"
    dest.mkdir(parents=True, exist_ok=True)
    missing = []
    for pkg in ("pynini", "cdifflib"):
        src_pkg = venv_sp / pkg
        src_mod = venv_sp / f"{pkg}.py"
        if not (src_pkg.is_dir() or src_mod.is_file()):
            missing.append(pkg)
            continue
        if src_pkg.is_dir():
            shutil.copytree(
                src_pkg,
                dest / pkg,
                ignore=shutil.ignore_patterns("__pycache__"),
                dirs_exist_ok=True,
            )
        if src_mod.is_file():
            shutil.copy2(src_mod, dest / src_mod.name)
        # Extension modules sit at the site-packages root in the dev venv;
        # same-dir DLL resolution loads them (pynini also needs _pywrapfst).
        pyds = list(venv_sp.glob(f"_{pkg}*.pyd")) + list(venv_sp.glob(f"{pkg}*.pyd"))
        if pkg == "pynini":
            pyds += list(venv_sp.glob("_pywrapfst*.pyd"))
        for pyd in pyds:
            shutil.copy2(pyd, dest / pyd.name)
        for di in venv_sp.glob(f"{pkg}-*.dist-info"):
            shutil.copytree(di, dest / di.name, dirs_exist_ok=True)
        print(f"  Vendored prebuilt {pkg} from the dev venv.", flush=True)
    # OpenFst DLLs (pynini's C++ runtime) also sit at the site-packages root.
    if (venv_sp / "pynini").is_dir():
        for dll in (
            list(venv_sp.glob("fst*.dll"))
            + list(venv_sp.glob("dl.dll"))
            + list(venv_sp.glob("libgcc_s_seh-1.dll"))
            + list(venv_sp.glob("libwinpthread-1.dll"))
        ):
            shutil.copy2(dll, dest / dll.name)
    if missing:
        print(
            "[ERROR] Prebuilt packages not found in the dev venv: "
            + ", ".join(missing)
        )
        print("        Install them once in the dev environment (see README), or")
        print("        build with --core-only to exclude IndexTTS/Demucs.")
        sys.exit(1)


def install_dependencies(
    out: Path, py_exe: Path, core_only: bool, cuda: bool, mirror: bool
) -> None:
    label = "core only" if core_only else "all extras"
    step(4, f"Installing Kakure and dependencies ({label}) ...")
    env = dict(os.environ)
    if mirror:
        env["PIP_INDEX_URL"] = TSINGHUA_MIRROR

    # The embedded (embeddable) Python cannot create pip's isolated build
    # environments (no venv module), so install the build backends into the
    # runtime itself and build kakure without isolation. setuptools is also
    # needed to build sdist-only dependencies (e.g. cdifflib) whose pyproject
    # declares setuptools.build_meta.
    if run(
        [
            str(py_exe),
            "-m",
            "pip",
            "install",
            "--no-input",
            "hatchling",
            "setuptools",
        ],
        env=env,
    ) != 0:
        print("[ERROR] Failed to install the build backends.")
        sys.exit(1)

    if not core_only:
        vendor_from_dev_venv(out)
        # IndexTTS-2.5: ensure the official clone exists so it can be installed
        # into the embedded runtime and bundled as source.
        ensure_indextts_clone()

    cmd = [
        str(py_exe),
        "-m",
        "pip",
        "install",
        "--no-input",
        "--no-warn-script-location",
        "--no-build-isolation",
        f"{ROOT}[kotoba,indextts,demucs]" if not core_only else str(ROOT),
    ]
    if cuda:
        cmd += ["--extra-index-url", PYTORCH_CUDA_INDEX]
    else:
        # The pytorch.org/cpu index publishes identical versions with a "+cpu"
        # local suffix, which PEP 440 sorts higher, so pip picks the CPU
        # wheels while everything else still resolves from PyPI.
        cmd += ["--extra-index-url", PYTORCH_CPU_INDEX]
    # The indextts2-inference wheel was previously appended here; the official
    # clone replaces it entirely (see INDEX_TTS_CLONE above).

    if run(cmd, env=env) != 0:
        print(
            "[ERROR] Failed to install dependencies. If it is a network issue, "
            "re-run with --mirror."
        )
        sys.exit(1)

    if not core_only:
        # The official index-tts repo is a source checkout (uv-managed project),
        # not a PyPI package. Install the `indextts` package itself with
        # --no-deps so pip keeps our pinned versions; the curated dep list is
        # installed separately below.
        clone_cmd = [
            str(py_exe),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--no-warn-script-location",
            "--no-build-isolation",
            "--no-deps",
            str(INDEX_TTS_CLONE),
        ]
        if run(clone_cmd, env=env) != 0:
            print("[ERROR] Failed to install the indextts package from the official clone.")
            sys.exit(1)

        # Install the curated IndexTTS inference deps (torch/torchaudio come
        # from the pytorch index already selected above).
        extra_cmd = [
            str(py_exe),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--no-warn-script-location",
        ]
        if cuda:
            extra_cmd += ["--extra-index-url", PYTORCH_CUDA_INDEX]
        else:
            extra_cmd += ["--extra-index-url", PYTORCH_CPU_INDEX]
        extra_cmd += INDEX_TTS_DEPS
        if run(extra_cmd, env=env) != 0:
            print("[ERROR] Failed to install IndexTTS inference dependencies.")
            sys.exit(1)

    # Provenance: record the exact versions that went into this package.
    lock = ROOT / "dist" / "requirements-lock.txt"
    with open(lock, "w", encoding="utf-8") as f:
        subprocess.call(
            [str(py_exe), "-m", "pip", "freeze"], stdout=f, env=env
        )
    print("  Requirements lock written to dist/requirements-lock.txt", flush=True)


# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------


def _register_ffmpeg_dll_dir(site_packages: Path) -> None:
    """Point the embedded Python at the bundled shared FFmpeg DLLs.

    torchcodec (a torchaudio dependency) loads libtorchcodec_core*.dll whose
    dependencies (avcodec-*.dll, ...) resolve via the Windows DLL search path,
    which does not include PATH. The .pth file registers the DLL directory
    before any user code runs. sys.prefix is the python/ dir, so the target
    resolves relative to it - keeping the package relocatable.
    """
    line = (
        "import sys, os; from pathlib import Path; "
        "os.add_dll_directory(str(Path(sys.prefix).resolve().parent / 'bin' / 'ffmpeg' / 'bin'))\n"
    )
    (site_packages / FFMPEG_DLL_PTH).write_text(line, encoding="utf-8")
    print("       Registered bundled ffmpeg DLLs for torchcodec.", flush=True)


def setup_ffmpeg(out: Path) -> None:
    bundled = out / "bin" / "ffmpeg" / "bin"
    if (bundled / "ffmpeg.exe").exists() and list(bundled.glob("avcodec-*.dll")):
        step(5, "Bundled shared ffmpeg already present, reusing it.")
        _register_ffmpeg_dll_dir(out / "python" / "Lib" / "site-packages")
        return

    # Reuse the dev-tree ffmpeg (install.py downloads it to bin\ffmpeg, older
    # checkouts used bin\ffmpeg-shared) when available - saves a ~57MB download.
    src = None
    for candidate in (ROOT / "bin" / "ffmpeg", ROOT / "bin" / "ffmpeg-shared"):
        if (candidate / "bin" / "ffmpeg.exe").exists() and list(
            (candidate / "bin").glob("avcodec-*.dll")
        ):
            src = candidate
            break
    if src is not None:
        step(5, f"Copying the shared ffmpeg from the dev tree ({src.name}) ...")
        shutil.copytree(src / "bin", bundled, dirs_exist_ok=True)
        _register_ffmpeg_dll_dir(out / "python" / "Lib" / "site-packages")
        return

    step(5, "Downloading the shared ffmpeg (~57MB) ...")
    archive = Path(os.environ.get("TEMP", str(ROOT))) / "kakure_ffmpeg.7z"
    download(FFMPEG_URL, archive, "ffmpeg 8.1.2 shared")
    dest = out / "bin" / "ffmpeg"
    dest.mkdir(parents=True, exist_ok=True)
    tar = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
    if run([str(tar), "-xf", str(archive), "-C", str(dest)]) != 0:
        print("[ERROR] Failed to extract ffmpeg.")
        sys.exit(1)
    archive.unlink(missing_ok=True)
    bin_dir = dest / "bin"
    bin_dir.mkdir(exist_ok=True)
    moved = False
    for sub in dest.glob("ffmpeg-*"):
        src_bin = sub / "bin"
        if (src_bin / "ffmpeg.exe").exists():
            for f in src_bin.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(bin_dir / f.name))
            shutil.rmtree(sub, ignore_errors=True)
            moved = True
            break
    if not moved or not (bin_dir / "ffmpeg.exe").exists():
        print("[ERROR] No shared ffmpeg found after extracting.")
        sys.exit(1)
    _register_ffmpeg_dll_dir(out / "python" / "Lib" / "site-packages")


# ---------------------------------------------------------------------------
# VC runtime DLLs
# ---------------------------------------------------------------------------


def copy_vc_runtime(py_dir: Path) -> None:
    """Copy VC runtime DLLs Windows may not ship.

    torch / ctranslate2 wheels link against vcruntime140_1.dll and msvcp140.dll.
    The embeddable zip only includes vcruntime140.dll, so pull the others from
    System32 when present (any VC++ redist install provides them).
    """
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for dll in ("vcruntime140_1.dll", "msvcp140.dll"):
        src = sys32 / dll
        if src.exists() and not (py_dir / dll).exists():
            shutil.copy2(src, py_dir / dll)
            print(f"  Copied {dll} into the embedded Python.", flush=True)


# ---------------------------------------------------------------------------
# App files
# ---------------------------------------------------------------------------


def ensure_indextts_clone() -> None:
    """Ensure the official index-tts repo is checked out in the dev tree.

    The official IndexTTS-2.5 is a source project (git clone + uv sync), not a
    PyPI package. The portable build pip-installs ``indextts`` from this clone
    into the embedded runtime and ships the source so the subprocess bridge can
    load it at runtime.
    """
    if (INDEX_TTS_CLONE / "pyproject.toml").exists():
        return
    git = shutil.which("git")
    if git is None:
        print("[ERROR] git was not found; IndexTTS-2.5 needs it. Install Git, or "
              "build with --core-only.")
        sys.exit(1)
    INDEX_TTS_CLONE.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Cloning the official index-tts repo -> {INDEX_TTS_CLONE} ...", flush=True)
    if run([git, "clone", "--depth", "1", INDEX_TTS_GIT, str(INDEX_TTS_CLONE)]) != 0:
        print("[ERROR] Failed to clone the official index-tts repo.")
        sys.exit(1)


def _hf_cache_snapshot(repo_dir_name: str) -> Path | None:
    """Return the local snapshot directory for an HF cache repo, or None.

    Prefers the commit hash recorded in ``refs/main``; falls back to the single
    snapshot when there is no ambiguity.
    """
    src = HF_CACHE / repo_dir_name
    if not src.is_dir():
        return None
    snapshots = src / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = src / "refs" / "main"
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
        snap = snapshots / commit
        if snap.is_dir() and list(snap.iterdir()):
            return snap
    entries = [e for e in snapshots.iterdir() if e.is_dir()]
    return entries[0] if len(entries) == 1 else None


def bundle_indextts(out: Path) -> str:
    """Bundle IndexTTS-2.5 (official source + main checkpoint + aux + reference).

    Returns ``"bundled"`` when the main checkpoint was bundled, else ``""``.
    The official repo source is copied to ``models/indextts-src/`` (wired to
    ``indextts_repo_dir``). The main checkpoint goes to the flat
    ``models/indextts/`` layout (wired to ``indextts_model_dir``). Aux models
    stay in the HF cache (``models/huggingface/hub/``);
    ``ensure_models_available`` migrates them into
    ``models/indextts/hf_cache/`` on first use. Reference audio in the dev
    tree's ``references/`` folder is copied to ``references/``.
    """
    dest_hub = out / "models" / "huggingface" / "hub"
    dest_main = out / "models" / "indextts"
    main_source = ""

    ensure_indextts_clone()
    if INDEX_TTS_CLONE.is_dir():
        shutil.copytree(
            INDEX_TTS_CLONE,
            out / "models" / "indextts-src",
            ignore=shutil.ignore_patterns(
                "__pycache__", ".git", ".venv", "checkpoints", "examples", "tools"
            ),
            dirs_exist_ok=True,
        )
        print("  Bundled the official index-tts repo source.", flush=True)

    snapshot = _hf_cache_snapshot("models--IndexTeam--IndexTTS-2.5")
    if snapshot:
        shutil.copytree(
            snapshot,
            dest_main,
            ignore=shutil.ignore_patterns("__pycache__", ".cache"),
            dirs_exist_ok=True,
        )
        print("  Bundled IndexTTS-2.5 main checkpoint (from the HF cache).", flush=True)
        main_source = "bundled"
    else:
        # Fallback: the dev tree may hold a manually downloaded copy.
        src_main = ROOT / "IndexTTS-2.5"
        if src_main.is_dir():
            shutil.copytree(
                src_main,
                dest_main,
                ignore=shutil.ignore_patterns("__pycache__", ".cache"),
                dirs_exist_ok=True,
            )
            print(
                "  Bundled IndexTTS-2.5 main checkpoint (from dev tree IndexTTS-2.5/).",
                flush=True,
            )
            main_source = "bundled"
        else:
            print(
                "  [WARN] IndexTTS-2.5 main checkpoint not found (HF cache or dev "
                "tree IndexTTS-2.5/); IndexTTS will download it on first use.",
                flush=True,
            )
    for repo in INDEX_TTS_AUX_REPOS:
        _bundle_hf_repo(repo, dest_hub)
    refs = ROOT / "references"
    if refs.is_dir():
        shutil.copytree(refs, out / "references", dirs_exist_ok=True)
    return main_source


def _reference_audio_name(out: Path) -> str:
    """First audio file bundled into the package's references/ folder."""
    refs = out / "references"
    if not refs.is_dir():
        return ""
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    for p in sorted(refs.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            return p.name
    return ""


def _packaged_config(whisper_default: str, overrides: dict[str, str] | None = None) -> str:
    example = (ROOT / "kakure.toml.example").read_text(encoding="utf-8")
    # 整合包 defaults: models stay inside the package, whisper model matches
    # what was bundled, and the hf-mirror.com endpoint so in-app model
    # downloads work in China where huggingface.co is blocked/slow.
    out = re.sub(
        r'^whisper_model = .*$', f'whisper_model = "{whisper_default}"', example, flags=re.M
    )
    out = re.sub(r'^model_dir = .*$', 'model_dir = "models"', out, flags=re.M)
    out = re.sub(r'^hf_endpoint = .*$', 'hf_endpoint = "https://hf-mirror.com"', out, flags=re.M)
    for key, value in (overrides or {}).items():
        out = re.sub(rf'^{re.escape(key)} = .*$', f'{key} = {value}', out, flags=re.M)
    return out


def write_app_files(out: Path, whisper_default: str, indextts_main: str) -> None:
    step(6, "Writing app files and config ...")
    overrides = {}
    if indextts_main:
        overrides["tts_backend"] = '"indextts"'
        overrides["indextts_repo_dir"] = '"models/indextts-src"'
        overrides["indextts_model_dir"] = '"models/indextts"'
        ref = _reference_audio_name(out)
        if ref:
            overrides["indextts_reference_audio"] = f'"references/{ref}"'
    (out / "kakure.toml").write_text(_packaged_config(whisper_default, overrides), encoding="utf-8")
    shutil.copyfile(ROOT / "kakure.toml.example", out / "kakure.toml.example")
    if (ROOT / "start-kakure.bat").exists():
        shutil.copyfile(ROOT / "start-kakure.bat", out / "start-kakure.bat")
    if (ROOT / "README.md").exists():
        shutil.copyfile(ROOT / "README.md", out / "README.md")
    guide = ROOT / "使用说明.txt"
    if guide.exists():
        shutil.copyfile(guide, out / "使用说明.txt")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _bundle_hf_repo(repo_dir_name: str, dest_hub: Path) -> bool:
    """Copy one HF cache repo into the package. Returns False when absent
    or when the local copy is incomplete (an ``.incomplete`` blob means a
    download was interrupted; bundling it would ship a broken model)."""
    src = HF_CACHE / repo_dir_name
    if not src.is_dir():
        return False
    if list(src.glob("blobs/*.incomplete")):
        # An .incomplete blob is only a problem when its completed counterpart
        # is missing (a stale leftover from an interrupted download whose blob
        # later completed is harmless - huggingface_hub sometimes leaves the
        # temp file behind).
        stale_only = all(
            (src / "blobs" / f.name.split(".")[0]).exists()
            for f in src.glob("blobs/*.incomplete")
        )
        if not stale_only:
            print(
                f"  [WARN] {repo_dir_name}: local copy is incomplete, skipping.",
                flush=True,
            )
            return False
    print(f"  Bundling {repo_dir_name} ...", flush=True)
    # The HF cache stores each file twice (blobs/ + hardlinked snapshots/).
    # Ship only snapshots/ + refs/ - that is what loaders and
    # try_to_load_from_cache() read, and it halves the package size. Any
    # later download just adds the missing blob.
    dest = dest_hub / repo_dir_name
    for sub in ("snapshots", "refs"):
        src_sub = src / sub
        if src_sub.is_dir():
            shutil.copytree(src_sub, dest / sub, dirs_exist_ok=True)
    return True


def bundle_models(out: Path, full: bool, no_whisper: bool) -> str:
    """Bundle models and return the packaged default whisper model name."""
    dest_hub = out / "models" / "huggingface" / "hub"
    if full:
        repos = FULL_MODEL_REPOS
        default_whisper = "large-v3"
    elif no_whisper:
        repos = []
        default_whisper = "small"
    else:
        repos = DEFAULT_WHISPER_REPOS
        default_whisper = "small"
    if not repos:
        step(7, "Skipping model bundling.")
        return default_whisper
    step(7, f"Bundling models from the local HF cache ({HF_CACHE}) ...")
    missing = [r for r in repos if not _bundle_hf_repo(r, dest_hub)]
    for r in missing:
        print(f"  [WARN] Not found locally, will download on first run: {r}")
    # Pick the default: first bundled repo in preference order.
    for repo, model in zip(DEFAULT_WHISPER_REPOS, DEFAULT_WHISPER_MODELS):
        if (dest_hub / repo).is_dir():
            default_whisper = model
            break
    return default_whisper


# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------


def make_zip(out: Path, version: str, zip_dir: Path | None = None) -> Path:
    step(9, "Creating the zip archive ...")
    zip_dir = zip_dir or ROOT / "dist"
    zip_path = zip_dir / f"Kakure-整合包-v{version}.zip"
    zip_path.unlink(missing_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for root, dirs, files in os.walk(out):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if name.endswith(".pyc") or name.endswith(".pyo"):
                    continue
                full = Path(root) / name
                arc = "Kakure/" + full.relative_to(out).as_posix()
                zf.write(full, arc)
                count += 1
    print(f"  {count} files -> {zip_path}", flush=True)
    return zip_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test(py_exe: Path) -> None:
    step(8, "Smoke-testing the packaged runtime ...")
    code = (
        "import sys, kakure, faster_whisper, edge_tts, pydub, uvicorn, fastapi, pydantic; "
        "print('OK  kakure', kakure.__file__)"
    )
    if run([str(py_exe), "-c", code]) != 0:
        print("[ERROR] Smoke test failed - the packaged runtime is broken.")
        sys.exit(1)
    # The packaged launcher invokes `python -m kakure.cli`; make sure the
    # entry module actually starts (argparse --help prints usage and exits).
    import subprocess as _sp

    out = _sp.run(
        [str(py_exe), "-m", "kakure.cli", "--help"], capture_output=True, text=True
    )
    if out.returncode != 0 or "usage: kakure" not in out.stdout:
        print("[ERROR] `python -m kakure.cli --help` failed - entry point broken.")
        print(out.stdout[-500:])
        print(out.stderr[-500:])
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Kakure portable package (整合包).")
    parser.add_argument(
        "--core-only", action="store_true", help="skip kotoba/indextts/demucs extras"
    )
    parser.add_argument("--cuda", action="store_true", help="install CUDA PyTorch (default: CPU)")
    parser.add_argument("--mirror", action="store_true", help="use the Tsinghua PyPI mirror")
    parser.add_argument(
        "--no-whisper-models", action="store_true", help="do not bundle whisper models"
    )
    parser.add_argument(
        "--with-indextts",
        action="store_true",
        help="bundle IndexTTS (main checkpoint + aux models) and reference "
        "audio; default TTS backend becomes indextts",
    )
    parser.add_argument(
        "--full-models",
        action="store_true",
        help="bundle every model in the local HF cache (implies --with-indextts)",
    )
    parser.add_argument("--no-zip", action="store_true", help="skip the final zip")
    parser.add_argument(
        "--zip-out",
        default="",
        help="directory for the zip archive (default: same as --out)",
    )
    parser.add_argument(
        "--zip-only",
        action="store_true",
        help="zip the existing <out>/Kakure folder and exit (no rebuild)",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "dist"), help="output directory (default: dist)"
    )
    args = parser.parse_args()

    if os.name != "nt":
        print("This builder is designed for Windows targets (run it on Windows).")
        sys.exit(1)

    version = parse_version()
    out = Path(args.out) / "Kakure"
    zip_dir = Path(args.zip_out) if args.zip_out else None
    if args.zip_only:
        if not out.is_dir():
            print(f"[ERROR] {out} not found - nothing to zip.")
            sys.exit(1)
        make_zip(out, version, zip_dir)
        print(f"Zip archive: {out.parent / f'Kakure-整合包-v{version}.zip'}")
        return
    print(f"Building Kakure v{version} portable package -> {out}", flush=True)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    py_exe = setup_python(out)
    enable_site(py_exe.parent)
    bootstrap_pip(py_exe, args.mirror)
    install_dependencies(out, py_exe, args.core_only, args.cuda, args.mirror)

    site_packages = out / "python" / "Lib" / "site-packages"
    if site_packages.is_dir():
        shutil.rmtree(out / "python" / "Scripts", ignore_errors=True)  # non-relocatable launchers

    setup_ffmpeg(out)
    copy_vc_runtime(py_exe.parent)
    whisper_default = bundle_models(out, args.full_models, args.no_whisper_models)
    indextts_main = ""
    if args.full_models or args.with_indextts:
        indextts_main = bundle_indextts(out)
    write_app_files(out, whisper_default, indextts_main)
    smoke_test(py_exe)

    if args.no_zip:
        step(10, "Done (zip skipped).")
    else:
        make_zip(out, version, zip_dir)
        step(10, "Done.")
    print()
    print("=" * 60)
    print(f"Package folder: {out}")
    if not args.no_zip:
        zpath = (zip_dir or ROOT / "dist") / f"Kakure-整合包-v{version}.zip"
        print(f"Zip archive:    {zpath}")
    print("Distribute the zip. Users unzip and double-click start-kakure.bat.")
    print("=" * 60)


if __name__ == "__main__":
    main()

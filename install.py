#!/usr/bin/env python3
"""Kakure one-click installer (main logic).

Invoked by `install.bat` after a usable Python is found. This script uses only the
Python standard library and runs before the virtual environment is created, so it
depends on no third-party packages.

Flow: version check -> virtualenv -> pip upgrade -> install dependencies ->
ffmpeg -> config file -> launch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_312_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
# Pinned FFmpeg 8.1.2 (full-shared build). It ships the avcodec/avformat/avutil/
# DLLs that torchcodec (0.15, supports FFmpeg 4-8) needs to load. gyan.dev's
# current "release" builds are FFmpeg 9 and static-only, neither of which works
# with torchcodec, so we pin an older shared build that does.
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-full_build-shared.7z"
FFMPEG_DLL_PTH = "torchcodec_ffmpeg_path.pth"
TSINGHUA_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def step(num: int, total: int, msg: str) -> None:
    print(f"[{num}/{total}] {msg}", flush=True)


def pause(prompt: str = "Press Enter to exit...") -> None:
    print(prompt, flush=True)
    try:
        input()
    except EOFError:
        pass


def ask(prompt: str, default: str = "") -> str:
    print(prompt, flush=True)
    try:
        return input().strip().lower() or default
    except EOFError:
        return default


def run(cmd: list[str], **kwargs) -> int:
    print(" ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd, **kwargs)


def check_python_version() -> None:
    # Runtime version gate: install.py may be invoked by any Python version, so this
    # must not be trimmed to the pyproject target version.
    if sys.version_info < (3, 10):  # noqa: UP036
        print(
            f"[ERROR] Detected Python {sys.version_info.major}.{sys.version_info.minor}, "
            "which is too old (3.10+ required)."
        )
        print("       Please install Python 3.12 and re-run this installer.")
        pause()
        sys.exit(1)
    print(
        f"[1/6] Detected Python {sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}",
        flush=True,
    )


def setup_venv() -> None:
    if VENV_PY.exists():
        print("[2/6] Virtual environment already exists, reusing it.", flush=True)
    else:
        print("[2/6] Creating virtual environment .venv ...", flush=True)
        if run([sys.executable, "-m", "venv", str(ROOT / ".venv")]) != 0:
            print("[ERROR] Failed to create the virtual environment.")
            pause()
            sys.exit(1)


def upgrade_pip() -> None:
    print("[3/6] Upgrading pip ...", flush=True)
    if run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "--quiet"]) != 0:
        print("[ERROR] Failed to upgrade pip. Check your network.")
        pause()
        sys.exit(1)


def install_package() -> None:
    print()
    print("Use the Tsinghua mirror to speed up downloads? (recommended, Enter=yes) [Y/n]")
    use_mirror = ask("").lower() != "n"
    env = dict(os.environ)
    if use_mirror:
        env["PIP_INDEX_URL"] = TSINGHUA_MIRROR
        print("Using the Tsinghua mirror.")
    else:
        print("Using the official PyPI index.")

    print()
    print(
        "[3/6] Installing Kakure and dependencies (first install may take a while)...",
        flush=True,
    )
    if run([str(VENV_PY), "-m", "pip", "install", "-e", ".", "--quiet"], env=env) != 0:
        print(
            "[ERROR] Failed to install dependencies. If it is a network issue, "
            "re-run this installer and choose y to use the Tsinghua mirror."
        )
        pause()
        sys.exit(1)
    print("Dependencies installed.", flush=True)
    print()


def _venv_site_packages() -> Path:
    return ROOT / ".venv" / "Lib" / "site-packages"


def _register_ffmpeg_dll_dir(dll_dir: Path) -> None:
    """Point the venv Python at the shared FFmpeg DLLs at interpreter startup.

    torchcodec loads libtorchcodec_core*.dll, whose dependencies (avcodec-*.dll,
    avformat-*.dll, ...) are resolved via the Windows DLL search path, which does
    NOT include PATH under the safe-DLL-search mode Python enables. The .pth file
    calls os.add_dll_directory() before any user code runs, so `import torchcodec`
    works out of the box.
    """
    site = _venv_site_packages()
    if not site.is_dir():
        print("[WARN] .venv site-packages not found; cannot register ffmpeg DLLs for torchcodec.")
        return
    line = f"import os; os.add_dll_directory(r'{dll_dir}')\n"
    (site / FFMPEG_DLL_PTH).write_text(line, encoding="utf-8")
    print(f"       Registered {dll_dir} for torchcodec.", flush=True)


def _shared_ffmpeg_dir() -> Path | None:
    """Directory of a system ffmpeg that ships torchcodec-compatible DLLs."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    dll_dir = Path(exe).resolve().parent
    for dll in dll_dir.glob("avcodec-*.dll"):
        m = re.fullmatch(r"avcodec-(\d+)\.dll", dll.name)
        # avcodec 58..62 == FFmpeg 4..8 (torchcodec 0.15 supports FFmpeg 4-8).
        if m and 58 <= int(m.group(1)) <= 62:
            return dll_dir
    return None


def setup_ffmpeg() -> None:
    bundled = ROOT / "bin" / "ffmpeg" / "bin"
    if (bundled / "ffmpeg.exe").exists() and list(bundled.glob("avcodec-*.dll")):
        print("[4/6] Found the bundled shared ffmpeg.", flush=True)
        _register_ffmpeg_dll_dir(bundled)
        return

    system_shared = _shared_ffmpeg_dir()
    if system_shared:
        print("[4/6] Detected a shared system ffmpeg with avcodec DLLs, reusing it.", flush=True)
        _register_ffmpeg_dll_dir(system_shared)
        return

    if shutil.which("ffmpeg"):
        print("[4/6] System ffmpeg is not a shared build (no avcodec DLLs),")
        print("       bundling a shared build so torchcodec can load.")
    else:
        print("[4/6] ffmpeg not detected, downloading the shared build (~57MB)...", flush=True)
    print(f"Download URL: {FFMPEG_URL}", flush=True)

    dest_dir = ROOT / "bin" / "ffmpeg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(os.environ.get("TEMP", str(ROOT))) / "kakure_ffmpeg.7z"
    try:
        urllib.request.urlretrieve(FFMPEG_URL, archive)
    except Exception as exc:
        print(f"[ERROR] Failed to download ffmpeg: {exc}")
        pause()
        sys.exit(1)

    print("Extracting ffmpeg ...", flush=True)
    # gyan.dev ships shared builds as .7z (BCJ2); Windows 10/11 ship a libarchive-
    # based tar.exe that can extract it, so no third-party tool is needed.
    tar = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
    try:
        if not tar.exists():
            raise RuntimeError(f"tar.exe not found at {tar}")
        if run([str(tar), "-xf", str(archive), "-C", str(dest_dir)]) != 0:
            raise RuntimeError("tar extraction returned an error")
    except Exception as exc:
        print(f"[ERROR] Failed to extract ffmpeg: {exc}")
        pause()
        sys.exit(1)
    finally:
        archive.unlink(missing_ok=True)

    # The archive extracts to bin\ffmpeg\ffmpeg-8.1.2-full_build-shared\bin\;
    # move everything (exes + DLLs) to a flat bin\ffmpeg\bin\ layout.
    bin_dir = dest_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    moved = False
    for sub in dest_dir.glob("ffmpeg-*"):
        src_bin = sub / "bin"
        if (src_bin / "ffmpeg.exe").exists():
            for src in src_bin.iterdir():
                if src.is_file():
                    shutil.move(str(src), str(bin_dir / src.name))
            shutil.rmtree(sub, ignore_errors=True)
            moved = True
            break

    if (
        not moved
        or not (bin_dir / "ffmpeg.exe").exists()
        or not list(bin_dir.glob("avcodec-*.dll"))
    ):
        print("[ERROR] No shared ffmpeg found after extracting.")
        pause()
        sys.exit(1)
    print("ffmpeg installed to the project bin\\ffmpeg directory.", flush=True)
    _register_ffmpeg_dll_dir(bin_dir)
    print()


def setup_config() -> None:
    if (ROOT / "kakure.toml").exists():
        print("[5/6] Detected kakure.toml, skipping generation.", flush=True)
    else:
        print("[5/6] Generating kakure.toml config file ...", flush=True)
        example = ROOT / "kakure.toml.example"
        if example.exists():
            shutil.copyfile(example, ROOT / "kakure.toml")
        else:
            print("[5/6] kakure.toml.example not found.", flush=True)


def finalize() -> None:
    print()
    print("=" * 60)
    print("Installation complete!")
    print()
    print("Usage:")
    print("1. Double-click start-kakure.bat every time to launch Kakure.")
    print("2. On first use, configure your OpenAI API Key (Settings page or edit kakure.toml).")
    print("3. The first run downloads the Whisper model (~3GB), please be patient.")
    print("=" * 60)
    print()
    choice = ask("Launch Kakure now? [Y/n] (Enter=launch)")
    if choice.lower() == "n":
        print("Installation complete. Double-click start-kakure.bat later to start.")
        pause()
        sys.exit(0)
    print("Launching Kakure ...", flush=True)
    launch = ROOT / "start-kakure.bat"
    if launch.exists():
        subprocess.call(["cmd", "/c", str(launch)], cwd=str(ROOT))
    else:
        print("[ERROR] start-kakure.bat not found.")
        pause()


def main() -> None:
    if os.name != "nt":
        print("This installer only supports Windows.")
        sys.exit(1)
    check_python_version()
    setup_venv()
    upgrade_pip()
    install_package()
    setup_ffmpeg()
    setup_config()
    finalize()


if __name__ == "__main__":
    main()

"""System detection — OS, arch, GPU, Python."""

import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemInfo:
    os: str          # Darwin | Linux | Windows
    arch: str        # arm64 | x86_64 | AMD64
    python_version: str
    apple_silicon: bool = False
    cuda: bool = False
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    homebrew: bool = False
    pip_cmd: str = "pip"

    @property
    def platform_key(self) -> str:
        if self.apple_silicon:
            return "apple_silicon"
        if self.cuda:
            return "cuda"
        return "cpu"

    @property
    def os_label(self) -> str:
        return {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}.get(self.os, self.os)

    @property
    def torch_index_url(self) -> Optional[str]:
        if self.cuda:
            v = self.cuda_version or "12.1"
            major = v.split(".")[0]
            minor = v.split(".")[1] if "." in v else "1"
            return f"https://download.pytorch.org/whl/cu{major}{minor.zfill(2)}"
        return None  # default PyPI torch is fine for macOS / CPU


def detect() -> SystemInfo:
    os_name = platform.system()
    arch = platform.machine()
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    pip_cmd = "pip3" if _cmd_exists("pip3") else "pip"

    info = SystemInfo(os=os_name, arch=arch, python_version=python_ver, pip_cmd=pip_cmd)

    if os_name == "Darwin" and arch == "arm64":
        info.apple_silicon = True

    # Homebrew
    info.homebrew = _cmd_exists("brew")

    # NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            info.cuda = True
            info.gpu_name = result.stdout.strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if info.cuda:
        try:
            r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            m = re.search(r"release (\d+\.\d+)", r.stdout)
            if m:
                info.cuda_version = m.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # fall back to nvidia-smi for CUDA version
            try:
                r2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                m2 = re.search(r"CUDA Version:\s*(\d+\.\d+)", r2.stdout)
                if m2:
                    info.cuda_version = m2.group(1)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    return info


def _cmd_exists(cmd: str) -> bool:
    try:
        subprocess.run(
            ["which" if platform.system() != "Windows" else "where", cmd],
            capture_output=True, timeout=3,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

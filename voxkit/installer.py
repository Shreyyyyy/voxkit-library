"""Core installation logic."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .detect import SystemInfo, _cmd_exists
from . import ui

console = Console()

# Correct HuggingFace repo IDs for mlx-whisper (with -mlx suffix)
_MLX_REPOS = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx-fp32",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# Package name → importable module name (for already-installed checks)
_PKG_TO_MODULE = {
    "edge-tts":      "edge_tts",
    "mlx-whisper":   "mlx_whisper",
    "faster-whisper":"faster_whisper",
    "TTS":           "TTS",
    "piper-tts":     "piper",
    "openai":        "openai",
    "anthropic":     "anthropic",
    "ollama":        "ollama",
    "groq":          "groq",
    "mistralai":     "mistralai",
    "cohere":        "cohere",
    "deepgram-sdk":  "deepgram",
    "assemblyai":    "assemblyai",
    "elevenlabs":    "elevenlabs",
    "llama-cpp-python": "llama_cpp",
    "transformers":  "transformers",
    "torch":         "torch",
    "torchaudio":    "torchaudio",
    "aiohttp":       "aiohttp",
    "pydub":         "pydub",
    "google-cloud-speech": "google.cloud.speech",
    "google-cloud-texttospeech": "google.cloud.texttospeech",
    "google-genai":  "google.genai",
}

_FUN_MSGS = [
    "Summoning the bits...",
    "Bribing the package gods...",
    "Downloading internet...",
    "Teaching robots to talk...",
    "Spinning up the hamsters...",
    "Making machine go brr...",
    "Waking up the neurons...",
    "Herding the bytes...",
]
_fun_idx = 0


def _next_fun() -> str:
    global _fun_idx
    msg = _FUN_MSGS[_fun_idx % len(_FUN_MSGS)]
    _fun_idx += 1
    return msg


# ── uv / pip bootstrap ───────────────────────────────────────────────────────

_uv_ready: bool | None = None   # cache: None=unchecked, True/False=result


def _ensure_uv() -> bool:
    """Return True if uv is available, installing it via pip if needed."""
    global _uv_ready
    if _uv_ready is not None:
        return _uv_ready

    if shutil.which("uv"):
        _uv_ready = True
        return True

    # Try Python module form (pip install uv puts the binary in venv/bin)
    try:
        import uv as _uv_mod  # noqa: F401
        _uv_ready = True
        return True
    except ImportError:
        pass

    # Bootstrap: install uv via pip once, then re-check
    ui.info("Installing [bold]uv[/bold] (fast package manager)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "uv", "--quiet"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0:
        ui.warn("Could not install uv — falling back to pip")
        _uv_ready = False
        return False

    # After installing, the binary lands in the same bin/ as the python interpreter
    uv_bin = Path(sys.executable).parent / "uv"
    if uv_bin.exists():
        import os
        os.environ["PATH"] = str(uv_bin.parent) + os.pathsep + os.environ.get("PATH", "")

    _uv_ready = bool(shutil.which("uv"))
    if _uv_ready:
        ui.success("uv installed")
    else:
        ui.warn("uv not found in PATH after install — falling back to pip")
    return _uv_ready


def _pip_cmd(index_url: str | None = None) -> list[str]:
    """Return the install command prefix, using uv when available."""
    if _ensure_uv():
        cmd = ["uv", "pip", "install", "--upgrade", "--quiet"]
    else:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--upgrade-strategy", "only-if-needed", "--quiet",
        ]
    if index_url:
        cmd += ["--index-url", index_url]
    return cmd


# ── pip ──────────────────────────────────────────────────────────────────────

def pip_install(packages: list[str], extra_flags: list[str] | None = None, dry_run: bool = False) -> bool:
    to_install = [p for p in packages if not _is_installed(p)]
    already    = [p for p in packages if _is_installed(p)]

    for pkg in already:
        ui.success(f"[dim]{_pkg_name(pkg)} already installed — skipping[/dim]")

    if not to_install:
        return True

    cmd = _pip_cmd() + to_install
    if extra_flags:
        cmd += extra_flags

    return _run_pip(cmd, label=" ".join(_pkg_name(p) for p in to_install[:2]), dry_run=dry_run)


def pip_install_torch_with_cuda(packages: list[str], index_url: str, dry_run: bool = False) -> bool:
    to_install = [p for p in packages if not _is_installed(p)]
    if not to_install:
        ui.success("[dim]torch/torchaudio already installed — skipping[/dim]")
        return True

    cmd = _pip_cmd(index_url=index_url) + to_install
    return _run_pip(cmd, label="torch (CUDA)", dry_run=dry_run)


# ── binary installs ───────────────────────────────────────────────────────────

def install_ollama(info: SystemInfo, dry_run: bool = False) -> bool:
    if info.os == "Darwin":
        if info.homebrew:
            ui.info("Using Homebrew to install Ollama...")
            return _run_streaming(["brew", "install", "ollama"], dry_run=dry_run)
        return _run_shell("curl -fsSL https://ollama.ai/install.sh | sh", dry_run=dry_run)
    if info.os == "Linux":
        return _run_shell("curl -fsSL https://ollama.ai/install.sh | sh", dry_run=dry_run)
    # Windows
    ui.warn("Automatic Ollama install not supported on Windows.")
    ui.info("Download manually from: https://ollama.ai/download/windows")
    return False


def pull_ollama_model(model: str, dry_run: bool = False) -> bool:
    ui.info(f"Pulling model [cyan]{model}[/cyan] via Ollama...")
    # ollama pull streams its own rich progress — let it go directly to terminal
    return _run_streaming(["ollama", "pull", model], dry_run=dry_run)


# ── STT installer ─────────────────────────────────────────────────────────────

def download_whisper_model(model_size: str, info: SystemInfo, dry_run: bool = False) -> bool:
    ui.info(f"Pre-caching Whisper [cyan]{model_size}[/cyan] model weights...")
    if dry_run:
        repo = _MLX_REPOS.get(model_size, _MLX_REPOS["base"])
        src = f"mlx-community/whisper-{model_size}" if info.apple_silicon else model_size
        ui.info(f"[dim](dry-run)[/dim] load {src}")
        return True
    try:
        if info.apple_silicon:
            import numpy as np
            import mlx_whisper
            repo = _MLX_REPOS.get(model_size, _MLX_REPOS["base"])
            silence = np.zeros(16000, dtype=np.float32)
            mlx_whisper.transcribe(silence, path_or_hf_repo=repo)
        else:
            from faster_whisper import WhisperModel
            WhisperModel(model_size, device="cpu", compute_type="int8")
        ui.success(f"Model [cyan]{model_size}[/cyan] cached.")
        return True
    except Exception as e:
        ui.error(f"Model cache failed: {e}")
        return False


def download_stt_model_if_needed(
    key: str, model_size: str, download_now: bool, info: SystemInfo, dry_run: bool = False
) -> None:
    if download_now and key == "whisper":
        download_whisper_model(model_size, info, dry_run=dry_run)


def install_stt(key: str, info: SystemInfo, model_size: str = "base", dry_run: bool = False) -> bool:
    from .registry.stt import STT_PROVIDERS

    provider = STT_PROVIDERS.get(key)
    if not provider:
        ui.error(f"Unknown STT provider: [bold]{key}[/bold]")
        return False

    variants = provider.get("variants", {})
    variant_key = _pick_variant(variants, info.platform_key)
    variant = variants.get(variant_key)
    if not variant:
        ui.error(f"No variant for platform {info.platform_key}")
        return False

    ui.section(f"Installing STT: {provider['emoji']} {provider['name']} — {variant['label']}")

    torch_pkgs = [p for p in variant["pip"] if p.startswith(("torch", "torchaudio"))]
    other_pkgs = [p for p in variant["pip"] if not p.startswith(("torch", "torchaudio"))]
    uses_cuda   = bool(variant.get("pip_flags")) and info.torch_index_url is not None

    ok = True
    if other_pkgs:
        ok = ok and pip_install(other_pkgs, dry_run=dry_run)
    if torch_pkgs:
        if uses_cuda and info.torch_index_url:
            ok = ok and pip_install_torch_with_cuda(torch_pkgs, info.torch_index_url, dry_run=dry_run)
        else:
            ok = ok and pip_install(torch_pkgs, dry_run=dry_run)

    _print_note(provider, variant_key)
    _print_api_key_hint(provider)
    return ok


# ── TTS installer ─────────────────────────────────────────────────────────────

def install_tts(key: str, info: SystemInfo, dry_run: bool = False) -> bool:
    from .registry.tts import TTS_PROVIDERS

    provider = TTS_PROVIDERS.get(key)
    if not provider:
        ui.error(f"Unknown TTS provider: [bold]{key}[/bold]")
        return False

    variants = provider.get("variants", {})
    variant_key = _pick_variant(variants, info.platform_key)
    variant = variants.get(variant_key)
    if not variant:
        ui.error(f"No variant for platform {info.platform_key}")
        return False

    ui.section(f"Installing TTS: {provider['emoji']} {provider['name']} — {variant['label']}")

    torch_pkgs = [p for p in variant["pip"] if p.startswith(("torch", "torchaudio"))]
    other_pkgs = [p for p in variant["pip"] if not p.startswith(("torch", "torchaudio"))]
    uses_cuda   = bool(variant.get("pip_flags")) and info.torch_index_url is not None

    ok = True
    if other_pkgs:
        ok = ok and pip_install(other_pkgs, dry_run=dry_run)
    if torch_pkgs:
        if uses_cuda and info.torch_index_url:
            ok = ok and pip_install_torch_with_cuda(torch_pkgs, info.torch_index_url, dry_run=dry_run)
        else:
            ok = ok and pip_install(torch_pkgs, dry_run=dry_run)

    _print_note(provider, variant_key)
    _print_api_key_hint(provider)
    return ok


# ── LLM installer ─────────────────────────────────────────────────────────────

def install_llm(key: str, info: SystemInfo, model: Optional[str] = None, dry_run: bool = False) -> bool:
    from .registry.llm import LLM_PROVIDERS

    provider = LLM_PROVIDERS.get(key)
    if not provider:
        ui.error(f"Unknown LLM provider: [bold]{key}[/bold]")
        return False

    ui.section(f"Installing LLM: {provider['emoji']} {provider['name']}")

    install_type = provider.get("install_type", "pip")
    ok = True

    if install_type in ("binary", "pip+binary"):
        if key == "ollama":
            if not _cmd_exists("ollama"):
                ok = ok and install_ollama(info, dry_run=dry_run)
            else:
                ui.success("Ollama binary already installed.")
        client_pip = provider.get("pip", [])
        if client_pip:
            ok = ok and pip_install(client_pip, dry_run=dry_run)
        if ok and model:
            ok = ok and pull_ollama_model(model, dry_run=dry_run)

    elif install_type == "pip":
        pip_variants = provider.get("pip_variants", {})
        pkgs = pip_variants.get(info.platform_key) or provider.get("pip", [])
        if pkgs:
            ok = ok and pip_install(pkgs, dry_run=dry_run)

    _print_api_key_hint(provider)
    return ok


# ── internal helpers ──────────────────────────────────────────────────────────

def _pick_variant(variants: dict, platform_key: str) -> str:
    if platform_key in variants:
        return platform_key
    if "all" in variants:
        return "all"
    for fallback in ("cpu", "cuda", "apple_silicon"):
        if fallback in variants:
            return fallback
    return next(iter(variants), "")


def _is_installed(package_spec: str) -> bool:
    """Return True if the package's module is already importable."""
    pkg = _pkg_name(package_spec)
    module = _PKG_TO_MODULE.get(pkg, pkg.replace("-", "_").split("[")[0])
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _pkg_name(spec: str) -> str:
    """Strip version specifiers: 'edge-tts>=6.1.0' → 'edge-tts'."""
    for sep in (">=", "<=", "==", "!=", ">", "<", "~="):
        if sep in spec:
            return spec.split(sep)[0].strip()
    return spec.strip()


def _run_pip(cmd: list[str], label: str = "", dry_run: bool = False) -> bool:
    """Run a pip command — captures all output, shows errors cleanly on failure."""
    if dry_run:
        ui.info(f"[dim](dry-run)[/dim] {' '.join(cmd)}")
        return True

    display = label or _pkg_name(cmd[-1]) if cmd else "packages"

    with Progress(
        SpinnerColumn(),
        TextColumn(f"  [cyan]{_next_fun()}[/cyan]  [dim]{display}[/dim]"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("", total=None)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → stdout for unified capture
            text=True,
        )

    if result.returncode == 0:
        ui.success(f"Installed [bold]{display}[/bold]")
        return True

    # ── show the actual pip error so nothing is silent ────────────────────────
    ui.error(f"pip failed (exit {result.returncode}) — [bold]{display}[/bold]")
    output = result.stdout or ""
    # Extract the most useful lines (ERROR / note / Caused by)
    lines = output.splitlines()
    useful = [
        l for l in lines
        if any(kw in l for kw in ("ERROR", "error", "Cannot", "No matching", "Caused by", "Note:", "required"))
    ]
    shown = useful[-12:] if useful else lines[-12:]
    for line in shown:
        console.print(f"    [dim red]{line}[/dim red]")
    console.print(f"  [dim]Run with --verbose to see full pip output.[/dim]")
    return False


def _run_streaming(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a command that streams its own output (ollama pull, brew, curl)."""
    if dry_run:
        ui.info(f"[dim](dry-run)[/dim] {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd)
    return result.returncode == 0


def _run_shell(cmd: str, dry_run: bool = False) -> bool:
    if dry_run:
        ui.info(f"[dim](dry-run)[/dim] {cmd}")
        return True
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def _print_note(provider: dict, variant_key: str) -> None:
    notes = provider.get("notes", {})
    note = notes.get(variant_key) or notes.get("all")
    if note:
        ui.info(f"[dim]{note}[/dim]")


def _print_api_key_hint(provider: dict) -> None:
    if provider.get("requires_api_key"):
        key_name = provider.get("api_key_name", "API_KEY")
        link = provider.get("api_link", "")
        console.print(f"\n  [bold yellow]API key required[/bold yellow] — set [cyan]{key_name}[/cyan] in your .env")
        if link:
            console.print(f"  Get one at: [link={link}]{link}[/link]")

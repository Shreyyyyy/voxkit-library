"""Core installation logic."""

import subprocess
import sys
from typing import Optional

# MLX Whisper HuggingFace repo per model size
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v3": "mlx-community/whisper-large-v3",
}

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .detect import SystemInfo, _cmd_exists
from . import ui

console = Console()

_FUN_INSTALLS = [
    "Summoning the bits...",
    "Bribing the package gods...",
    "Downloading internet...",
    "Teaching robots to talk...",
    "Spinning up the hamsters...",
    "Making machine go brr...",
]

_FUN_IDX = 0


def _next_fun() -> str:
    global _FUN_IDX
    msg = _FUN_INSTALLS[_FUN_IDX % len(_FUN_INSTALLS)]
    _FUN_IDX += 1
    return msg


# ── pip ─────────────────────────────────────────────────────────────────────

def pip_install(packages: list[str], extra_flags: list[str] | None = None, dry_run: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    if extra_flags:
        cmd += extra_flags
    return _run(cmd, dry_run=dry_run)


def pip_install_torch_with_cuda(packages: list[str], index_url: str, dry_run: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages + [
        "--index-url", index_url
    ]
    return _run(cmd, dry_run=dry_run)


# ── binary (Ollama) ──────────────────────────────────────────────────────────

def install_ollama(info: SystemInfo, dry_run: bool = False) -> bool:
    if info.os == "Darwin":
        if info.homebrew:
            ui.info("Using Homebrew to install Ollama...")
            return _run(["brew", "install", "ollama"], dry_run=dry_run)
        else:
            return _run_shell("curl -fsSL https://ollama.ai/install.sh | sh", dry_run=dry_run)
    elif info.os == "Linux":
        return _run_shell("curl -fsSL https://ollama.ai/install.sh | sh", dry_run=dry_run)
    else:  # Windows
        ui.warn("Automatic install not supported on Windows.")
        ui.info("Please download the installer from: https://ollama.ai/download/windows")
        return False


def pull_ollama_model(model: str, dry_run: bool = False) -> bool:
    ui.info(f"Pulling model [cyan]{model}[/cyan] via Ollama...")
    return _run(["ollama", "pull", model], dry_run=dry_run)


# ── STT installer ────────────────────────────────────────────────────────────

def download_whisper_model(model_size: str, info: SystemInfo, dry_run: bool = False) -> bool:
    """Pre-cache a Whisper model by loading it once."""
    ui.info(f"Downloading Whisper [cyan]{model_size}[/cyan] model weights (this may take a few minutes)...")
    if dry_run:
        if info.apple_silicon:
            ui.info(f"[dim](dry-run)[/dim] mlx_whisper.transcribe(silence, path_or_hf_repo='{_MLX_REPOS[model_size]}')")
        else:
            ui.info(f"[dim](dry-run)[/dim] WhisperModel('{model_size}', compute_type='int8')")
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
        ui.success(f"Model [cyan]{model_size}[/cyan] cached locally.")
        return True
    except Exception as e:
        ui.error(f"Model download failed: {e}")
        return False


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
        ui.error(f"No variant found for {info.platform_key}")
        return False

    ui.section(f"Installing STT: {provider['emoji']} {provider['name']} — {variant['label']}")

    # Split torch packages from regular packages for CUDA index-url handling
    torch_pkgs = [p for p in variant["pip"] if p.startswith(("torch", "torchaudio"))]
    other_pkgs = [p for p in variant["pip"] if not p.startswith(("torch", "torchaudio"))]
    pip_flags_for = variant.get("pip_flags_for", [])
    uses_cuda_index = bool(variant.get("pip_flags")) and info.torch_index_url is not None

    ok = True

    if other_pkgs:
        ui.info(f"pip install {' '.join(other_pkgs)}")
        ok = ok and pip_install(other_pkgs, dry_run=dry_run)

    if torch_pkgs:
        if uses_cuda_index and info.torch_index_url:
            ui.info(f"pip install {' '.join(torch_pkgs)} --index-url {info.torch_index_url}")
            ok = ok and pip_install_torch_with_cuda(torch_pkgs, info.torch_index_url, dry_run=dry_run)
        else:
            ui.info(f"pip install {' '.join(torch_pkgs)}")
            ok = ok and pip_install(torch_pkgs, dry_run=dry_run)

    _print_note(provider, variant_key)
    _print_api_key_hint(provider)
    return ok


def download_stt_model_if_needed(key: str, model_size: str, download_now: bool, info: SystemInfo, dry_run: bool = False) -> None:
    """Called after install_stt when user wants to pre-cache the model."""
    if not download_now or key not in ("whisper",):
        return
    if not dry_run:
        ui.info(f"Pre-caching model [cyan]{model_size}[/cyan]...")
    download_whisper_model(model_size, info, dry_run=dry_run)


# ── TTS installer ────────────────────────────────────────────────────────────

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
        ui.error(f"No variant found for {info.platform_key}")
        return False

    ui.section(f"Installing TTS: {provider['emoji']} {provider['name']} — {variant['label']}")

    torch_pkgs = [p for p in variant["pip"] if p.startswith(("torch", "torchaudio"))]
    other_pkgs = [p for p in variant["pip"] if not p.startswith(("torch", "torchaudio"))]
    uses_cuda_index = bool(variant.get("pip_flags")) and info.torch_index_url is not None

    ok = True
    if other_pkgs:
        ok = ok and pip_install(other_pkgs, dry_run=dry_run)
    if torch_pkgs:
        if uses_cuda_index and info.torch_index_url:
            ok = ok and pip_install_torch_with_cuda(torch_pkgs, info.torch_index_url, dry_run=dry_run)
        else:
            ok = ok and pip_install(torch_pkgs, dry_run=dry_run)

    _print_note(provider, variant_key)
    _print_api_key_hint(provider)
    return ok


# ── LLM installer ────────────────────────────────────────────────────────────

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
        # install Python client
        client_pip = provider.get("pip", [])
        if client_pip:
            ok = ok and pip_install(client_pip, dry_run=dry_run)

        if ok and model:
            ok = ok and pull_ollama_model(model, dry_run=dry_run)

    elif install_type == "pip":
        # handle per-platform pip variants (e.g. llama-cpp-python[metal])
        pip_variants = provider.get("pip_variants", {})
        pkgs = pip_variants.get(info.platform_key) or provider.get("pip", [])
        if pkgs:
            ok = ok and pip_install(pkgs, dry_run=dry_run)

    _print_api_key_hint(provider)
    return ok


# ── helpers ──────────────────────────────────────────────────────────────────

def _pick_variant(variants: dict, platform_key: str) -> str:
    if platform_key in variants:
        return platform_key
    if "all" in variants:
        return "all"
    # fallback: cpu → cuda → apple_silicon
    fallbacks = ["cpu", "cuda", "apple_silicon"]
    for f in fallbacks:
        if f in variants:
            return f
    return next(iter(variants), "")


def _run(cmd: list[str], dry_run: bool = False) -> bool:
    if dry_run:
        ui.info(f"[dim](dry-run)[/dim] {' '.join(cmd)}")
        return True
    with Progress(
        SpinnerColumn(),
        TextColumn(f"  [cyan]{_next_fun()}[/cyan]"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("", total=None)
        result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        ui.success(f"Done: [bold]{' '.join(cmd[:4])}...[/bold]")
        return True
    else:
        ui.error(f"Failed (exit {result.returncode}): {' '.join(cmd[:4])}")
        return False


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
        console.print(
            f"\n  [bold yellow]API key required[/bold yellow] — set [cyan]{key_name}[/cyan] in your .env"
        )
        if link:
            console.print(f"  Get one at: [link={link}]{link}[/link]")

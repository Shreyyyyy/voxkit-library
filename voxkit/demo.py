"""Demo payload — runs a quick end-to-end voice stack test."""

import asyncio
import importlib
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from . import ui

console = Console()

DEMO_TEXT = "Hello! This is voxkit speaking. Your voice stack is alive and ready to rock."
DEMO_AUDIO = Path(tempfile.gettempdir()) / "voxkit_demo.mp3"

# Correct HuggingFace repo IDs for mlx-whisper (with -mlx suffix)
_MLX_REPOS = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx-fp32",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


def run(whisper_model: str = "base") -> None:
    """Run TTS → STT → LLM demo and print a summary table."""
    console.print(
        Panel(
            "[bold white]voxkit demo[/bold white]\n"
            "[dim]Testing your installed voice stack end-to-end...[/dim]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    results: dict[str, tuple[bool, str]] = {}

    # ── 1. TTS ──────────────────────────────────────────────────────────────
    ui.section("Step 1 — Text-to-Speech")
    tts_ok, tts_msg = _demo_tts()
    results["TTS"] = (tts_ok, tts_msg)

    # ── 2. STT ──────────────────────────────────────────────────────────────
    ui.section("Step 2 — Speech-to-Text")
    if tts_ok and DEMO_AUDIO.exists():
        stt_ok, stt_msg = _demo_stt(whisper_model)
    else:
        stt_ok, stt_msg = False, "Skipped — TTS produced no audio"
        ui.warn("Skipping STT: no audio file from TTS step.")
    results["STT"] = (stt_ok, stt_msg)

    # ── 3. LLM ──────────────────────────────────────────────────────────────
    ui.section("Step 3 — Language Model")
    llm_ok, llm_msg = _demo_llm()
    results["LLM"] = (llm_ok, llm_msg)

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print()
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
    t.add_column("Component", style="bold")
    t.add_column("Status", justify="center")
    t.add_column("Details")

    for component, (ok, msg) in results.items():
        status = "[bold green]✓ pass[/bold green]" if ok else "[bold red]✗ fail[/bold red]"
        t.add_row(component, status, msg)

    console.print(t)
    console.print()

    all_ok = all(ok for ok, _ in results.values())
    if all_ok:
        console.print(
            Panel(
                "[bold green]All systems go! 🎉[/bold green]\n"
                "[dim]Your voice stack passed the demo. Time to build something cool.[/dim]",
                border_style="green",
                expand=False,
            )
        )
    else:
        failed = [c for c, (ok, _) in results.items() if not ok]
        console.print(
            Panel(
                f"[bold yellow]{', '.join(failed)} not working yet.[/bold yellow]\n"
                f"[dim]Run [cyan]voxkit install {failed[0].lower()} <provider>[/cyan] to fix it.[/dim]",
                border_style="yellow",
                expand=False,
            )
        )


# ── TTS ──────────────────────────────────────────────────────────────────────

def _demo_tts() -> tuple[bool, str]:
    # edge-tts
    if _has("edge_tts"):
        ui.info(f'Synthesising: [dim]"{DEMO_TEXT}"[/dim]')
        ui.info("Engine: [cyan]edge-tts[/cyan] (en-US-JennyNeural)")
        try:
            asyncio.run(_edge_tts(DEMO_TEXT, str(DEMO_AUDIO)))
            size_kb = DEMO_AUDIO.stat().st_size // 1024
            ui.success(f"Audio saved → {DEMO_AUDIO}  ({size_kb} KB)")
            return True, f"edge-tts → {DEMO_AUDIO.name}"
        except Exception as e:
            ui.error(str(e))

    # Coqui TTS
    if _has("TTS"):
        try:
            from TTS.api import TTS as CoquiTTS
            ui.info("Engine: [cyan]Coqui TTS[/cyan]")
            tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            wav_path = str(DEMO_AUDIO).replace(".mp3", ".wav")
            tts.tts_to_file(text=DEMO_TEXT, file_path=wav_path)
            DEMO_AUDIO.__class__(wav_path).rename(DEMO_AUDIO)
            ui.success(f"Audio saved → {DEMO_AUDIO.name}")
            return True, f"coqui → {DEMO_AUDIO.name}"
        except Exception as e:
            ui.error(str(e))

    ui.warn("No TTS engine found. Install one:")
    ui.info("  [cyan]voxkit install tts edge[/cyan]   ← free, easiest")
    return False, "No TTS installed — run: voxkit install tts edge"


async def _edge_tts(text: str, path: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await communicate.save(path)


# ── STT ──────────────────────────────────────────────────────────────────────

def _demo_stt(model_size: str = "base") -> tuple[bool, str]:
    audio = str(DEMO_AUDIO)

    # mlx-whisper (Apple Silicon)
    if _has("mlx_whisper"):
        try:
            import mlx_whisper
            repo = _MLX_REPOS.get(model_size, _MLX_REPOS["base"])
            ui.info(f"Engine: [cyan]mlx-whisper[/cyan] · model: [dim]{model_size}[/dim]")
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo)
            text = result.get("text", "").strip()
            ui.success(f'Transcribed: "[italic]{text}[/italic]"')
            return True, f'"{text}"'
        except Exception as e:
            ui.error(str(e))

    # faster-whisper (CUDA / CPU)
    if _has("faster_whisper"):
        try:
            from faster_whisper import WhisperModel
            ui.info(f"Engine: [cyan]faster-whisper[/cyan] · model: [dim]{model_size}[/dim]")
            model = WhisperModel(model_size, compute_type="int8")
            segments, _ = model.transcribe(audio)
            text = " ".join(s.text for s in segments).strip()
            ui.success(f'Transcribed: "[italic]{text}[/italic]"')
            return True, f'"{text}"'
        except Exception as e:
            ui.error(str(e))

    ui.warn("No STT engine found. Install one:")
    ui.info("  [cyan]voxkit install stt whisper[/cyan]")
    return False, "No STT installed — run: voxkit install stt whisper"


# ── LLM ──────────────────────────────────────────────────────────────────────

def _demo_llm() -> tuple[bool, str]:
    prompt = "Say hello in exactly one short sentence."

    # Ollama (local)
    if _has("ollama") and _cmd_ok("ollama"):
        try:
            import ollama as _ollama
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            lines = [l for l in r.stdout.strip().splitlines()[1:] if l.strip()]
            if lines:
                model = lines[0].split()[0]
                ui.info(f"Engine: [cyan]ollama[/cyan] · model: [dim]{model}[/dim]")
                resp = _ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                text = resp.message.content.strip()
                ui.success(f'Response: "[italic]{text}[/italic]"')
                return True, f'ollama:{model} → "{text[:60]}"'
        except Exception as e:
            ui.error(str(e))

    # OpenAI
    if _has("openai"):
        import os
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                ui.info("Engine: [cyan]openai[/cyan] · gpt-4o-mini")
                client = openai.OpenAI()
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                )
                text = r.choices[0].message.content.strip()
                ui.success(f'Response: "[italic]{text}[/italic]"')
                return True, f'openai → "{text[:60]}"'
            except Exception as e:
                ui.error(str(e))

    # Anthropic
    if _has("anthropic"):
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                ui.info("Engine: [cyan]anthropic[/cyan] · claude-haiku")
                client = anthropic.Anthropic()
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=60,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text.strip()
                ui.success(f'Response: "[italic]{text}[/italic]"')
                return True, f'anthropic → "{text[:60]}"'
            except Exception as e:
                ui.error(str(e))

    ui.warn("No LLM configured. Install one:")
    ui.info("  [cyan]voxkit install llm ollama[/cyan]   ← local, no API key")
    return False, "No LLM configured — run: voxkit install llm ollama"


# ── utils ─────────────────────────────────────────────────────────────────────

def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _cmd_ok(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"], capture_output=True, timeout=3)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

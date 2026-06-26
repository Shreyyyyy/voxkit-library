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

DEMO_QUESTION = "What is the capital of France? Give a one-sentence answer."
DEMO_QUESTION_AUDIO = Path(tempfile.gettempdir()) / "voxkit_demo_question.mp3"
DEMO_RESPONSE_AUDIO = Path(tempfile.gettempdir()) / "voxkit_demo_response.mp3"

# Correct HuggingFace repo IDs for mlx-whisper (with -mlx suffix)
_MLX_REPOS = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx-fp32",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


def run(whisper_model: str = "base") -> None:
    """Run STT → LLM → TTS demo pipeline and print a summary table."""
    console.print(
        Panel(
            "[bold white]voxkit demo[/bold white]\n"
            "[dim]Testing your installed voice stack end-to-end...[/dim]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    # Silently generate question audio to simulate user speaking
    console.print(
        f'[dim]Simulating user speech: "[italic]{DEMO_QUESTION}[/italic]"[/dim]'
    )
    if not _prepare_question_audio():
        console.print(
            Panel(
                "[bold red]Cannot start demo — no TTS engine installed.[/bold red]\n\n"
                "The demo needs TTS to synthesise a sample question that STT then transcribes.\n\n"
                "  Fix (free, easiest):\n"
                "    [cyan]voxkit install tts edge[/cyan]\n"
                "      → edge-tts works on all platforms, no API key needed\n\n"
                "  Then re-run:  [cyan]voxkit demo[/cyan]",
                border_style="red",
                expand=False,
            )
        )
        return
    console.print()

    results: dict[str, tuple[bool, str]] = {}

    # ── 1. STT ──────────────────────────────────────────────────────────────
    ui.section("Step 1 — Speech-to-Text")
    stt_ok, stt_msg, transcription = _demo_stt(whisper_model)
    results["STT"] = (stt_ok, stt_msg)

    # ── 2. LLM ──────────────────────────────────────────────────────────────
    ui.section("Step 2 — Language Model")
    if stt_ok and transcription:
        llm_ok, llm_msg, llm_response = _demo_llm(transcription)
    else:
        llm_ok, llm_msg, llm_response = False, "Skipped — STT produced no text", ""
        ui.warn("Skipping LLM: no transcription from STT step.")
    results["LLM"] = (llm_ok, llm_msg)

    # ── 3. TTS ──────────────────────────────────────────────────────────────
    ui.section("Step 3 — Text-to-Speech")
    if llm_ok and llm_response:
        tts_ok, tts_msg = _demo_tts(llm_response)
    else:
        tts_ok, tts_msg = False, "Skipped — LLM produced no response"
        ui.warn("Skipping TTS: no response from LLM step.")
    results["TTS"] = (tts_ok, tts_msg)

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


# ── Setup: generate question audio ────────────────────────────────────────────

def _prepare_question_audio() -> bool:
    """Synthesise the demo question to audio so STT has something to transcribe."""
    if _has("edge_tts"):
        try:
            asyncio.run(_edge_tts(DEMO_QUESTION, str(DEMO_QUESTION_AUDIO)))
            return True
        except Exception:
            pass

    if _has("TTS"):
        try:
            from TTS.api import TTS as CoquiTTS
            tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            wav_path = str(DEMO_QUESTION_AUDIO).replace(".mp3", ".wav")
            tts.tts_to_file(text=DEMO_QUESTION, file_path=wav_path)
            Path(wav_path).rename(DEMO_QUESTION_AUDIO)
            return True
        except Exception:
            pass

    return False


# ── STT ──────────────────────────────────────────────────────────────────────

def _demo_stt(model_size: str = "base") -> tuple[bool, str, str]:
    audio = str(DEMO_QUESTION_AUDIO)

    # mlx-whisper (Apple Silicon)
    if _has("mlx_whisper"):
        try:
            import mlx_whisper
            repo = _MLX_REPOS.get(model_size, _MLX_REPOS["base"])
            ui.info(f"Engine: [cyan]mlx-whisper[/cyan] · model: [dim]{model_size}[/dim]")
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo)
            text = result.get("text", "").strip()
            ui.success(f'Transcribed: "[italic]{text}[/italic]"')
            return True, f'"{text}"', text
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
            return True, f'"{text}"', text
        except Exception as e:
            ui.error(str(e))

    console.print(
        Panel(
            "[bold yellow]No STT engine installed.[/bold yellow]\n\n"
            "STT (Speech-to-Text) transcribes the user's speech into text.\n"
            "Without it, the pipeline cannot start — the LLM has nothing to read.\n\n"
            "  Fix (local, works offline):\n"
            "    [cyan]voxkit install stt whisper[/cyan]\n"
            "      → Apple Silicon: mlx-whisper (Metal GPU, fast)\n"
            "      → Linux / Windows: faster-whisper (CPU or CUDA)\n\n"
            "  Fix (cloud, no GPU needed):\n"
            "    [cyan]voxkit install stt deepgram[/cyan]    then set [dim]DEEPGRAM_API_KEY[/dim]\n"
            "    [cyan]voxkit install stt assemblyai[/cyan]  then set [dim]ASSEMBLYAI_API_KEY[/dim]",
            border_style="yellow",
            expand=False,
        )
    )
    return False, "No STT — run: voxkit install stt whisper", ""


# ── LLM ──────────────────────────────────────────────────────────────────────

def _demo_llm(prompt: str) -> tuple[bool, str, str]:
    ui.info(f'Prompt from STT: [dim]"{prompt}"[/dim]')

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
                return True, f'ollama:{model} → "{text[:60]}"', text
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
                return True, f'openai → "{text[:60]}"', text
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
                return True, f'anthropic → "{text[:60]}"', text
            except Exception as e:
                ui.error(str(e))

    console.print(
        Panel(
            "[bold yellow]No LLM configured.[/bold yellow]\n\n"
            "The LLM reads the transcribed text and generates a response.\n"
            "Without it, STT output has nowhere to go.\n\n"
            "  Fix (local, no API key, runs fully offline after setup):\n"
            "    [cyan]voxkit install llm ollama[/cyan]\n"
            "      → Pulls Llama / Mistral / Phi / Gemma to run on your machine\n\n"
            "  Fix (cloud API, no GPU needed):\n"
            "    [cyan]voxkit install llm openai[/cyan]     then set [dim]OPENAI_API_KEY[/dim]\n"
            "    [cyan]voxkit install llm anthropic[/cyan]  then set [dim]ANTHROPIC_API_KEY[/dim]\n"
            "    [cyan]voxkit install llm google[/cyan]     then set [dim]GOOGLE_API_KEY[/dim]",
            border_style="yellow",
            expand=False,
        )
    )
    return False, "No LLM — run: voxkit install llm ollama", ""


# ── TTS ──────────────────────────────────────────────────────────────────────

def _demo_tts(text: str) -> tuple[bool, str]:
    ui.info(f'Synthesising LLM response: [dim]"{text}"[/dim]')

    # edge-tts
    if _has("edge_tts"):
        ui.info("Engine: [cyan]edge-tts[/cyan] (en-US-JennyNeural)")
        try:
            asyncio.run(_edge_tts(text, str(DEMO_RESPONSE_AUDIO)))
            size_kb = DEMO_RESPONSE_AUDIO.stat().st_size // 1024
            ui.success(f"Audio saved → {DEMO_RESPONSE_AUDIO}  ({size_kb} KB)")
            return True, f"edge-tts → {DEMO_RESPONSE_AUDIO.name}"
        except Exception as e:
            ui.error(str(e))

    # Coqui TTS
    if _has("TTS"):
        try:
            from TTS.api import TTS as CoquiTTS
            ui.info("Engine: [cyan]Coqui TTS[/cyan]")
            tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            wav_path = str(DEMO_RESPONSE_AUDIO).replace(".mp3", ".wav")
            tts.tts_to_file(text=text, file_path=wav_path)
            Path(wav_path).rename(DEMO_RESPONSE_AUDIO)
            ui.success(f"Audio saved → {DEMO_RESPONSE_AUDIO.name}")
            return True, f"coqui → {DEMO_RESPONSE_AUDIO.name}"
        except Exception as e:
            ui.error(str(e))

    console.print(
        Panel(
            "[bold yellow]No TTS engine installed.[/bold yellow]\n\n"
            "TTS (Text-to-Speech) converts the LLM's response into audio the user hears.\n"
            "Without it, the pipeline produces text but the user never hears anything.\n\n"
            "  Fix (free, easiest, no API key):\n"
            "    [cyan]voxkit install tts edge[/cyan]\n"
            "      → edge-tts · 400+ voices · works on all platforms\n\n"
            "  Fix (local, higher quality):\n"
            "    [cyan]voxkit install tts piper[/cyan]   ← fast, fully offline\n"
            "    [cyan]voxkit install tts coqui[/cyan]   ← voice cloning, 17 languages\n\n"
            "  Fix (cloud, most realistic):\n"
            "    [cyan]voxkit install tts elevenlabs[/cyan]  then set [dim]ELEVENLABS_API_KEY[/dim]",
            border_style="yellow",
            expand=False,
        )
    )
    return False, "No TTS — run: voxkit install tts edge"


async def _edge_tts(text: str, path: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await communicate.save(path)


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

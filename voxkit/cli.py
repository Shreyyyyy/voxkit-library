"""voxkit CLI — playful voice stack installer."""

import sys
from typing import Optional

import click
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

from .detect import detect as _detect_system
from .registry import STT_PROVIDERS, TTS_PROVIDERS, LLM_PROVIDERS
from . import ui, installer

console = Console()

# ── Edge TTS voice presets per language family ────────────────────────────────
_EDGE_VOICE_PRESETS = {
    "English (US / UK / AU)": {
        "voices": [
            ("en-US-JennyNeural", "Jenny — warm, conversational (US)"),
            ("en-US-GuyNeural", "Guy — confident, clear (US)"),
            ("en-GB-SoniaNeural", "Sonia — polished, British"),
            ("en-AU-NatashaNeural", "Natasha — friendly, Australian"),
        ],
        "default": "en-US-JennyNeural",
    },
    "Indic (Hindi, Tamil, Telugu, Kannada, Bengali...)": {
        "voices": [
            ("hi-IN-SwaraNeural", "Swara — Hindi, natural female"),
            ("hi-IN-MadhurNeural", "Madhur — Hindi, clear male"),
            ("ta-IN-PallaviNeural", "Pallavi — Tamil, female"),
            ("te-IN-ShrutiNeural", "Shruti — Telugu, female"),
        ],
        "default": "hi-IN-SwaraNeural",
    },
    "European (Spanish, French, German, Italian...)": {
        "voices": [
            ("es-ES-ElviraNeural", "Elvira — Spanish (Spain), female"),
            ("fr-FR-DeniseNeural", "Denise — French, female"),
            ("de-DE-KatjaNeural", "Katja — German, female"),
            ("it-IT-ElsaNeural", "Elsa — Italian, female"),
        ],
        "default": "es-ES-ElviraNeural",
    },
    "Asian (Chinese, Japanese, Korean...)": {
        "voices": [
            ("zh-CN-XiaoxiaoNeural", "Xiaoxiao — Mandarin, female"),
            ("zh-CN-YunxiNeural", "Yunxi — Mandarin, male"),
            ("ja-JP-NanamiNeural", "Nanami — Japanese, female"),
            ("ko-KR-SunHiNeural", "SunHi — Korean, female"),
        ],
        "default": "zh-CN-XiaoxiaoNeural",
    },
}

# ── Whisper model sizes ───────────────────────────────────────────────────────
_WHISPER_MODELS = [
    ("tiny",     "39M",   "fastest · good for demos · lower accuracy"),
    ("base",     "74M",   "balanced · recommended for most use cases"),
    ("small",    "244M",  "better accuracy · still fast"),
    ("medium",   "769M",  "great accuracy · needs 4GB+ RAM"),
    ("large-v3", "1.5B",  "best accuracy · needs 8GB+ RAM"),
]


# ── main entry point ──────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """voxkit — install STT, TTS, and LLM engines the smart way."""
    if ctx.invoked_subcommand is None:
        ui.print_banner()
        info = _detect_system()
        ui.print_system_panel(info)
        console.print("[dim]New here? Run [bold cyan]voxkit help[/bold cyan] to see every command.[/dim]\n")
        _print_quick_ref()


def _print_quick_ref() -> None:
    rows = [
        ("[bold cyan]voxkit help[/bold cyan]",          "Full command reference with examples"),
        ("[bold cyan]voxkit wizard[/bold cyan]",         "Interactive setup — pick STT + TTS + LLM"),
        ("[bold cyan]voxkit detect[/bold cyan]",         "Show what your system supports"),
        ("[bold cyan]voxkit list all[/bold cyan]",       "Browse all providers"),
        ("[bold cyan]voxkit install stt whisper[/bold cyan]", "Install Whisper STT"),
        ("[bold cyan]voxkit install tts edge[/bold cyan]",    "Install Edge TTS (free)"),
        ("[bold cyan]voxkit install llm ollama[/bold cyan]",  "Install Ollama (local LLM)"),
        ("[bold cyan]voxkit demo[/bold cyan]",           "Test your installed stack"),
        ("[bold cyan]voxkit doctor[/bold cyan]",         "Check what's already installed"),
    ]
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("Command", no_wrap=True)
    t.add_column("Description", style="dim")
    for cmd, desc in rows:
        t.add_row(cmd, desc)
    console.print(t)
    console.print()


# ── help ──────────────────────────────────────────────────────────────────────

@main.command(name="help")
def show_help() -> None:
    """Full command reference — setup, install, test, publish."""
    ui.print_banner()

    def _row(cmd: str, desc: str) -> str:
        return f"  [bold cyan]{cmd:<42}[/bold cyan][dim]{desc}[/dim]"

    sections = [
        ("QUICK START", [
            ("pip install voxkit",                   "Install voxkit"),
            ("voxkit wizard",                         "Interactive full setup  ← best for new users"),
            ("voxkit demo",                           "Test your installed voice stack"),
        ]),
        ("DISCOVER YOUR SYSTEM", [
            ("voxkit detect",                         "Show OS, GPU, recommended stack"),
            ("voxkit list [stt|tts|llm|all]",        "Browse all available providers"),
            ("voxkit doctor",                         "Check what's already installed"),
        ]),
        ("INSTALL — Speech-to-Text (STT)", [
            ("voxkit install stt whisper",            "Auto-selects mlx / faster-whisper / CPU"),
            ("voxkit install stt seamless",           "Meta SeamlessM4T — great for Indic"),
            ("voxkit install stt deepgram",           "Deepgram Nova-2 API — ultra-fast"),
            ("voxkit install stt assemblyai",         "AssemblyAI API — great accuracy"),
            ("voxkit install stt google",             "Google Cloud Speech API"),
            ("voxkit install stt openai-whisper",     "OpenAI Whisper API — no local GPU needed"),
        ]),
        ("INSTALL — Text-to-Speech (TTS)", [
            ("voxkit install tts edge",               "Edge TTS — free, 400+ voices, asks language"),
            ("voxkit install tts sarvam",             "Sarvam AI — best Indic language TTS"),
            ("voxkit install tts piper",              "Piper — fast local TTS, ~100ms latency"),
            ("voxkit install tts coqui",              "Coqui XTTS — voice cloning, 17 languages"),
            ("voxkit install tts elevenlabs",         "ElevenLabs API — most realistic voices"),
            ("voxkit install tts openai-tts",         "OpenAI TTS-1 / TTS-1-HD API"),
            ("voxkit install tts google-tts",         "Google Cloud TTS — WaveNet / Journey"),
        ]),
        ("INSTALL — Language Models (LLM)", [
            ("voxkit install llm ollama",             "Local LLMs — Llama, Mistral, Phi, Gemma..."),
            ("voxkit install llm openai",             "OpenAI GPT-4o / o1 API"),
            ("voxkit install llm anthropic",          "Anthropic Claude 4 API"),
            ("voxkit install llm google",             "Google Gemini 2.5 API"),
            ("voxkit install llm groq",               "Groq — ultra-fast LPU inference"),
            ("voxkit install llm mistral",            "Mistral AI API"),
            ("voxkit install llm cohere",             "Cohere Command R+ — RAG-optimized"),
            ("voxkit install llm llama-cpp",          "llama.cpp — GGUF models locally"),
        ]),
        ("FLAGS", [
            ("--dry-run",                             "Print install commands without executing"),
            ("-m / --model llama3.2",                 "Specify Ollama model to pull"),
        ]),
    ]

    body_lines = []
    for title, rows in sections:
        body_lines.append(f"\n  [bold white]{title}[/bold white]")
        body_lines.append("  " + "─" * 60)
        for cmd, desc in rows:
            body_lines.append(_row(cmd, desc))

    body_lines.append("\n  [bold white]EXAMPLES[/bold white]")
    body_lines.append("  " + "─" * 60)
    body_lines.append("  [bold]Full local stack (offline after install):[/bold]")
    body_lines.append("    [cyan]voxkit install stt whisper[/cyan]")
    body_lines.append("    [cyan]voxkit install tts edge[/cyan]")
    body_lines.append("    [cyan]voxkit install llm ollama -m llama3.2[/cyan]")
    body_lines.append("    [cyan]voxkit demo[/cyan]")
    body_lines.append("")
    body_lines.append("  [bold]Cloud API stack:[/bold]")
    body_lines.append("    [cyan]voxkit install stt deepgram[/cyan]")
    body_lines.append("    [cyan]voxkit install tts elevenlabs[/cyan]")
    body_lines.append("    [cyan]voxkit install llm openai[/cyan]")
    body_lines.append("")
    body_lines.append("  [bold]Indic language stack:[/bold]")
    body_lines.append("    [cyan]voxkit install stt seamless[/cyan]")
    body_lines.append("    [cyan]voxkit install tts sarvam[/cyan]")
    body_lines.append("    [cyan]voxkit install llm google[/cyan]")
    body_lines.append("")
    body_lines.append("  [bold]Dry-run (see commands without running):[/bold]")
    body_lines.append("    [cyan]voxkit install stt whisper --dry-run[/cyan]")
    body_lines.append("    [cyan]voxkit wizard --dry-run[/cyan]")

    console.print(
        Panel(
            "\n".join(body_lines),
            title="[bold white] VOXKIT COMMAND REFERENCE [/bold white]",
            border_style="cyan",
            expand=True,
        )
    )


# ── detect ────────────────────────────────────────────────────────────────────

@main.command()
def detect() -> None:
    """Show detected system specs and platform capabilities."""
    info = _detect_system()
    ui.print_system_panel(info)

    rows = [
        ("Apple Silicon",    "yes ✓" if info.apple_silicon else "no",                                 "green" if info.apple_silicon else "dim"),
        ("NVIDIA CUDA",      f"yes ✓  (CUDA {info.cuda_version})" if info.cuda else "no",             "green" if info.cuda else "dim"),
        ("GPU",              info.gpu_name or "—",                                                      "cyan" if info.gpu_name else "dim"),
        ("Homebrew",         "yes" if info.homebrew else "no",                                          "green" if info.homebrew else "dim"),
        ("Recommended STT",  _rec_stt(info),                                                            "bold cyan"),
        ("Recommended TTS",  "edge-tts (free) or sarvam (Indic)",                                      "bold cyan"),
        ("Recommended LLM",  _rec_llm(info),                                                            "bold cyan"),
    ]

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("", style="bold white")
    t.add_column("")
    for label, val, style in rows:
        t.add_row(label, f"[{style}]{val}[/{style}]")
    console.print(t)
    console.print()
    console.print("[dim]Run [cyan]voxkit list all[/cyan] to see every provider.[/dim]")


def _rec_stt(info) -> str:
    if info.apple_silicon:
        return "whisper  →  mlx-whisper (Metal)"
    if info.cuda:
        return "whisper  →  faster-whisper (CUDA)"
    return "whisper  →  faster-whisper (CPU)"


def _rec_llm(info) -> str:
    if info.apple_silicon or info.cuda:
        return "ollama (local) or any API provider"
    return "openai / groq / google (API)"


# ── list ──────────────────────────────────────────────────────────────────────

@main.command(name="list")
@click.argument("category", type=click.Choice(["stt", "tts", "llm", "all"]), default="all")
def list_providers(category: str) -> None:
    """List available providers.\n\nCATEGORY: stt | tts | llm | all"""
    info = _detect_system()

    if category in ("stt", "all"):
        console.print(ui.providers_table(STT_PROVIDERS, info.platform_key, "Speech-to-Text (STT) Engines"))
        console.print()
    if category in ("tts", "all"):
        console.print(ui.providers_table(TTS_PROVIDERS, info.platform_key, "Text-to-Speech (TTS) Engines"))
        console.print()
    if category in ("llm", "all"):
        _llm_table()


def _llm_table() -> None:
    t = Table(title="LLM Providers", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Key", style="cyan", no_wrap=True)
    t.add_column("Name", style="bold")
    t.add_column("Description")
    t.add_column("Type", justify="center")
    t.add_column("API Key?", justify="center")
    for key, p in LLM_PROVIDERS.items():
        llm_type = "[green]local[/green]" if p.get("type") == "local" else "[blue]api[/blue]"
        api_req  = "[yellow]yes[/yellow]" if p.get("requires_api_key") else "[green]no[/green]"
        t.add_row(key, f"{p.get('emoji','')} {p['name']}", p["description"], llm_type, api_req)
    console.print(t)


# ── install ───────────────────────────────────────────────────────────────────

@main.group()
def install() -> None:
    """Install an STT, TTS, or LLM component."""


@install.command(name="stt")
@click.argument("provider")
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
def install_stt(provider: str, dry_run: bool) -> None:
    """Install a speech-to-text engine.

    \b
    PROVIDER options:
      whisper        Auto-selects mlx / faster-whisper / CPU
      seamless       Meta SeamlessM4T — great for Indic languages
      google         Google Cloud Speech API
      deepgram       Deepgram Nova-2 API
      assemblyai     AssemblyAI API
      openai-whisper OpenAI Whisper API
    """
    info = _detect_system()
    ui.print_system_panel(info)

    if provider not in STT_PROVIDERS:
        ui.error(f"Unknown STT provider: [bold]{provider}[/bold]")
        _suggest(provider, list(STT_PROVIDERS.keys()))
        sys.exit(1)

    p = STT_PROVIDERS[provider]
    console.print(f"\n  {p['emoji']}  [bold]{p['name']}[/bold] — {p['description']}\n")

    # ── Whisper: ask model size + download-now ────────────────────────────────
    model_size = "base"
    download_now = False

    if provider == "whisper" and not dry_run and _is_interactive():
        model_choices = [
            f"{'⚡' if m=='tiny' else '✨' if m=='base' else '🎯' if m=='small' else '📦' if m=='medium' else '🏆'} "
            f"{m:<10}  {size:<6}  {desc}"
            for m, size, desc in _WHISPER_MODELS
        ]
        default_idx = 1  # "base"
        chosen = questionary.select(
            "Which model size do you want?",
            choices=model_choices,
            default=model_choices[default_idx],
        ).ask()
        if chosen:
            model_size = _WHISPER_MODELS[model_choices.index(chosen)][0]

        download_now = questionary.confirm(
            f"Download the {model_size} model weights now? "
            "(otherwise downloads automatically on first use)",
            default=False,
        ).ask()

    ok = installer.install_stt(provider, info, model_size=model_size, dry_run=dry_run)

    if ok and provider == "whisper":
        installer.download_stt_model_if_needed(provider, model_size, download_now, info, dry_run=dry_run)

    if ok:
        console.print(f"\n  [dim]Test it with:[/dim]  [cyan]voxkit demo[/cyan]")
    _final(ok, f"STT [bold]{provider}[/bold]")


@install.command(name="tts")
@click.argument("provider")
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
def install_tts(provider: str, dry_run: bool) -> None:
    """Install a text-to-speech engine.

    \b
    PROVIDER options:
      edge        Microsoft Edge TTS — free, 400+ voices
      sarvam      Sarvam AI — best Indic TTS (API key needed)
      piper       Piper — fast offline TTS
      coqui       Coqui XTTS — voice cloning, 17 languages
      elevenlabs  ElevenLabs API — most realistic voices
      openai-tts  OpenAI TTS-1 / TTS-1-HD
      google-tts  Google Cloud TTS
    """
    info = _detect_system()
    ui.print_system_panel(info)

    if provider not in TTS_PROVIDERS:
        ui.error(f"Unknown TTS provider: [bold]{provider}[/bold]")
        _suggest(provider, list(TTS_PROVIDERS.keys()))
        sys.exit(1)

    p = TTS_PROVIDERS[provider]
    console.print(f"\n  {p['emoji']}  [bold]{p['name']}[/bold] — {p['description']}\n")

    # ── Edge TTS: ask language family + show voice names ─────────────────────
    if provider == "edge" and not dry_run and _is_interactive():
        lang_choices = list(_EDGE_VOICE_PRESETS.keys()) + ["Browse all — I'll pick with edge-tts --list-voices"]
        lang_pick = questionary.select(
            "Which language family is your primary use case?",
            choices=lang_choices,
        ).ask()

        if lang_pick and lang_pick in _EDGE_VOICE_PRESETS:
            preset = _EDGE_VOICE_PRESETS[lang_pick]
            voice_choices = [f"{v}  —  {desc}" for v, desc in preset["voices"]]
            voice_pick = questionary.select(
                "Pick a default voice (you can change this in your code):",
                choices=voice_choices,
            ).ask()
            if voice_pick:
                chosen_voice = voice_pick.split("  —  ")[0].strip()
                console.print(f"\n  [dim]Default voice:[/dim] [cyan]{chosen_voice}[/cyan]")
                console.print(f"  [dim]Usage in code:[/dim] [dim]edge_tts.Communicate(text, voice=\"{chosen_voice}\")[/dim]\n")

    # ── Sarvam: remind about API key ─────────────────────────────────────────
    if provider == "sarvam" and not dry_run:
        console.print(
            Panel(
                "Sarvam AI requires an API key.\n"
                "Get yours at: [link=https://dashboard.sarvam.ai]https://dashboard.sarvam.ai[/link]\n"
                "Then set [cyan]SARVAM_API_KEY=your-key[/cyan] in your .env file.",
                border_style="yellow",
                expand=False,
            )
        )
        if _is_interactive() and not questionary.confirm("Continue with install?", default=True).ask():
            ui.warn("Aborted.")
            return

    ok = installer.install_tts(provider, info, dry_run=dry_run)
    if ok:
        console.print(f"\n  [dim]Test it with:[/dim]  [cyan]voxkit demo[/cyan]")
    _final(ok, f"TTS [bold]{provider}[/bold]")


@install.command(name="llm")
@click.argument("provider")
@click.option("--model", "-m", default=None, help="Ollama model to pull (e.g. llama3.2)")
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
def install_llm(provider: str, model: Optional[str], dry_run: bool) -> None:
    """Install a language model provider.

    \b
    PROVIDER options:
      ollama     Local LLMs via Ollama — Llama, Mistral, Phi, Gemma
      openai     OpenAI GPT-4o API
      anthropic  Anthropic Claude 4 API
      google     Google Gemini 2.5 API
      groq       Groq — ultra-fast LPU inference
      mistral    Mistral AI API
      cohere     Cohere Command R+ — RAG-optimized
      llama-cpp  llama.cpp — run GGUF models locally
    """
    info = _detect_system()
    ui.print_system_panel(info)

    if provider not in LLM_PROVIDERS:
        ui.error(f"Unknown LLM provider: [bold]{provider}[/bold]")
        _suggest(provider, list(LLM_PROVIDERS.keys()))
        sys.exit(1)

    p = LLM_PROVIDERS[provider]
    console.print(f"\n  {p['emoji']}  [bold]{p['name']}[/bold] — {p['description']}\n")

    # ── Ollama: RAM-aware model picker ────────────────────────────────────────
    if provider == "ollama" and model is None and not dry_run and _is_interactive():
        ram_gb = _get_ram_gb()
        popular = LLM_PROVIDERS["ollama"]["popular_models"]

        if ram_gb:
            console.print(f"  [dim]Detected RAM:[/dim] [cyan]{ram_gb:.1f} GB[/cyan]")
            recommended = _recommend_ollama_model(ram_gb)
            console.print(f"  [dim]Recommended for your RAM:[/dim] [cyan]{recommended}[/cyan]\n")

        model_choices = [f"{m:<30}  {desc}" for m, desc in popular] + ["skip — install binary only, I'll pull later"]
        chosen = questionary.select(
            "Which model do you want to pull?",
            choices=model_choices,
        ).ask()
        if chosen and not chosen.startswith("skip"):
            model = chosen.split()[0].strip()

    # ── API providers: show key setup instructions ────────────────────────────
    if p.get("requires_api_key") and not dry_run and _is_interactive():
        key_name = p.get("api_key_name", "API_KEY")
        api_link = p.get("api_link", "")
        console.print(
            Panel(
                f"[bold]{p['name']}[/bold] requires an API key.\n"
                f"  • Key name:  [cyan]{key_name}[/cyan]\n"
                f"  • Get one:   [link={api_link}]{api_link}[/link]\n"
                f"  • Add to your .env:  [dim]{key_name}=your-key-here[/dim]",
                border_style="yellow",
                expand=False,
            )
        )
        if not questionary.confirm("Continue with install?", default=True).ask():
            ui.warn("Aborted.")
            return

        # Show popular models if available
        models_list = p.get("popular_models", [])
        if models_list:
            console.print("\n  [bold white]Popular models:[/bold white]")
            for m, desc in models_list:
                console.print(f"    [cyan]{m:<35}[/cyan] [dim]{desc}[/dim]")
            console.print()

    ok = installer.install_llm(provider, info, model=model, dry_run=dry_run)
    if ok:
        console.print(f"\n  [dim]Test it with:[/dim]  [cyan]voxkit demo[/cyan]")
    _final(ok, f"LLM [bold]{provider}[/bold]")


# ── wizard ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
def wizard(dry_run: bool) -> None:
    """Interactive setup — guides you through STT + TTS + LLM install."""
    if not _is_interactive() and not dry_run:
        ui.error("wizard requires an interactive terminal. Run in a real shell, not a pipe.")
        sys.exit(1)
    ui.print_banner()
    info = _detect_system()
    ui.print_system_panel(info)

    console.print(
        Panel(
            "[bold]Let's build your voice stack![/bold]\n"
            "[dim]I'll ask a few questions and install everything for your system.[/dim]",
            border_style="magenta",
            expand=False,
        )
    )
    console.print()

    # ── Step 1: STT ──────────────────────────────────────────────────────────
    console.print("[bold cyan]Step 1 / 3 — Speech-to-Text (STT)[/bold cyan]")
    console.print("[dim]  Converts user speech → text, which is sent to the LLM.[/dim]\n")
    stt_choices = _build_provider_choices(STT_PROVIDERS, info.platform_key)
    stt_label = questionary.select("Which STT engine?", choices=stt_choices + ["⏭  skip"]).ask()
    stt_key = _label_to_key(stt_label, STT_PROVIDERS, info.platform_key) if stt_label != "⏭  skip" else None

    stt_model = "base"
    stt_download = False
    if stt_key == "whisper" and not dry_run:
        model_choices = [
            f"{'⚡' if m=='tiny' else '✨' if m=='base' else '🎯' if m=='small' else '📦' if m=='medium' else '🏆'} "
            f"{m:<10}  {size:<6}  {desc}"
            for m, size, desc in _WHISPER_MODELS
        ]
        chosen = questionary.select("  Which Whisper model size?", choices=model_choices, default=model_choices[1]).ask()
        if chosen:
            stt_model = _WHISPER_MODELS[model_choices.index(chosen)][0]
        stt_download = questionary.confirm(f"  Download {stt_model} model now?", default=False).ask()

    # ── Step 2: LLM ──────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2 / 3 — Language Model (LLM)[/bold cyan]")
    console.print("[dim]  Receives the transcribed text and generates a response.[/dim]\n")
    llm_keys = list(LLM_PROVIDERS.keys())
    llm_labels = [
        f"{LLM_PROVIDERS[k]['emoji']}  {LLM_PROVIDERS[k]['name']:<22} — {LLM_PROVIDERS[k]['description']}"
        for k in llm_keys
    ]
    llm_map = dict(zip(llm_labels, llm_keys))
    llm_label = questionary.select("Which LLM provider?", choices=llm_labels + ["⏭  skip"]).ask()
    llm_key = llm_map.get(llm_label) if llm_label != "⏭  skip" else None

    ollama_model = None
    if llm_key == "ollama" and not dry_run:
        ram_gb = _get_ram_gb()
        if ram_gb:
            console.print(f"  [dim]RAM: {ram_gb:.1f} GB → recommended:[/dim] [cyan]{_recommend_ollama_model(ram_gb)}[/cyan]")
        popular = LLM_PROVIDERS["ollama"]["popular_models"]
        mc = [f"{m:<30}  {desc}" for m, desc in popular] + ["skip — pull later"]
        chosen = questionary.select("  Which Ollama model to pull?", choices=mc).ask()
        if chosen and not chosen.startswith("skip"):
            ollama_model = chosen.split()[0].strip()

    # ── Step 3: TTS ──────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 3 / 3 — Text-to-Speech (TTS)[/bold cyan]")
    console.print("[dim]  Converts the LLM's response → spoken audio for the user.[/dim]\n")
    tts_choices = _build_provider_choices(TTS_PROVIDERS, info.platform_key)
    tts_label = questionary.select("Which TTS engine?", choices=tts_choices + ["⏭  skip"]).ask()
    tts_key = _label_to_key(tts_label, TTS_PROVIDERS, info.platform_key) if tts_label != "⏭  skip" else None

    edge_voice = None
    if tts_key == "edge" and not dry_run:
        lang_pick = questionary.select(
            "  Language family?",
            choices=list(_EDGE_VOICE_PRESETS.keys()),
        ).ask()
        if lang_pick and lang_pick in _EDGE_VOICE_PRESETS:
            preset = _EDGE_VOICE_PRESETS[lang_pick]
            voice_choices = [f"{v}  —  {desc}" for v, desc in preset["voices"]]
            vp = questionary.select("  Default voice?", choices=voice_choices).ask()
            if vp:
                edge_voice = vp.split("  —  ")[0].strip()

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print()
    stt_summary = f"{stt_key} ({stt_model})" if stt_key == "whisper" else (stt_key or "skipped")
    llm_summary = f"ollama ({ollama_model})" if llm_key == "ollama" and ollama_model else (llm_key or "skipped")
    tts_summary = f"{tts_key} ({edge_voice})" if tts_key == "edge" and edge_voice else (tts_key or "skipped")

    console.print(
        Panel(
            f"  STT  →  [cyan]{stt_summary}[/cyan]\n"
            f"  LLM  →  [cyan]{llm_summary}[/cyan]\n"
            f"  TTS  →  [cyan]{tts_summary}[/cyan]",
            title="[bold white] Your install plan (STT → LLM → TTS) [/bold white]",
            border_style="green",
            expand=False,
        )
    )

    if not questionary.confirm("Ready? Install now?", default=True).ask():
        ui.warn("Aborted. Nothing was installed.")
        return

    results: dict[str, bool] = {}

    if stt_key:
        results["STT"] = installer.install_stt(stt_key, info, model_size=stt_model, dry_run=dry_run)
        if results["STT"] and stt_key == "whisper":
            installer.download_stt_model_if_needed(stt_key, stt_model, stt_download, info, dry_run=dry_run)

    if llm_key:
        results["LLM"] = installer.install_llm(llm_key, info, model=ollama_model, dry_run=dry_run)

    if tts_key:
        results["TTS"] = installer.install_tts(tts_key, info, dry_run=dry_run)

    # ── Final report ─────────────────────────────────────────────────────────
    console.print()
    for component, ok in results.items():
        status = "[bold green]✓ success[/bold green]" if ok else "[bold red]✗ failed[/bold red]"
        console.print(f"  {component}: {status}")

    if results and all(results.values()):
        console.print(
            Panel(
                "[bold green]All done! 🎉[/bold green]\n"
                "[dim]Run [cyan]voxkit demo[/cyan] to test your stack end-to-end.[/dim]",
                border_style="green",
                expand=False,
            )
        )
    elif results:
        ui.warn("Some components failed — check output above.")


# ── demo ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--model", "-m", default="base", show_default=True,
              help="Whisper model size for STT test (tiny/base/small/medium/large-v3)")
def demo(model: str) -> None:
    """Run a live end-to-end voice pipeline: STT → LLM → TTS.

    Simulates a user question, transcribes it with STT, sends the text to
    your LLM, then speaks the response aloud with TTS — all in one shot.
    """
    from . import demo as demo_mod
    demo_mod.run(whisper_model=model)


# ── doctor ────────────────────────────────────────────────────────────────────

@main.command()
def doctor() -> None:
    """Check your voice pipeline (STT → LLM → TTS) and show what's missing."""
    import importlib
    import subprocess as _sp

    from .detect import _cmd_exists

    def _installed(module: str) -> bool:
        try:
            importlib.import_module(module)
            return True
        except ImportError:
            return False

    def _row(t: Table, label: str, module: str, note: str = "") -> bool:
        ok = _installed(module)
        status = "[bold green]✓ installed[/bold green]" if ok else "[dim]not installed[/dim]"
        t.add_row(label, status, note)
        return ok

    console.print(
        Panel(
            "[bold white]Voice Pipeline Health Check[/bold white]\n"
            "[dim]STT → LLM → TTS  ·  each step feeds into the next[/dim]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    # ── Step 1: STT ──────────────────────────────────────────────────────────
    console.print("[bold cyan]Step 1 — Speech-to-Text (STT)[/bold cyan]")
    console.print("[dim]Converts user speech → text for the LLM[/dim]")
    stt_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    stt_table.add_column("Engine", style="bold", no_wrap=True)
    stt_table.add_column("Status", justify="center")
    stt_table.add_column("Note", style="dim")

    stt_any = False
    stt_any |= _row(stt_table, "mlx-whisper",    "mlx_whisper",   "Apple Silicon · GPU-accelerated")
    stt_any |= _row(stt_table, "faster-whisper", "faster_whisper","CPU / CUDA · cross-platform")
    stt_any |= _row(stt_table, "deepgram",        "deepgram",      "Cloud API (DEEPGRAM_API_KEY)")
    stt_any |= _row(stt_table, "assemblyai",      "assemblyai",    "Cloud API (ASSEMBLYAI_API_KEY)")
    stt_any |= _row(stt_table, "openai (whisper)","openai",        "Cloud API (OPENAI_API_KEY)")
    console.print(stt_table)

    if not stt_any:
        console.print(
            Panel(
                "[bold yellow]No STT engine found.[/bold yellow]\n"
                "Without STT, speech cannot be transcribed → the pipeline cannot start.\n\n"
                "  Fix (local, works offline):\n"
                "    [cyan]voxkit install stt whisper[/cyan]\n"
                "      → Apple Silicon: installs mlx-whisper (Metal GPU, very fast)\n"
                "      → Linux / Windows: installs faster-whisper (CPU or CUDA)\n\n"
                "  Fix (cloud API):\n"
                "    [cyan]voxkit install stt deepgram[/cyan]   then set [dim]DEEPGRAM_API_KEY[/dim]",
                border_style="yellow",
                expand=False,
            )
        )
    console.print()

    # ── Step 2: LLM ──────────────────────────────────────────────────────────
    console.print("[bold cyan]Step 2 — Language Model (LLM)[/bold cyan]")
    console.print("[dim]Receives transcribed text and generates a response[/dim]")
    llm_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    llm_table.add_column("Provider", style="bold", no_wrap=True)
    llm_table.add_column("Status", justify="center")
    llm_table.add_column("Note", style="dim")

    ollama_bin = _cmd_exists("ollama")
    ollama_status = "[bold green]✓ installed[/bold green]" if ollama_bin else "[dim]not installed[/dim]"
    llm_table.add_row("ollama", ollama_status, "Local LLMs — no API key needed")
    _row(llm_table, "openai",    "openai",     "GPT-4o (OPENAI_API_KEY)")
    _row(llm_table, "anthropic", "anthropic",  "Claude (ANTHROPIC_API_KEY)")
    _row(llm_table, "google",    "google.genai","Gemini (GOOGLE_API_KEY)")
    _row(llm_table, "groq",      "groq",        "Fast inference (GROQ_API_KEY)")
    _row(llm_table, "mistral",   "mistralai",   "Mistral API (MISTRAL_API_KEY)")
    _row(llm_table, "llama-cpp", "llama_cpp",   "Local GGUF models")
    console.print(llm_table)

    llm_any = ollama_bin or any(
        _installed(m) for m in ["openai", "anthropic", "google.genai", "groq", "mistralai", "llama_cpp"]
    )

    if ollama_bin:
        r = _sp.run(["ollama", "list"], capture_output=True, text=True)
        if r.returncode == 0:
            lines = [l for l in r.stdout.strip().splitlines()[1:] if l.strip()]
            if lines:
                console.print("  [bold]Pulled Ollama models:[/bold]")
                for line in lines:
                    console.print(f"    [cyan]{line.split()[0]}[/cyan]")
            else:
                console.print(
                    "  [yellow]Ollama installed but no models pulled yet.[/yellow]\n"
                    "  Pull one:  [cyan]ollama pull llama3.2[/cyan]"
                )

    if not llm_any:
        console.print(
            Panel(
                "[bold yellow]No LLM configured.[/bold yellow]\n"
                "Without an LLM, there is nothing to understand the user's speech or reply.\n\n"
                "  Fix (local, no API key, runs offline after setup):\n"
                "    [cyan]voxkit install llm ollama[/cyan]\n"
                "      → Downloads Ollama + pulls a model (Llama, Mistral, Phi, Gemma...)\n\n"
                "  Fix (cloud API, no GPU needed):\n"
                "    [cyan]voxkit install llm openai[/cyan]     then set [dim]OPENAI_API_KEY[/dim]\n"
                "    [cyan]voxkit install llm anthropic[/cyan]  then set [dim]ANTHROPIC_API_KEY[/dim]",
                border_style="yellow",
                expand=False,
            )
        )
    console.print()

    # ── Step 3: TTS ──────────────────────────────────────────────────────────
    console.print("[bold cyan]Step 3 — Text-to-Speech (TTS)[/bold cyan]")
    console.print("[dim]Converts the LLM response → spoken audio for the user[/dim]")
    tts_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tts_table.add_column("Engine", style="bold", no_wrap=True)
    tts_table.add_column("Status", justify="center")
    tts_table.add_column("Note", style="dim")

    tts_any = False
    tts_any |= _row(tts_table, "edge-tts",   "edge_tts",  "Free · 400+ voices · no API key")
    tts_any |= _row(tts_table, "TTS (Coqui)","TTS",        "Local · voice cloning · 17 languages")
    tts_any |= _row(tts_table, "piper-tts",  "piper",      "Fast local TTS · ~100 ms latency")
    tts_any |= _row(tts_table, "elevenlabs", "elevenlabs", "Cloud API (ELEVENLABS_API_KEY)")
    tts_any |= _row(tts_table, "openai-tts", "openai",     "Cloud API (OPENAI_API_KEY)")
    console.print(tts_table)

    if not tts_any:
        console.print(
            Panel(
                "[bold yellow]No TTS engine found.[/bold yellow]\n"
                "Without TTS, the LLM response stays as text — users won't hear anything.\n\n"
                "  Fix (free, easiest, no API key):\n"
                "    [cyan]voxkit install tts edge[/cyan]\n"
                "      → Installs edge-tts · 400+ voices · works on all platforms\n\n"
                "  Fix (local, highest quality):\n"
                "    [cyan]voxkit install tts coqui[/cyan]   ← voice cloning, 17 languages\n"
                "    [cyan]voxkit install tts piper[/cyan]   ← fast, fully offline",
                border_style="yellow",
                expand=False,
            )
        )
    console.print()

    # ── Overall verdict ───────────────────────────────────────────────────────
    all_ok = stt_any and llm_any and tts_any
    if all_ok:
        console.print(
            Panel(
                "[bold green]Pipeline ready![/bold green]  STT ✓  →  LLM ✓  →  TTS ✓\n"
                "[dim]Run [cyan]voxkit demo[/cyan] to test it end-to-end.[/dim]",
                border_style="green",
                expand=False,
            )
        )
    else:
        missing = []
        if not stt_any:
            missing.append("STT  →  [cyan]voxkit install stt whisper[/cyan]")
        if not llm_any:
            missing.append("LLM  →  [cyan]voxkit install llm ollama[/cyan]")
        if not tts_any:
            missing.append("TTS  →  [cyan]voxkit install tts edge[/cyan]")
        console.print(
            Panel(
                "[bold yellow]Pipeline incomplete.[/bold yellow]  Install what's missing:\n\n"
                + "\n".join(f"  {m}" for m in missing)
                + "\n\n[dim]Or run [cyan]voxkit wizard[/cyan] for a guided setup.[/dim]",
                border_style="yellow",
                expand=False,
            )
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_provider_choices(providers: dict, platform_key: str) -> list[str]:
    choices = []
    for key, p in providers.items():
        variants = p.get("variants", {})
        vk = _pick_variant(variants, platform_key)
        variant = variants.get(vk, {})
        label = variant.get("label", "all platforms")
        api_note = "  [API key]" if p.get("requires_api_key") else ""
        choices.append(f"{p['emoji']}  {p['name']:<20}{api_note}  —  {label}")
    return choices


def _label_to_key(label: str, providers: dict, platform_key: str) -> str:
    choices = _build_provider_choices(providers, platform_key)
    keys = list(providers.keys())
    try:
        return keys[choices.index(label)]
    except ValueError:
        return label


def _pick_variant(variants: dict, platform_key: str) -> str:
    if platform_key in variants:
        return platform_key
    if "all" in variants:
        return "all"
    return next(iter(variants), "")


def _final(ok: bool, label: str) -> None:
    console.print()
    if ok:
        console.print(Panel(f"[bold green]✓ {label} installed successfully![/bold green]", border_style="green", expand=False))
    else:
        console.print(Panel(f"[bold red]✗ Installation failed — check output above.[/bold red]", border_style="red", expand=False))
        sys.exit(1)


def _suggest(key: str, valid: list[str]) -> None:
    import difflib
    close = difflib.get_close_matches(key, valid, n=3, cutoff=0.4)
    if close:
        ui.info(f"Did you mean: [cyan]{' | '.join(close)}[/cyan]?")
    else:
        ui.info(f"Valid options: [cyan]{' | '.join(valid)}[/cyan]")


def _get_ram_gb() -> Optional[float]:
    import platform, subprocess
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
            return int(r.stdout.strip()) / 1024 ** 3
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return None


def _is_interactive() -> bool:
    """True when stdin is a real terminal (not a pipe or CI environment)."""
    return sys.stdin.isatty()


def _recommend_ollama_model(ram_gb: float) -> str:
    if ram_gb >= 16:
        return "llama3.1:8b  or  mistral"
    if ram_gb >= 8:
        return "llama3.2  or  phi3:mini"
    return "gemma2:2b  or  phi3:mini  (lightweight)"

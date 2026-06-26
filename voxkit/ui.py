"""Terminal UI helpers — Rich-based."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

BANNER = r"""
 __   __  ___   _     _  __  ___  _____
 \ \ / / / _ \ | |   | |/ / |_ _||_   _|
  \ V / | | | || |   | ' /   | |   | |
   \_/  | |_| || |___| . \   | |   | |
        \___/ |_____|_|\_\ |___| |_|
"""

TAGLINE = "  your voice stack installer — smart, speedy & a lil' sassy"


def print_banner() -> None:
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    console.print(f"[dim]{TAGLINE}[/dim]\n")


def print_system_panel(info) -> None:
    os_icon = {"Darwin": "🍎", "Linux": "🐧", "Windows": "🪟"}.get(info.os, "💻")
    gpu_line = ""
    if info.apple_silicon:
        gpu_line = "\n  [green]GPU[/green]   Apple Silicon (Metal)"
    elif info.cuda:
        gpu_line = f"\n  [green]GPU[/green]   {info.gpu_name or 'NVIDIA'} · CUDA {info.cuda_version or '?'}"
    else:
        gpu_line = "\n  [yellow]GPU[/yellow]   None detected (CPU mode)"

    body = (
        f"  [bold]OS[/bold]    {os_icon} {info.os_label} · {info.arch}"
        f"\n  [bold]Python[/bold] {info.python_version}"
        f"{gpu_line}"
        f"\n  [bold]Mode[/bold]  [cyan]{info.platform_key}[/cyan]"
    )
    console.print(
        Panel(body, title="[bold white]System Detected[/bold white]", border_style="cyan", expand=False)
    )
    console.print()


def providers_table(providers: dict, platform_key: str, title: str) -> Table:
    t = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Key", style="cyan", no_wrap=True)
    t.add_column("Name", style="bold")
    t.add_column("Description")
    t.add_column("Platform", style="dim")
    t.add_column("API Key?", justify="center")

    for key, p in providers.items():
        variant_key = _best_variant(p.get("variants", {}), platform_key)
        variant = p.get("variants", {}).get(variant_key, {})
        platform_label = variant.get("label", variant_key or "all platforms")
        api_req = "[yellow]yes[/yellow]" if p.get("requires_api_key") else "[green]no[/green]"
        t.add_row(key, f"{p.get('emoji', '')} {p['name']}", p["description"], platform_label, api_req)

    return t


def _best_variant(variants: dict, platform_key: str) -> str:
    if platform_key in variants:
        return platform_key
    if "all" in variants:
        return "all"
    return next(iter(variants), "")


def success(msg: str) -> None:
    console.print(f"  [bold green]✓[/bold green] {msg}")


def info(msg: str) -> None:
    console.print(f"  [bold cyan]→[/bold cyan] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [bold yellow]⚠[/bold yellow]  {msg}")


def error(msg: str) -> None:
    console.print(f"  [bold red]✗[/bold red] {msg}")


def section(title: str) -> None:
    console.print(f"\n[bold white]{title}[/bold white]")
    console.print("─" * len(title))

"""TTS provider registry."""

TTS_PROVIDERS = {
    "edge": {
        "name": "Edge TTS",
        "description": "Microsoft Edge TTS — 400+ voices, works offline",
        "emoji": "🗣️",
        "requires_api_key": False,
        "variants": {
            "all": {
                "label": "Edge TTS (all platforms)",
                "pip": ["edge-tts>=6.1.0"],
            },
        },
        "notes": {
            "all": "Free, high-quality neural voices. No API key needed.",
        },
    },
    "sarvam": {
        "name": "Sarvam AI TTS",
        "description": "Sarvam AI — best-in-class Indic language TTS",
        "emoji": "🇮🇳",
        "requires_api_key": True,
        "api_key_name": "SARVAM_API_KEY",
        "api_link": "https://dashboard.sarvam.ai",
        "variants": {
            "all": {
                "label": "Sarvam AI (aiohttp)",
                "pip": ["aiohttp>=3.9.0"],
            },
        },
        "notes": {
            "all": "Supports Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali and more.",
        },
    },
    "piper": {
        "name": "Piper TTS",
        "description": "Fast local neural TTS — runs entirely offline",
        "emoji": "🪈",
        "requires_api_key": False,
        "variants": {
            "apple_silicon": {
                "label": "Piper TTS (macOS arm64)",
                "pip": ["piper-tts>=1.2.0"],
                "post_install": [
                    "echo '💡 Download a voice model: https://huggingface.co/rhasspy/piper-voices'",
                ],
            },
            "cuda": {
                "label": "Piper TTS (Linux/Windows CUDA)",
                "pip": ["piper-tts>=1.2.0"],
            },
            "cpu": {
                "label": "Piper TTS (CPU)",
                "pip": ["piper-tts>=1.2.0"],
            },
        },
        "notes": {
            "all": "Very fast, ~100ms latency. Download voice models separately.",
        },
    },
    "coqui": {
        "name": "Coqui XTTS",
        "description": "Coqui XTTS-v2 — voice cloning, 17 languages",
        "emoji": "🐸",
        "requires_api_key": False,
        "variants": {
            "cuda": {
                "label": "Coqui TTS (CUDA)",
                "pip": ["TTS>=0.22.0", "torch>=2.2.0", "torchaudio>=2.2.0"],
                "pip_flags": ["--index-url", "{torch_index_url}"],
                "pip_flags_for": ["torch>=2.2.0", "torchaudio>=2.2.0"],
            },
            "apple_silicon": {
                "label": "Coqui TTS (Apple Silicon / MPS)",
                "pip": ["TTS>=0.22.0", "torch>=2.2.0", "torchaudio>=2.2.0"],
            },
            "cpu": {
                "label": "Coqui TTS (CPU)",
                "pip": ["TTS>=0.22.0"],
            },
        },
        "notes": {
            "all": "Downloads ~2 GB XTTS-v2 model on first use.",
            "cuda": "Runs best with 8 GB+ VRAM.",
        },
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "description": "ElevenLabs API — most realistic AI voices",
        "emoji": "🔈",
        "requires_api_key": True,
        "api_key_name": "ELEVENLABS_API_KEY",
        "api_link": "https://elevenlabs.io/app",
        "variants": {
            "all": {
                "label": "ElevenLabs SDK",
                "pip": ["elevenlabs>=1.0.0"],
            },
        },
    },
    "openai-tts": {
        "name": "OpenAI TTS",
        "description": "OpenAI TTS-1 / TTS-1-HD — natural voices via API",
        "emoji": "🤖",
        "requires_api_key": True,
        "api_key_name": "OPENAI_API_KEY",
        "api_link": "https://platform.openai.com/api-keys",
        "variants": {
            "all": {
                "label": "OpenAI SDK",
                "pip": ["openai>=1.0.0"],
            },
        },
    },
    "google-tts": {
        "name": "Google Cloud TTS",
        "description": "Google Text-to-Speech API — WaveNet + Journey voices",
        "emoji": "☁️",
        "requires_api_key": True,
        "api_key_name": "GOOGLE_APPLICATION_CREDENTIALS",
        "api_link": "https://console.cloud.google.com/speech",
        "variants": {
            "all": {
                "label": "Google Cloud TTS",
                "pip": ["google-cloud-texttospeech>=2.0.0"],
            },
        },
    },
}

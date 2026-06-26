"""STT provider registry."""

# Each provider has:
#   name, description, emoji, requires_api_key
#   variants: dict[platform_key -> install spec]
#     platform_key: "apple_silicon" | "cuda" | "cpu" | "all"
#     spec: { label, pip, pip_flags (optional), post_install (optional) }

STT_PROVIDERS = {
    "whisper": {
        "name": "Whisper",
        "description": "OpenAI Whisper — best accuracy, 90+ languages",
        "emoji": "🎙️",
        "requires_api_key": False,
        "variants": {
            "apple_silicon": {
                "label": "MLX Whisper (Apple Silicon — Metal GPU)",
                "pip": ["mlx-whisper>=0.3.0"],
            },
            "cuda": {
                "label": "Faster Whisper (NVIDIA CUDA)",
                "pip": ["faster-whisper>=1.0.0", "torch>=2.2.0", "torchaudio>=2.2.0"],
                "pip_flags": ["--index-url", "{torch_index_url}"],
                "pip_flags_for": ["torch>=2.2.0", "torchaudio>=2.2.0"],
            },
            "cpu": {
                "label": "Faster Whisper (CPU — int8)",
                "pip": ["faster-whisper>=1.0.0"],
            },
        },
        "notes": {
            "apple_silicon": "Uses MLX for blazing-fast on-device transcription.",
            "cuda": "CUDA 11.8+ required. Installs torch with CUDA wheels.",
            "cpu": "Runs on any CPU; slower but works everywhere.",
        },
    },
    "seamless": {
        "name": "SeamlessM4T",
        "description": "Meta's multilingual model — great for Indic languages",
        "emoji": "🌐",
        "requires_api_key": False,
        "variants": {
            "all": {
                "label": "SeamlessM4T (all platforms)",
                "pip": [
                    "transformers>=4.40.0",
                    "huggingface-hub>=0.23.0",
                    "torch>=2.2.0",
                    "torchaudio>=2.2.0",
                    "sentencepiece>=0.1.99",
                ],
            },
        },
        "notes": {
            "all": "Downloads ~2 GB model from Hugging Face on first use.",
        },
    },
    "google": {
        "name": "Google Cloud STT",
        "description": "Google Speech-to-Text API — fast, reliable",
        "emoji": "☁️",
        "requires_api_key": True,
        "api_key_name": "GOOGLE_APPLICATION_CREDENTIALS",
        "api_link": "https://console.cloud.google.com/speech",
        "variants": {
            "all": {
                "label": "Google Cloud Speech",
                "pip": ["google-cloud-speech>=2.0.0"],
            },
        },
    },
    "deepgram": {
        "name": "Deepgram",
        "description": "Deepgram Nova-2 API — ultra-fast streaming STT",
        "emoji": "🔊",
        "requires_api_key": True,
        "api_key_name": "DEEPGRAM_API_KEY",
        "api_link": "https://console.deepgram.com",
        "variants": {
            "all": {
                "label": "Deepgram SDK",
                "pip": ["deepgram-sdk>=3.0.0"],
            },
        },
    },
    "assemblyai": {
        "name": "AssemblyAI",
        "description": "AssemblyAI API — great accuracy + speaker detection",
        "emoji": "🏗️",
        "requires_api_key": True,
        "api_key_name": "ASSEMBLYAI_API_KEY",
        "api_link": "https://www.assemblyai.com/dashboard",
        "variants": {
            "all": {
                "label": "AssemblyAI SDK",
                "pip": ["assemblyai>=0.20.0"],
            },
        },
    },
    "openai-whisper": {
        "name": "OpenAI Whisper API",
        "description": "Whisper via OpenAI API — no local GPU needed",
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
}

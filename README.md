# voxkit

**Playful CLI installer for STT, TTS, and LLM engines — auto-detects your platform.**

```bash
pip install voxkit
voxkit wizard       # interactive setup
voxkit demo         # test your stack
```

## What it does

`voxkit` detects your system (Apple Silicon, NVIDIA CUDA, or CPU-only) and installs the right packages for speech-to-text, text-to-speech, and language models — asking the right questions along the way.

## Commands

| Command | Description |
|---|---|
| `voxkit help` | Full command reference |
| `voxkit detect` | Show OS, GPU, recommended stack |
| `voxkit list [stt\|tts\|llm\|all]` | Browse all providers |
| `voxkit install stt whisper` | Install Whisper STT (auto-selects mlx / faster-whisper) |
| `voxkit install tts edge` | Install Edge TTS — free, 400+ voices |
| `voxkit install llm ollama` | Install Ollama + pull a local model |
| `voxkit demo` | Live end-to-end test: TTS → STT → LLM |
| `voxkit doctor` | Check what's installed |
| `voxkit wizard` | Interactive full setup |

## Platform support

| Platform | STT | GPU |
|---|---|---|
| macOS Apple Silicon | mlx-whisper | Metal |
| Linux / Windows + NVIDIA | faster-whisper | CUDA |
| Any CPU | faster-whisper (int8) | — |

## Supported providers

**STT:** Whisper (mlx / faster), SeamlessM4T, Google, Deepgram, AssemblyAI, OpenAI  
**TTS:** Edge TTS, Sarvam AI, Piper, Coqui XTTS, ElevenLabs, OpenAI TTS, Google TTS  
**LLM:** Ollama, OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere, llama.cpp

## Example: full local stack

```bash
voxkit install stt whisper        # asks model size + download-now
voxkit install tts edge           # asks language family + voice
voxkit install llm ollama         # asks which model, RAM-aware
voxkit demo                       # TTS → STT → LLM end-to-end test
```

## Example: cloud API stack

```bash
voxkit install stt deepgram
voxkit install tts elevenlabs
voxkit install llm openai
```

## Flags

```bash
--dry-run          # print commands without running them
-m / --model       # specify Ollama model (e.g. voxkit install llm ollama -m phi3)
```

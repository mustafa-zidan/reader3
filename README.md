# Reader3

A lightweight, self-hosted EPUB & PDF reader for reading books alongside LLMs.

![reader3](reader3.png)

## Quick Start

```bash
# Install dependencies
uv sync

# Run the app
uv run python launcher.py
```

The browser opens automatically. Upload a book and start reading!

## Features

- 📚 **EPUB & PDF Support** — Read both formats with infinite scroll
- 🔖 **Bookmarks & Highlights** — Save passages with notes, highlight in 5 colors
- 🔍 **Search** — Full-text search across all books (Ctrl/⌘+F)
- 📊 **Reading Progress** — Auto-saves position, resume where you left off
- 📤 **Export** — Export notes to JSON or Markdown
- 📋 **Easy Copy** — Batch selects and copies text for LLM conversations
- 🤖 **AI Chat** — Chat with LLMs (LM Studio/Ollama) about your reading
- 🔊 **Text-to-Speech** — Read aloud with Orpheus TTS or browser speech

## AI Chat Setup

Reader3 supports AI chat integration with local LLMs via [LM Studio](https://lmstudio.ai/)
or [Ollama](https://ollama.ai/).

1. Install LM Studio or Ollama
2. Load a model (e.g., Llama, Mistral, etc.)
3. Start the local server
4. In Reader3, click the 🤖 AI icon to configure:
    - Select provider (LM Studio or Ollama)
    - Enter server URL (default: `http://localhost:1234/v1` for LM Studio)
    - Select your model
5. Use the 💬 Chat icon to open the AI sidebar

**Features:**

- Send selected text to chat for discussion
- Get summaries, explanations, or translations
- Chat history saved per book

## Text-to-Speech (Orpheus TTS)

Reader3 supports high-quality TTS using [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) via LM Studio, with
automatic fallback to browser speech synthesis.

### Quick Setup (Browser TTS)

TTS works out of the box using your browser's built-in speech synthesis. Just click the 🔊 speaker icons next to
paragraphs.

### Orpheus TTS Setup (High Quality)

For higher quality speech with multiple voice options:

1. **Install LM Studio** — Download from [lmstudio.ai](https://lmstudio.ai/)

2. **Download Orpheus Model** — In LM Studio, search for and download:
   ```
   isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF
   ```
   Or download directly from [HuggingFace](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF)

3. **Install TTS Dependencies**:
   ```bash
   pip install reader3[tts]
   # Or manually:
   pip install torch numpy sounddevice snac
   ```

4. **Load the Model** — In LM Studio, load the Orpheus model and start the server

5. **Configure Reader3** — The AI settings URL should point to LM Studio (default: `http://localhost:1234/v1`)

### Available Voices

Orpheus TTS supports 8 voices:

- **tara** (default) — Best overall voice for general use
- **leah**, **jess**, **mia**, **zoe** — Female voices
- **leo**, **dan**, **zac** — Male voices

### Adding Emotion

You can add emotion tags to text for expressive speech:

```
<giggle> <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>
```

### References

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) — Original model
- [orpheus-tts-local](https://github.com/isaiahbjork/orpheus-tts-local) — Local implementation guide

## Keyboard Shortcuts

| Shortcut   | Action          |
|------------|-----------------|
| `Ctrl/⌘+F` | Search          |
| `Ctrl/⌘+B` | Bookmarks panel |
| `Escape`   | Close modals    |

## Building Executable

```bash
uv run python build_executable.py
```

Creates `dist/Reader3.app` (macOS) or `dist/Reader3.exe` (Windows).

## License

MIT

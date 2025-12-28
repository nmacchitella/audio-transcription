# Audio Transcription

A Streamlit web app for transcribing long audio files using Groq's Whisper Large V3 API.

## Features

- **Smart chunking** - Handles audio files longer than Groq's 25MB limit by splitting into 10-minute chunks with overlap
- **Intelligent merging** - Uses longest common sequence algorithm to seamlessly join transcriptions
- **Multi-language support** - English, Italian, Spanish, French, German, Portuguese, Dutch, Russian, Chinese, Japanese, Korean
- **Multiple formats** - MP3, WAV, M4A, FLAC, OGG, MP4, WEBM
- **Password protection** - Simple authentication for the web interface

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install ffmpeg

- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

### 3. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
[groq]
api_key = "your-groq-api-key"

[app]
password = "your-password"
```

Get your Groq API key at https://console.groq.com

### 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, enter your password, upload an audio file, and transcribe.

## How it works

1. Audio is split into 10-minute chunks with 10-second overlap
2. Each chunk is transcribed via Groq API (very fast)
3. Overlapping transcriptions are merged using word alignment
4. Final transcript is available for download as text

## License

MIT

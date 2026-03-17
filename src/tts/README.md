# TeachingMonster TTS - Implementation Overview

This README documents the current TTS implementation used in TeachingMonster and the data flow for generating speech with word-level timing metadata.

Overview
- TTS module uses two main components:
  - Edge TTS for cloud-based, natural-sounding Chinese/ multilingual voices
  - faster-whisper for local ASR to extract word-level timestamps
- The goal is to produce an audio file per slide along with per-word timing information that can be aligned to the script text for synchronized narration visuals.

Core Components
1) Edge TTS
- Voice: zh-TW-HsiaoChenNeural (Mandarin, female, natural)
- Output: MP3 audio file
- Invocation: edge_tts.Communicate(text, voice).save(output_path)
- Notes: Free, requires internet access; no API key required.

2) faster-whisper (WhisperModel)
- Purpose: Transcribe the generated audio to obtain word-level timestamps
- Setup: WhisperModel("base", device="cpu", compute_type="int8")
- Output: segments with words and their start/end times

3) Alignment / Tokenization
- Tokenization: jieba (Chinese) to split the script into tokens
- Alignment: match tokenized script words to ASR words using a simple lookahead window and monotonicity adjustments
- Output: timings per slide as a list of (start, end) pairs per script word

Usage Flow (high-level)
- Call TTSModule.load() to load the ASR model
- Call TTSModule.run(scripts: List[str]) to generate:
  - output_root containing per-slide MP3s (1.mp3, 2.mp3, ...)
  - timings: List[List[Tuple[float, float]]] with timings per script word
  - asr_words: List[List[Tuple[str, float, float]]] with ASR word data per slide

Notes & Trade-offs
- Edge TTS provides natural voices but requires network connectivity.
- faster-whisper runs locally on CPU by default; consider GPU for speed if available.
- If ASR timestamps are missing or misaligned, a fallback timing distribution is used.

Extensibility
- This README can be extended to document planned Hugging Face TTS integration, or alternative voice providers.


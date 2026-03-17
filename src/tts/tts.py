"""
Improved TTS Module for TeachingMonster.
Uses Edge TTS (free, natural voice) + faster-whisper (word-level timestamps).

Input: list[str] - narration text per slide
Output: folder with mp3 files, list[list[tuple[float, float]]] - word timings per slide
"""

import asyncio
import os
import re
from typing import List, Tuple


class TTSModule:
    def __init__(self, output_root: str = "./dummy_tts"):
        self.output_root = output_root
        self.is_loaded = False
        # Edge TTS voice - natural sounding Chinese + English voices
        self.voice = "zh-TW-HsiaoChenNeural"  # Mandarin, female, natural
        self._asr_model = None  # faster-whisper

    def load(self):
        """Load ASR model for word-level timestamps."""
        from faster_whisper import WhisperModel
        
        # Use CPU for compatibility
        self._asr_model = WhisperModel("base", device="cpu", compute_type="int8")
        self.is_loaded = True

    @staticmethod
    def _cleanup_old_numbered_mp3(output_root: str) -> None:
        os.makedirs(output_root, exist_ok=True)
        for fn in os.listdir(output_root):
            if fn.endswith(".mp3") and fn[:-4].isdigit():
                try:
                    os.remove(os.path.join(output_root, fn))
                except OSError:
                    pass

    def _generate_tts_audio(self, text: str, output_path: str) -> None:
        """Generate TTS audio using Edge TTS."""
        import edge_tts
        
        async def _tts():
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
        
        asyncio.run(_tts())

    def _transcribe_with_timestamps(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """Get word-level timestamps using faster-whisper."""
        if self._asr_model is None:
            return []
        
        segments, _info = self._asr_model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
        )
        
        words = []
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                word = (w.word or "").strip()
                if word:
                    words.append((word, float(w.start), float(w.end)))
        return words

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text, handling both Chinese and English properly."""
        try:
            import jieba
            jieba.setLogLevel(20)  # suppress debug logs
            return list(jieba.cut(text))
        except ImportError:
            return text.split()

    @staticmethod
    def _normalize_token(s: str) -> str:
        """Normalize token for loose matching."""
        s = (s or "").strip().lower()
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        s = s.replace("\u2019", "'").replace("'", "")
        return s

    @staticmethod
    def _ensure_monotonic_nonneg(timings: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Ensure non-negative and monotonic increasing timings."""
        out = []
        prev_end = 0.0
        for s, e in timings:
            s, e = float(s), float(e)
            s = max(0.0, s)
            e = max(0.0, e)
            if e < s:
                e = s
            if s < prev_end:
                s = prev_end
            if e < s:
                e = s
            out.append((s, e))
            prev_end = e
        return out

    def _align_script_to_asr(
        self,
        script: str,
        asr_words: List[Tuple[str, float, float]],
        fallback_total_dur: float,
    ) -> List[Tuple[float, float]]:
        """Align script words to ASR timestamps with fallback."""
        script_words = self._tokenize(script)
        if not script_words:
            return []
        
        if not asr_words:
            total = max(0.25, fallback_total_dur)
            per = total / len(script_words)
            return [(i * per, (i + 1) * per) for i in range(len(script_words))]
        
        norm_asr = [self._normalize_token(w) for w, _, _ in asr_words]
        norm_script = [self._normalize_token(w) for w in script_words]
        
        aligned = []
        j = 0
        max_lookahead = 6
        
        for target in norm_script:
            if not target:
                if j < len(asr_words):
                    aligned.append((asr_words[j][1], asr_words[j][2]))
                else:
                    last = aligned[-1][1] if aligned else 0.0
                    aligned.append((last, last))
                continue
            
            found_idx = None
            for k in range(j, min(len(asr_words), j + max_lookahead)):
                if norm_asr[k] == target:
                    found_idx = k
                    break
            
            if found_idx is not None:
                _, s, e = asr_words[found_idx]
                aligned.append((s, e))
                j = found_idx + 1
            elif j < len(asr_words):
                _, s, e = asr_words[j]
                aligned.append((s, e))
                j += 1
            else:
                last = aligned[-1][1] if aligned else 0.0
                aligned.append((last, last))
        
        aligned = self._ensure_monotonic_nonneg(aligned)
        
        if aligned and aligned[-1][1] <= 0.0:
            total = max(0.25, fallback_total_dur)
            per = total / len(script_words)
            aligned = [(i * per, (i + 1) * per) for i in range(len(script_words))]
            aligned = self._ensure_monotonic_nonneg(aligned)
        
        return aligned

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import wave
            with wave.open(audio_path, "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception:
            return 1.0

    def run(self, scripts: List[str]) -> Tuple[str, List[List[Tuple[float, float]]], List[List]]:
        """Generate TTS audio and word-level timestamps for each script.
        
        Returns:
            (output_root, all_timings, all_asr_words)
        """
        assert self.is_loaded, "Call load() before run()"
        
        self._cleanup_old_numbered_mp3(self.output_root)
        all_timings = []
        all_asr_words = []
        
        for idx, script in enumerate(scripts, start=1):
            mp3_path = os.path.join(self.output_root, f"{idx}.mp3")
            
            # Handle empty/invalid script
            if not isinstance(script, str) or not script.strip():
                self._create_silent_mp3(mp3_path, duration=0.25)
                all_timings.append([])
                all_asr_words.append([])
                continue
            
            try:
                # Step 1: Generate TTS audio
                self._generate_tts_audio(script, mp3_path)
                
                # Step 2: Get word timestamps via ASR
                asr_words = self._transcribe_with_timestamps(mp3_path)
                
                # Step 3: Calculate total duration
                if asr_words:
                    fallback_dur = asr_words[-1][2]
                else:
                    fallback_dur = self._get_audio_duration(mp3_path)
                
                # Step 4: Align script to ASR timestamps
                timings = self._align_script_to_asr(script, asr_words, fallback_dur)
                all_timings.append(timings)
                all_asr_words.append(asr_words)
                
            except Exception as e:
                # Fallback
                words = self._tokenize(script)
                if words:
                    total = 3.0
                    per = total / len(words)
                    timings = [(i * per, (i + 1) * per) for i in range(len(words))]
                else:
                    timings = []
                timings = self._ensure_monotonic_nonneg(timings)
                all_timings.append(timings)
                all_asr_words.append([])
                
                try:
                    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                        self._create_silent_mp3(mp3_path, duration=timings[-1][1] if timings else 0.25)
                except Exception:
                    pass
        
        return self.output_root, all_timings, all_asr_words

    @staticmethod
    def _create_silent_mp3(path: str, duration: float = 0.25) -> None:
        """Create a minimal valid MP3 file (silent)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        mp3_header = bytes([
            0xFF, 0xFB, 0x90, 0x00,  # MPEG-1 Layer 3, 128kbps, 44100Hz
        ])
        
        num_frames = max(1, int(duration / 0.026))
        frame_size = 417
        
        with open(path, "wb") as f:
            for _ in range(num_frames):
                f.write(mp3_header)
                f.write(b"\x00" * (frame_size - len(mp3_header)))

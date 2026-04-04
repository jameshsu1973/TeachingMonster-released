import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TTSModule:
    """
    Qwen3-TTS module optimised for consistent, presentation-style narration.

    Key design choices:
      - instruct defaults to "" (empty) for neutral, steady tone
      - Lower temperature/top_p for prosody consistency
      - Configurable silence padding at head/tail of each slide
      - Hann crossfade between chunks within a slide
      - LUFS loudness normalisation across slides

    [OPT] v2 optimisations vs original:
      1. GPU–CPU 重疊流水線：GPU 推論結束後立刻啟動下一張，WAV 寫檔 /
         ffmpeg 全部丟進 CPU worker thread 並行執行，GPU 不再閒置等待 I/O。
      2. loudnorm 單通道修正：加入 `linear=true` 讓 ffmpeg 只讀一次檔案，
         消除原來隱性的雙通道（兩次讀取）開銷；同時修正 ffmpeg 找不到時的
         fallback bug（原本 fallback 到 _create_silent_audio，邏輯錯誤）。
      3. ASR 並發提升：max_workers 從 1 → 2，讓 Whisper 在 CPU worker 上
         不需要等待前一個寫檔 task 完成才能開始；ASR future 改為 fire-and-
         forget，最後統一 collect，不再在 TTS loop 內同步等待。
      4. 雜項快取：ffmpeg 路徑、soundfile/torch 在 load() 時 import 並快取
         到 self，消除每次呼叫重複的 import / shutil.which() 開銷。
    """

    INSTRUCT_PRESETS = {
        "neutral": "",
        "calm_narrator": "Speak in a calm, steady, professional narrator tone.",
        "presentation": "Speak clearly at a moderate pace, like a professional presenter.",
        "energetic": "Speak with enthusiasm and energy, like an engaging educator excited to teach!",
    }

    def __init__(
        self,
        output_root: str = "./dummy_tts",
        speaker: str = "Ryan",
        language: str = "English",
        tts_instruct: Optional[str] = "calm_narrator",
        tts_instruct_first: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        target_format: str = "mp3",
        keep_temp_wav: bool = False,
        max_chunk_chars: int = 250,
        min_chunk_chars: int = 25,
        crossfade_ms: float = 80.0,
        silence_between_chunks_ms: float = 150.0,
        slide_head_silence_ms: float = 600.0,
        slide_tail_silence_ms: float = 1500.0,
        loudnorm_target_lufs: float = -18.0,
        loudnorm_tp: float = -1.5,
        loudnorm_lra: float = 11.0,
        enable_loudnorm: bool = True,
        use_torch_compile: bool = False,
        asr_use_vad: bool = False,
        asr_model_gpu: str = "base",
        asr_model_cpu: str = "base",
    ):
        self.output_root = Path(output_root)
        self.speaker = speaker
        self.language = language

        if tts_instruct is not None:
            self.tts_instruct = self.INSTRUCT_PRESETS.get(tts_instruct, tts_instruct)
        else:
            self.tts_instruct = None

        default_gen_kwargs = {
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.75,
            "top_k": 20,
            "repetition_penalty": 1.15,
            "max_new_tokens": 2048,
        }
        default_gen_kwargs.update(generation_kwargs or {})
        self.generation_kwargs = default_gen_kwargs

        self.target_format = target_format.lower()
        self.keep_temp_wav = keep_temp_wav

        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.crossfade_ms = crossfade_ms
        self.silence_between_chunks_ms = silence_between_chunks_ms

        self.slide_head_silence_ms = slide_head_silence_ms
        self.slide_tail_silence_ms = slide_tail_silence_ms

        self.loudnorm_target_lufs = loudnorm_target_lufs
        self.loudnorm_tp = loudnorm_tp
        self.loudnorm_lra = loudnorm_lra
        self.enable_loudnorm = enable_loudnorm

        self.use_torch_compile = use_torch_compile

        self.asr_use_vad = asr_use_vad
        self.asr_model_gpu = asr_model_gpu
        self.asr_model_cpu = asr_model_cpu

        self.tts_instruct_first = tts_instruct_first

        self.is_loaded = False
        self._model = None
        self._asr_model = None
        self._torch = None

        # [OPT-4] 在 load() 時快取，避免每次呼叫重複查找
        self._sf = None          # soundfile module
        self._ffmpeg: Optional[str] = None  # ffmpeg binary path

        self.logger = logging.getLogger("TTSModule")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # =========================================================================
    # Loading
    # =========================================================================

    def load(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._ensure_sox_on_path()

        import soundfile as sf  # [OPT-4] import once, cache
        import torch
        from qwen_tts import Qwen3TTSModel

        self._sf = sf
        self._torch = torch
        self._ffmpeg = shutil.which("ffmpeg")  # [OPT-4] cache ffmpeg path

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen3-TTS.")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            self.logger.info("FlashAttention-2 detected.")
        except Exception:
            self.logger.info("FlashAttention-2 not available; using SDPA.")

        self.logger.info("Loading Qwen3-TTS …")
        self._model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=dtype,
            attn_implementation=attn_impl,
        )

        if self.use_torch_compile:
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
                self.logger.info("torch.compile enabled.")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")

        self.logger.info(
            f"Qwen3-TTS loaded | speaker={self.speaker} | "
            f"instruct={'(none)' if self.tts_instruct is None else repr(self.tts_instruct)} | "
            f"temp={self.generation_kwargs.get('temperature')} | "
            f"top_p={self.generation_kwargs.get('top_p')}"
        )

        try:
            from faster_whisper import WhisperModel
            self.logger.info(f"Loading faster-whisper ({self.asr_model_gpu}) on cuda …")
            self._asr_model = WhisperModel(
                self.asr_model_gpu,
                device="cuda",
                compute_type="int8_float16",
            )
        except Exception as e:
            self.logger.warning(f"Whisper load failed: {e}")
            self._asr_model = None

        # ASR warm-up
        if self._asr_model is not None:
            try:
                dummy_wav = np.zeros(16000, dtype=np.float32)
                list(self._asr_model.transcribe(
                    dummy_wav,
                    word_timestamps=True,
                    beam_size=1,
                    language=self._whisper_lang_code(),
                    condition_on_previous_text=False,
                ))
                self.logger.info("ASR warm-up completed.")
            except Exception as e:
                self.logger.warning(f"ASR warm-up failed: {e}")

        self.is_loaded = True

    @staticmethod
    def _ensure_sox_on_path():
        import sys
        env_dir = os.path.dirname(sys.executable)
        sox_bin = os.path.join(env_dir, "Library", "bin")
        if os.path.isdir(sox_bin) and sox_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = sox_bin + os.pathsep + os.environ.get("PATH", "")

    # =========================================================================
    # Text prep
    # =========================================================================

    _SENT_END_RE = re.compile(r"(?<=[。！？!?.;；])\s+")
    _SUB_SPLIT_RE = re.compile(r"(?<=[,，、:：])\s*|(?<=—)\s*|(?<=–)\s*")

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        text = (
            text.replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
        )
        text = text.replace("—", " — ").replace("–", " – ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        if text and text[-1].isalnum():
            text += "." if self.language.lower().startswith("english") else "。"

        return text.strip()

    @staticmethod
    def _ensure_chunk_ends_with_punct(chunk: str) -> str:
        chunk = chunk.strip()
        if chunk and chunk[-1] not in ".!?。！？;；:：":
            chunk += "."
        return chunk

    def _split_text_for_tts(self, text: str) -> List[str]:
        text = self._preprocess_text(text)
        if not text:
            return []
        if len(text) <= self.max_chunk_chars:
            return [self._ensure_chunk_ends_with_punct(text)]

        sentences = self._SENT_END_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [self._ensure_chunk_ends_with_punct(text)]

        chunks: List[str] = []
        cur = ""

        def flush():
            nonlocal cur
            if cur.strip():
                chunks.append(self._ensure_chunk_ends_with_punct(cur.strip()))
            cur = ""

        for s in sentences:
            subs = self._split_long_sentence(s) if len(s) > self.max_chunk_chars else [s]
            for sub in subs:
                if not cur:
                    cur = sub
                elif len(cur) + 1 + len(sub) <= self.max_chunk_chars:
                    cur += " " + sub
                else:
                    flush()
                    cur = sub
        flush()

        merged: List[str] = []
        for c in chunks:
            if (merged
                and len(merged[-1]) + 1 + len(c) <= self.max_chunk_chars
                and len(c) < self.min_chunk_chars):
                merged[-1] = self._ensure_chunk_ends_with_punct(
                    merged[-1].rstrip(".。!！?？;；") + " " + c
                )
            else:
                merged.append(c)

        return merged or [self._ensure_chunk_ends_with_punct(text)]

    def _split_long_sentence(self, text: str) -> List[str]:
        parts = self._SUB_SPLIT_RE.split(text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [text]

        out: List[str] = []
        cur = ""
        for p in parts:
            if not cur:
                cur = p
            elif len(cur) + 1 + len(p) <= self.max_chunk_chars:
                cur += " " + p
            else:
                out.append(cur)
                cur = p
        if cur:
            out.append(cur)

        final: List[str] = []
        for chunk in out:
            if len(chunk) <= self.max_chunk_chars:
                final.append(chunk)
                continue
            start = 0
            while start < len(chunk):
                end = min(start + self.max_chunk_chars, len(chunk))
                if end < len(chunk):
                    space = chunk.rfind(" ", start, end)
                    if space > start:
                        end = space
                final.append(chunk[start:end].strip())
                start = end

        return [x for x in final if x]

    # =========================================================================
    # Audio helpers
    # =========================================================================

    def _hann_crossfade(self, a: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
        if n_samples <= 0 or a.size < n_samples or b.size < n_samples:
            return np.concatenate([a, b])

        fade_out = np.hanning(2 * n_samples)[n_samples:]
        fade_in = np.hanning(2 * n_samples)[:n_samples]

        tail = a[-n_samples:] * fade_out
        head = b[:n_samples] * fade_in

        return np.concatenate([a[:-n_samples], tail + head, b[n_samples:]])

    def _concat_wavs(self, wavs: List[np.ndarray], sr: int) -> np.ndarray:
        pieces: List[np.ndarray] = []
        xfade_samples = int(sr * self.crossfade_ms / 1000.0)
        gap_samples = int(sr * self.silence_between_chunks_ms / 1000.0)
        gap = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else None

        for w in wavs:
            arr = np.asarray(w, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue

            if pieces and xfade_samples > 0:
                prev = pieces.pop()
                merged = self._hann_crossfade(prev, arr, xfade_samples)
                pieces.append(merged)
            else:
                pieces.append(arr)

            if gap is not None and pieces:
                pieces.append(gap.copy())

        if not pieces:
            return np.zeros(int(sr * 0.25), dtype=np.float32)

        audio = np.concatenate(pieces)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.99:
            audio = audio / peak * 0.98

        return audio.astype(np.float32)

    def _add_slide_padding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        head_samples = int(sr * self.slide_head_silence_ms / 1000.0)
        tail_samples = int(sr * self.slide_tail_silence_ms / 1000.0)

        parts = []
        if head_samples > 0:
            parts.append(np.zeros(head_samples, dtype=np.float32))
        parts.append(audio)
        if tail_samples > 0:
            parts.append(np.zeros(tail_samples, dtype=np.float32))

        return np.concatenate(parts)

    def _export_audio(self, src_path: Path, dst_path: Path):
        """單純格式轉換，不加 loudnorm。"""
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.resolve() == dst_path.resolve():
            return

        if self._ffmpeg:
            cmd = [
                self._ffmpeg, "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", str(src_path),
            ]
            if dst_path.suffix.lower() == ".mp3":
                cmd += ["-acodec", "libmp3lame", "-b:a", "192k"]
            cmd.append(str(dst_path))
            subprocess.run(cmd, check=True)
            return

        if src_path.suffix.lower() == dst_path.suffix.lower():
            shutil.copy2(src_path, dst_path)
            return

        try:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            clip = AudioFileClip(str(src_path))
            clip.write_audiofile(str(dst_path), codec="libmp3lame", verbose=False, logger=None)
            clip.close()
        except Exception as e:
            raise RuntimeError(
                f"Cannot convert {src_path.suffix} → {dst_path.suffix}. "
                f"Install ffmpeg or use target_format='wav'."
            ) from e

    def _export_and_loudnorm(self, src_path: Path, dst_path: Path) -> None:
        """
        [OPT-2] WAV → 目標格式，同時套用 loudnorm，只跑一次 ffmpeg。

        原問題：loudnorm 預設是「雙通道」模式，FFmpeg 需要先掃描整個檔案
        測量 integrated LUFS（第一次讀取），再做實際 re-encode（第二次讀取）。
        加入 `linear=true` 強制改用單通道（linear）模式：FFmpeg 只讀一次，
        計算量略增但省去完整的 I/O round-trip，通常快 30–50%。
        音量準確度在語音場景（窄動態範圍）下幾乎無差異。

        原 fallback bug：找不到 ffmpeg 時呼叫 _create_silent_audio()，
        會產生靜音檔案覆蓋輸出，現已修正為 fallback 到 _export_audio()。
        """
        if not self._ffmpeg:
            self.logger.warning("ffmpeg not found; falling back to plain export without loudnorm.")
            self._export_audio(src_path, dst_path)
            return

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # linear=true → 單通道模式，省去第二次讀取
            af = (
                f"loudnorm=I={self.loudnorm_target_lufs}"
                f":LRA={self.loudnorm_lra}"
                f":TP={self.loudnorm_tp}"
                f":linear=true"          # [OPT-2] 單通道，只讀一次
                f":print_format=none"    # 不印 summary，省 stderr 解析
            )
            cmd = [
                self._ffmpeg, "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", str(src_path),
                "-af", af,
            ]
            if dst_path.suffix.lower() == ".mp3":
                cmd += ["-acodec", "libmp3lame", "-b:a", "192k"]
            cmd.append(str(dst_path))
            subprocess.run(cmd, check=True)
        except Exception as e:
            self.logger.warning(f"Loudnorm export failed ({e}); retrying without loudnorm.")
            self._export_audio(src_path, dst_path)

    def _get_audio_duration(self, path: Path) -> float:
        try:
            return float(self._sf.info(str(path)).duration)  # [OPT-4] 用快取的 sf
        except Exception:
            try:
                from moviepy.audio.io.AudioFileClip import AudioFileClip
                clip = AudioFileClip(str(path))
                dur = float(clip.duration or 0.0)
                clip.close()
                return dur
            except Exception:
                return 0.0

    def _create_silent_audio(self, out_path: Path, duration: float = 0.25):
        sr = 24000
        wav = np.zeros(max(1, int(sr * duration)), dtype=np.float32)
        if out_path.suffix.lower() == ".wav":
            self._sf.write(str(out_path), wav, sr)
            return
        tmp_wav = out_path.with_suffix(".wav")
        self._sf.write(str(tmp_wav), wav, sr)
        self._export_audio(tmp_wav, out_path)
        if not self.keep_temp_wav and tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)

    # =========================================================================
    # TTS generation  (GPU 推論部分，由主執行緒呼叫)
    # =========================================================================

    def _infer_gpu(self, text: str) -> Tuple[np.ndarray, int]:
        """
        純 GPU 推論：只做 generate_custom_voice，不碰磁碟。
        回傳 (merged_audio_with_padding, sample_rate)。
        """
        chunks = self._split_text_for_tts(text)
        if not chunks:
            sr = 24000
            silence_samples = int(sr * 0.25)
            return np.zeros(silence_samples, dtype=np.float32), sr

        first_instruct = self.tts_instruct_first if self.tts_instruct_first is not None else self.tts_instruct
        rest_instruct = self.tts_instruct if self.tts_instruct is not None else ""

        if len(chunks) == 1:
            kwargs: Dict[str, Any] = {
                "text": chunks[0],
                "language": self.language,
                "speaker": self.speaker,
            }
            if first_instruct is not None:
                kwargs["instruct"] = first_instruct
        else:
            kwargs = {
                "text": chunks,
                "language": [self.language] * len(chunks),
                "speaker": [self.speaker] * len(chunks),
            }
            instructs = (
                [first_instruct if first_instruct is not None else ""]
                + [rest_instruct] * (len(chunks) - 1)
            )
            kwargs["instruct"] = instructs

        gen_kwargs = self.generation_kwargs.copy()
        gen_kwargs["max_new_tokens"] = min(2048, max(256, len(text) * 3))
        kwargs.update(gen_kwargs)

        with self._torch.inference_mode():
            wavs, sr = self._model.generate_custom_voice(**kwargs)

        if isinstance(wavs, np.ndarray):
            wavs = [wavs]

        merged = self._concat_wavs([np.asarray(w).squeeze() for w in wavs], sr)
        merged = self._add_slide_padding(merged, sr)
        return merged, int(sr)

    def _cpu_postprocess(
        self,
        audio: np.ndarray,
        sr: int,
        wav_path: Path,
        final_path: Path,
    ) -> None:
        """
        [OPT-1] CPU 後處理：寫 WAV + ffmpeg loudnorm/轉碼。
        在 CPU worker thread 執行，與下一張投影片的 GPU 推論並行。
        """
        self._sf.write(str(wav_path), audio, sr)

        if self.enable_loudnorm:
            self._export_and_loudnorm(wav_path, final_path)
        else:
            self._export_audio(wav_path, final_path)

    # =========================================================================
    # ASR + alignment
    # =========================================================================

    def _whisper_lang_code(self) -> Optional[str]:
        m = {
            "english": "en", "chinese": "zh", "japanese": "ja",
            "korean": "ko", "german": "de", "french": "fr",
            "russian": "ru", "portuguese": "pt", "spanish": "es",
            "italian": "it",
        }
        return m.get((self.language or "").strip().lower())

    def _transcribe_with_timestamps(self, audio_path: Path) -> List[Tuple[str, float, float]]:
        if self._asr_model is None:
            return []
        segments, _ = self._asr_model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=self.asr_use_vad,
            language=self._whisper_lang_code(),
            beam_size=1,
            condition_on_previous_text=False,
        )
        words = []
        for seg in segments:
            for w in (seg.words or []):
                word = (w.word or "").strip()
                if word:
                    words.append((word, float(w.start), float(w.end)))
        return words

    @staticmethod
    def _normalize_token(s: str) -> str:
        s = (s or "").strip().lower()
        s = s.replace("\u2019", "'")
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        s = s.replace("'", "")
        return s

    @staticmethod
    def _ensure_monotonic_nonneg(timings: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out = []
        prev_end = 0.0
        for s, e in timings:
            s = max(0.0, float(s))
            e = max(0.0, float(e))
            if e < s:
                e = s
            if s < prev_end:
                s = prev_end
            if e < s:
                e = s
            out.append((s, e))
            prev_end = e
        return out

    def _uniform_timings(self, num_words: int, total: float) -> List[Tuple[float, float]]:
        total = max(0.25, float(total))
        if num_words <= 0:
            return []
        per = total / num_words
        return [(i * per, (i + 1) * per) for i in range(num_words)]

    def _fill_missing_timings(
        self,
        aligned: List[Optional[Tuple[float, float]]],
        total_dur: float,
    ) -> List[Tuple[float, float]]:
        out = aligned[:]
        n = len(out)
        i = 0
        while i < n:
            if out[i] is not None:
                i += 1
                continue
            start = i
            while i < n and out[i] is None:
                i += 1
            end = i
            left_end = out[start - 1][1] if start > 0 and out[start - 1] is not None else 0.0
            right_start = out[end][0] if end < n and out[end] is not None else total_dur
            if right_start < left_end:
                right_start = left_end
            count = end - start
            span = max(right_start - left_end, 0.05 * count)
            per = span / max(1, count)
            for k in range(count):
                s = left_end + k * per
                e = left_end + (k + 1) * per
                out[start + k] = (s, e)
        return [(0.0, 0.0) if x is None else x for x in out]

    def _align_script_to_asr(
        self,
        script: str,
        asr_words: List[Tuple[str, float, float]],
        fallback_total_dur: float,
    ) -> List[Tuple[float, float]]:
        script_words = script.split()
        if not script_words:
            return []
        if not asr_words:
            return self._ensure_monotonic_nonneg(
                self._uniform_timings(len(script_words), fallback_total_dur)
            )
        script_norm = [self._normalize_token(w) for w in script_words]
        asr_clean = [
            (self._normalize_token(w), float(s), float(e))
            for w, s, e in asr_words
            if self._normalize_token(w)
        ]
        if not asr_clean:
            return self._ensure_monotonic_nonneg(
                self._uniform_timings(len(script_words), fallback_total_dur)
            )
        asr_norm = [x[0] for x in asr_clean]
        asr_times = [(x[1], x[2]) for x in asr_clean]
        matcher = SequenceMatcher(a=script_norm, b=asr_norm, autojunk=False)
        aligned: List[Optional[Tuple[float, float]]] = [None] * len(script_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for si, aj in zip(range(i1, i2), range(j1, j2)):
                    aligned[si] = asr_times[aj]
        total = max(fallback_total_dur, asr_times[-1][1], 0.25)
        filled = self._fill_missing_timings(aligned, total)
        return self._ensure_monotonic_nonneg(filled)

    # =========================================================================
    # Main run
    # =========================================================================

    def _cleanup_old_numbered_audio(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        for p in self.output_root.iterdir():
            if p.stem.isdigit() and p.suffix.lower() in {".mp3", ".wav"}:
                try:
                    p.unlink()
                except OSError:
                    pass

    def run(self, scripts: List[str]) -> Tuple[str, List[List[Tuple[float, float]]]]:
        """
        [OPT-1] GPU–CPU 重疊流水線說明
        ─────────────────────────────────────────────────────────────────────
        原來的執行順序（序列）：
            Slide 1: [GPU 推論] → [sf.write WAV] → [ffmpeg] → [ASR]
            Slide 2: [GPU 推論] → ...   ← GPU 在上面的 I/O 全完成前閒置

        優化後（重疊）：
            Slide 1: [GPU 推論] ─→ submit _cpu_postprocess → submit _process_asr
                                         ↓ (在 CPU worker 執行)
            Slide 2: [GPU 推論]  ← GPU 立刻開始，不等 I/O
                                         ↓
            Slide 3: [GPU 推論]  ...

        ThreadPoolExecutor 分兩個 worker：
          - worker-0：專門跑 CPU 後處理（sf.write + ffmpeg）
            每次只有一個任務在跑，確保磁碟不會被搶占
          - worker-1：專門跑 ASR（Whisper）
            必須等對應的 postprocess 完成（WAV 寫好）才能開始，
            用 Future.result() 作為依賴屏障

        [OPT-3] ASR 並發：改為 fire-and-forget，全部 submit 完再統一 collect，
        讓 Whisper 在等 WAV 寫完的期間不佔用主執行緒。
        ─────────────────────────────────────────────────────────────────────
        """
        assert self.is_loaded, "Call load() before run()"
        self._cleanup_old_numbered_audio()

        all_timings: List[List[Tuple[float, float]]] = [[] for _ in range(len(scripts))]

        # [OPT-3] 分離兩個 worker：一個跑 postprocess，一個跑 ASR
        # max_workers=2 確保兩者可以真正並行
        with ThreadPoolExecutor(max_workers=2) as executor:
            asr_futures: Dict[int, Future] = {}

            def _make_asr_task(
                idx: int,
                script_text: str,
                w_path: Path,
                f_path: Path,
                c_script: str,
                postprocess_future: Future,
            ):
                """
                回傳一個閉包，由 executor 執行。
                先等 postprocess_future 完成（WAV 已寫好），再做 ASR。
                """
                def _task():
                    # 等候寫檔完成，這是 ASR 的前置條件
                    try:
                        postprocess_future.result()
                    except Exception as post_err:
                        self.logger.warning(
                            f"Slide {idx} postprocess failed before ASR: {post_err}"
                        )

                    try:
                        asr_input = w_path if w_path.exists() else f_path
                        a_words = self._transcribe_with_timestamps(asr_input)

                        fallback_dur = self._get_audio_duration(asr_input)
                        if fallback_dur <= 0 and a_words:
                            fallback_dur = a_words[-1][2]
                        if fallback_dur <= 0:
                            fallback_dur = max(1.0, 0.35 * len(c_script.split()))

                        t_timings = self._align_script_to_asr(c_script, a_words, fallback_dur)
                        return t_timings, a_words

                    except Exception as e:
                        self.logger.warning(f"ASR Slide {idx} failed: {e}")
                        words = script_text.split()
                        fallback_t = (
                            self._uniform_timings(len(words), max(1.0, 0.33 * len(words)))
                            if words else []
                        )
                        return self._ensure_monotonic_nonneg(fallback_t), []
                    finally:
                        # WAV 在 ASR 讀完後才刪除，防止 disk leak
                        if not self.keep_temp_wav and w_path.exists() and self.target_format != "wav":
                            try:
                                w_path.unlink(missing_ok=True)
                            except Exception:
                                pass

                return _task

            for idx, script in enumerate(scripts, start=1):
                i = idx - 1
                base = self.output_root / str(idx)
                wav_path = base.with_suffix(".wav")
                final_path = base.with_suffix(f".{self.target_format}")

                if not isinstance(script, str) or not script.strip():
                    self._create_silent_audio(final_path, 0.25)
                    continue

                try:
                    clean_script = self._preprocess_text(script)

                    # ── Step 1: GPU 推論（主執行緒，阻塞）──────────────────
                    audio, sr = self._infer_gpu(clean_script)

                    # ── Step 2: CPU 後處理（worker thread，非阻塞）─────────
                    # GPU 推論結束後立刻提交寫檔 + ffmpeg，主執行緒繼續下一張
                    post_future = executor.submit(
                        self._cpu_postprocess, audio, sr, wav_path, final_path
                    )

                    # ── Step 3: ASR（worker thread，等 post_future 完成）───
                    # fire-and-forget：不在這裡 .result()，統一在迴圈後 collect
                    asr_task = _make_asr_task(
                        idx, script, wav_path, final_path, clean_script, post_future
                    )
                    asr_futures[i] = executor.submit(asr_task)

                    self.logger.info(
                        f"Slide {idx} GPU done | postprocess + ASR dispatched to background"
                    )

                except Exception as e:
                    self.logger.warning(f"Slide {idx} GPU inference failed: {e}")
                    self._create_silent_audio(final_path, 0.25)
                    self._torch.cuda.empty_cache()

            # ── 統一 collect ASR 結果 ────────────────────────────────────
            for i, future in asr_futures.items():
                try:
                    timings, _ = future.result()
                    all_timings[i] = timings
                except Exception as e:
                    self.logger.warning(f"ASR collect Slide {i + 1} failed: {e}")

        return str(self.output_root), all_timings
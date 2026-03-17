"""
Test script for the improved TTS module.
"""

import os
import importlib.util

# Direct import bypassing src package
spec = importlib.util.spec_from_file_location(
    "tts", os.path.join(os.path.dirname(__file__), "src", "tts", "tts.py")
)
tts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tts_module)
TTSModule = tts_module.TTSModule

def main():
    print("=" * 60)
    print("TTS Module Test - Edge TTS + faster-whisper")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    tts = TTSModule(output_root=output_dir)
    
    print("\n[1/3] Loading models...")
    tts.load()
    print("  Models loaded OK!")
    
    scripts = [
        "今天我們要學習 Self-Attention 的機制，這是 Transformer 模型中最核心的概念。",
        "Self-Attention 允許模型在處理每個詞時，同時關注序列中的所有其他詞。",
        "簡單來說，它會計算 Query、Key 和 Value 三個矩陣的注意力分數。",
    ]
    
    print(f"\n[2/3] Generating audio for {len(scripts)} slides...")
    for i, s in enumerate(scripts, 1):
        print(f"  Slide {i}: {s[:50]}...")
    
    output_root, all_timings, all_asr_words = tts.run(scripts)
    
    print(f"\n[3/3] Results:")
    print(f"  Output folder: {os.path.abspath(output_root)}")
    
    for i, (script, timings, asr_words) in enumerate(zip(scripts, all_timings, all_asr_words), 1):
        mp3_path = os.path.join(output_root, f"{i}.mp3")
        exists = os.path.exists(mp3_path)
        size = os.path.getsize(mp3_path) if exists else 0
        
        tokenized = tts._tokenize(script)
        
        print(f"\n  --- Slide {i} ---")
        print(f"  MP3: {os.path.basename(mp3_path)} ({size} bytes)")
        print(f"  Script (raw split): {len(script.split())} tokens")
        print(f"  Script (jieba):     {len(tokenized)} tokens")
        print(f"  ASR detected:       {len(asr_words)} words")
        print(f"  Aligned timings:    {len(timings)} entries")
        
        if timings:
            print(f"  Audio duration:     {timings[-1][1]:.2f}s")
            print(f"  First 5 tokenized words with timings:")
            for word, (s, e) in zip(tokenized[:5], timings[:5]):
                print(f"    '{word}': {s:.3f}s - {e:.3f}s")
            if len(timings) > 5:
                print(f"    ... and {len(timings) - 5} more")
        
        if asr_words:
            print(f"  First 5 ASR words:")
            for word, s, e in asr_words[:5]:
                print(f"    '{word}': {s:.3f}s - {e:.3f}s")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

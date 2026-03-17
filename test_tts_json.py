"""Quick test with JSON output for easy viewing."""
import os, json, importlib.util

spec = importlib.util.spec_from_file_location("tts", os.path.join("src", "tts", "tts.py"))
tts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tts_module)

tts = tts_module.TTSModule(output_root="./test_output")
tts.load()

scripts = [
    "今天我們要學習 Self-Attention 的機制，這是 Transformer 模型中最核心的概念。",
    "Self-Attention 允許模型在處理每個詞時，同時關注序列中的所有其他詞。",
    "簡單來說，它會計算 Query、Key 和 Value 三個矩陣的注意力分數。",
]

output_root, all_timings, all_asr_words = tts.run(scripts)

result = []
for i, (script, timings) in enumerate(zip(scripts, all_timings), 1):
    words = tts._tokenize(script)
    result.append({
        "slide": i,
        "script": script,
        "tokenized_words": words,
        "num_tokens": len(words),
        "mp3_file": f"{i}.mp3",
        "mp3_size_bytes": os.path.getsize(os.path.join(output_root, f"{i}.mp3")),
        "duration_sec": round(timings[-1][1], 2) if timings else 0,
        "timings": [{"word": w, "start": round(s, 3), "end": round(e, 3)} 
                     for w, (s, e) in zip(words, timings)],
    })

with open("test_output/result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Done! Check test_output/result.json and the .mp3 files.")

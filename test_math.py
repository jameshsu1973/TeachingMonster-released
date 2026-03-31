"""Quick math formula TTS test"""
import sys
sys.path.insert(0, "src/tts")
from tts import TTSModule

tts = TTSModule(
    output_root="test_math_output",
    speaker="Ryan",
    language="English",
    tts_instruct="neutral",
)
tts.load()

math_texts = [
    # 簡單分數
    "The quadratic formula is x equals negative b plus or minus the square root of b squared minus four a c, all over two a.",
    # LaTeX 風格
    r"f of x equals the integral from negative infinity to infinity of e to the negative x squared, divided by the square root of two pi, dx.",
    # 複雜希臘符號
    "Delta equals sigma sub i equals one to n of w sub i times x sub i.",
    # 極限與微分
    "The derivative of e to the x is e to the x. The limit as x approaches zero of sine x over x equals one.",
    # 矩陣
    "Matrix A equals the three by three matrix with entries a one one, a one two, a one three in the first row.",
    # 總結
    "In summary, Einstein's equation E equals m c squared shows the equivalence of mass and energy.",
]

math_texts = [
    # 簡單分數
    "The quadratic formula is x equals negative b plus or minus the square root of b squared minus four a c, all over two a.",
    # LaTeX 風格
    "f of x equals the integral from negative infinity to infinity of e to the negative x squared, divided by the square root of two pi, dx.",
    # 複雜希臘符號
    "Delta equals the sum from i equals one to n of w sub i times x sub i.",
    # 極限與微分
    "The derivative of e to the x is e to the x. The limit as x approaches zero of sine x over x equals one.",
    # 矩陣
    "Matrix A equals a three by three matrix with entries a one one, a one two, a one three in the first row.",
    # 總結
    "In summary, Einstein's equation E equals m c squared shows the equivalence of mass and energy.",
]

root, timings, asr = tts.run(math_texts)
print(f"Done! Check {root}")

"""
LaTeX Math → Spoken English converter
======================================
插入在 TTS 之前，把 LaTeX 公式轉成自然口語文字。
"""

import re
from typing import Dict, Tuple


class LatexToSpeech:
    """
    規則式 LaTeX → 語音文字轉換器。

    使用方式：
        converter = LatexToSpeech()
        spoken = converter.convert(r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}")
        # → "x equals the fraction with numerator negative b plus or minus square root of b squared minus four a c, and denominator two a"
    """

    # ── 希臘字母對照表 ──────────────────────────────────────────────────
    GREEK = {
        "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
        "epsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta",
        "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu",
        "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho",
        "sigma": "sigma", "tau": "tau", "upsilon": "upsilon", "phi": "phi",
        "chi": "chi", "psi": "psi", "omega": "omega",
        "Gamma": "Gamma", "Delta": "Delta", "Theta": "Theta", "Lambda": "Lambda",
        "Xi": "Xi", "Pi": "Pi", "Sigma": "Sigma", "Phi": "Phi",
        "Psi": "Psi", "Omega": "Omega",
        # 常見大寫
        "Sigma": "Sigma", "Omega": "Omega", "Pi": "Pi",
        # 特殊
        "infty": "infinity", "lam": "lambda",  # 常見錯誤拼寫
    }

    # ── 常見數學符號 → 文字 ──────────────────────────────────────────────
    SYMBOLS: Dict[str, str] = {
        "\\le": "less than or equal to",
        "\\leq": "less than or equal to",
        "\\ge": "greater than or equal to",
        "\\geq": "greater than or equal to",
        "\\neq": "not equal to",
        "\\approx": "approximately equal to",
        "\\equiv": "equivalent to",
        "\\pm": "plus or minus",
        "\\mp": "minus or plus",
        "\\times": "times",
        "\\cdot": "times",
        "\\div": "divided by",
        "\\forall": "for all",
        "\\exists": "there exists",
        "\\in": "in",
        "\\notin": "not in",
        "\\subset": "subset of",
        "\\subseteq": "subset of or equal to",
        "\\cup": "union",
        "\\cap": "intersection",
        "\\emptyset": "empty set",
        "\\rightarrow": "goes to",
        "\\Rightarrow": "implies",
        "\\leftarrow": "comes from",
        "\\Leftarrow": "is implied by",
        "\\leftrightarrow": "if and only if",
        "\\infty": "infinity",
        "\\partial": "partial derivative",
        "\\nabla": "nabla",
        "\\sum": "sum",
        "\\prod": "product",
        "\\int": "integral",
        "\\iint": "double integral",
        "\\oint": "contour integral",
        "\\sin": "sine",
        "\\cos": "cosine",
        "\\tan": "tangent",
        "\\cot": "cotangent",
        "\\sec": "secant",
        "\\csc": "cosecant",
        "\\arcsin": "arc sine",
        "\\arccos": "arc cosine",
        "\\arctan": "arc tangent",
        "\\sinh": "hyperbolic sine",
        "\\cosh": "hyperbolic cosine",
        "\\tanh": "hyperbolic tangent",
        "\\log": "log",
        "\\ln": "natural log",
        "\\exp": "exponential",
        "\\lim": "limit",
        "\\max": "maximum",
        "\\min": "minimum",
        "\\sup": "supremum",
        "\\inf": "infimum",
        "\\det": "determinant",
        "\\deg": "degree",
        "\\mod": "modulo",
    }

    # ── 分數對照表 ──────────────────────────────────────────────────────
    ORDINALS = {
        0: "zeroth", 1: "first", 2: "second", 3: "third", 4: "fourth",
        5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
        10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
        14: "fourteenth", 15: "fifteenth", 16: "sixteenth",
        17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
        20: "twentieth", 21: "twenty-first", 22: "twenty-second",
        30: "thirtieth", 40: "fortieth", 50: "fiftieth",
        60: "sixtieth", 70: "seventieth", 80: "eightieth",
        90: "ninetieth", 100: "hundredth",
    }

    # ── 建構子 ──────────────────────────────────────────────────────────
    def __init__(self, *, simplify_fractions: bool = True, ordinal_threshold: int = 20):
        self.simplify_fractions = simplify_fractions
        self.ordinal_threshold = ordinal_threshold

    # ════════════════════════════════════════════════════════════════════
    # 公開 API
    # ════════════════════════════════════════════════════════════════════

    def convert(self, text: str) -> str:
        """
        把含 LaTeX 的文字轉成純口語版本。
        會遍歷整個字串，跳過 $...$ 和 $$...$$ 內的數學公式並轉換。
        """
        result = []
        i = 0
        math_depth = 0   # 是否在公式環境內
        math_start = 0

        while i < len(text):
            # 進入數學環境
            if i < len(text) - 1:
                if text[i] == '$' and text[i+1] == '$':
                    if math_depth == 0:
                        math_start = i + 2
                    math_depth += 2
                    i += 2
                    continue
                elif text[i] == '$':
                    if math_depth == 0:
                        math_start = i + 1
                    math_depth += 1
                    i += 1
                    continue
                elif i < len(text) - 5 and text[i:i+6] == '\\begin':
                    if math_depth == 0:
                        math_start = i
                    math_depth += 1
                    i += 6
                    continue
                elif i < len(text) - 5 and text[i:i+5] == '\\end[':
                    if math_depth > 0:
                        math_depth -= 1
                    i += 5
                    continue
                elif text[i:i+4] == '\\end':
                    if math_depth > 0:
                        math_depth -= 1
                    i += 4
                    continue

            if math_depth > 0:
                i += 1
                continue

            result.append(text[i])
            i += 1

        # 把 ansi 顏色碼之類的過濾掉
        clean = re.sub(r'\$+', '', text)
        # 如果沒有數學環境，整段當作數學處理
        if '$' not in text and '\\\\(' not in text and '\\\\[' not in text and '\\begin' not in text:
            return self._convert_latex(clean)
        return text

    def convert_inline(self, latex: str) -> str:
        """只轉換一段純 LaTeX 公式（不含 $ 符號）"""
        return self._convert_latex(latex.strip())

    # ════════════════════════════════════════════════════════════════════
    # 內部：主轉換邏輯
    # ════════════════════════════════════════════════════════════════════

    def _convert_latex(self, latex: str) -> str:
        latex = latex.strip()

        # ── 基本清理 ───────────────────────────────────────────────────
        latex = latex.strip()
        # 移除 displaymath / wrap
        latex = re.sub(r'^\$\$|\$\$$', '', latex)
        latex = re.sub(r'^\\\(|\\\)$$', '', latex)
        latex = re.sub(r'^\\\[|\\\]$$', '', latex)

        # ── 遞迴：先把嵌套的結構剝開 ──────────────────────────────────
        # 從最複雜的結構開始處理
        latex = self._convert_envs(latex)       # equation, align, gather ...
        latex = self._convert_fractions(latex)   # \frac{a}{b}
        latex = self._convert_limits(latex)     # \lim, \sum, \int from/to (要在 sup/sub 之前)
        latex = self._convert_superscripts(latex) # x^2, x^{23}  (只在原始 LaTeX)
        latex = self._convert_subscripts(latex)  # x_1, x_{ij}
        latex = self._convert_roots(latex)        # \sqrt{x}, \sqrt[n]{x}
        latex = self._convert_primes(latex)       # x'
        latex = self._convert_bar(latex)          # \bar{x}, \overline{ab}
        latex = self._convert_hat_tilde(latex)    # \hat{x}, \tilde{x}
        latex = self._convert_greek(latex)        # \alpha, \beta ...
        latex = self._convert_symbols(latex)      # \leq, \times ...
        latex = self._convert_functions(latex)    # \sin, \log ...
        latex = self._convert_binomial(latex)    # \binom{a}{b}
        latex = self._convert_matrix(latex)       # \begin{matrix}...
        latex = self._convert_ddots(latex)        # \cdots \vdots \ddots
        latex = self._convert_text_mode(latex)    # \text{...}
        latex = self._convert_braces(latex)       # {abc} → abc

        # ── 通用替換（符號、空白整理）─────────────────────────────────
        latex = self._cleanup(latex)

        return latex

    # ── 矩陣環境 ──────────────────────────────────────────────────────
    def _convert_matrix(self, latex: str) -> str:
        """處理 \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} 等"""
        for mname in ['matrix', 'bmatrix', 'vmatrix', 'pmatrix']:
            start_pat = rf'\\begin\{{{mname}\}}'
            end_pat = rf'\\end\{{{mname}\}}'
            # 找配對的 begin/end
            pattern = r'\\begin\{' + mname + r'\}\s*((?:[^{}&\\]|\\.)*?)\\end\{' + mname + r'\}'
            def replacer(m, name=mname):
                content = m.group(1)
                # 用 & 分欄，\\\\ 分列
                raw_entries = re.split(r'&', content)
                entries = []
                for entry in raw_entries:
                    parts = re.split(r'\\\\', entry)
                    for p in parts:
                        p = p.strip()
                        if p:
                            entries.append(self.convert_inline(p))
                n = len(entries)
                # 推斷行列
                for r in range(2, 5):
                    for c in range(2, 5):
                        if r * c == n:
                            row_word = {2: 'two by two', 3: 'three by three', 4: 'four by four'}.get(r, f'{r} by {c}')
                            readable = ', '.join(entries)
                            if name == 'vmatrix':
                                return f'determinant of the {row_word} matrix with entries {readable}'
                            if name == 'bmatrix':
                                return f'the {row_word} matrix with entries {readable}'
                            return f'{row_word} matrix with entries {readable}'
                return f'matrix with entries {", ".join(entries)}'
            latex = re.sub(pattern, replacer, latex, flags=re.DOTALL)
        return latex

    # ── \frac{a}{b} ───────────────────────────────────────────────────
    def _convert_fractions(self, latex: str) -> str:
        """把 \frac{numerator}{denominator} 轉成口語分數"""
        while r'\frac' in latex:
            pattern = r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            match = re.search(pattern, latex)
            if not match:
                break
            num = match.group(1).strip()
            den = match.group(2).strip()

            num_s = self._convert_latex(num)
            den_s = self._convert_latex(den)

            # 特殊分數
            if num == '1' and den_s:
                den_ord = self._to_ordinal(den_s)
                if den_ord:
                    spoken = f"1 over {den_s}" if den_s == den else f"1 over {den_s}"
                    if self.simplify_fractions:
                        spoken = f"1 over {den_s}"
                    latex = latex[:match.start()] + spoken + latex[match.end():]
                    continue

            if self.simplify_fractions:
                spoken = f"the fraction with numerator {num_s}, and denominator {den_s}"
            else:
                spoken = f"the fraction {num_s} over {den_s}"
            latex = latex[:match.start()] + spoken + latex[match.end():]
        return latex

    # ── 上標（次方）──────────────────────────────────────────────────
    def _convert_superscripts(self, latex: str) -> str:
        """處理 x^2, x^{23}, x^{n+1}"""
        # 先處理 ^{...}
        while re.search(r'\^{', latex):
            pattern = r'\^\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            def replacer(m):
                exp = m.group(1)
                if exp in ('+', '-', '\'', '"'):
                    return m.group(0)
                exp_s = self._convert_latex(exp)
                return f" to the power of {exp_s}"
            latex = re.sub(pattern, replacer, latex)
        # 再處理 ^.
        pattern = r'\^([a-zA-Z0-9\-\+\.]+)'
        def replacer2(m):
            exp = m.group(1)
            if exp in ('+', '-', '.', '2', '3', '1'):
                if exp == '2':
                    return " squared"
                if exp == '3':
                    return " cubed"
                if exp == '1':
                    return " to the power of one"
            return f" to the power of {exp}"
        latex = re.sub(pattern, replacer2, latex)
        return latex

    # ── 下標 ───────────────────────────────────────────────────────────
    def _convert_subscripts(self, latex: str) -> str:
        """處理 x_1, x_{ij}, x_i+1"""
        while re.search(r'_{', latex):
            pattern = r'_\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            def replacer(m):
                sub = m.group(1)
                sub_s = self._convert_latex(sub)
                return f" sub {sub_s}"
            latex = re.sub(pattern, replacer, latex)
        pattern = r'_([a-zA-Z0-9\+\-]+)'
        def replacer2(m):
            sub = m.group(1)
            return f" sub {sub}"
        latex = re.sub(pattern, replacer2, latex)
        return latex

    # ── 根號 ──────────────────────────────────────────────────────────
    def _convert_roots(self, latex: str) -> str:
        """處理 \sqrt{x}, \sqrt[n]{x}"""
        # \sqrt[n]{x}
        pattern = r'\\sqrt\[([^\]]+)\]\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        def replacer(m):
            n = m.group(1)
            radicand = m.group(2)
            if n == '2':
                n_s = "square"
            elif n == '3':
                n_s = "cube"
            else:
                n_s = self._convert_latex(n)
            rad_s = self._convert_latex(radicand)
            return f"the {n_s} root of {rad_s}"
        latex = re.sub(pattern, replacer, latex)
        # \sqrt{x}
        pattern2 = r'\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        def replacer2(m):
            radicand = m.group(1)
            rad_s = self._convert_latex(radicand)
            return f"the square root of {rad_s}"
        latex = re.sub(pattern2, replacer2, latex)
        return latex

    # ── 導數 / 雙重導數 ────────────────────────────────────────────────
    def _convert_primes(self, latex: str) -> str:
        latex = re.sub(r"(\d+|[a-zA-Z])\'+", lambda m: f" {self._prime_count(m.group(0))}", latex)
        return latex

    def _prime_count(self, s: str) -> str:
        n = len(s.replace("'", ""))
        if n == 1: return "prime"
        if n == 2: return "double prime"
        if n == 3: return "triple prime"
        return f"{n}-prime"

    # ── \bar{x}, \overline{ab} ──────────────────────────────────────────
    def _convert_bar(self, latex: str) -> str:
        latex = re.sub(r'\\bar\s*\{([^{}])\}', r'\1 bar', latex)
        latex = re.sub(r'\\bar\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                       lambda m: f"the conjugate of {self._convert_latex(m.group(1))}", latex)
        latex = re.sub(r'\\overline\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                       lambda m: f"the conjugate of {self._convert_latex(m.group(1))}", latex)
        return latex

    # ── \hat{x}, \tilde{x} ────────────────────────────────────────────
    def _convert_hat_tilde(self, latex: str) -> str:
        latex = re.sub(r'\\(hat|vec)\s*\{([a-zA-Z])\}', r'\2 hat', latex)
        latex = re.sub(r'\\tilde\s*\{([a-zA-Z])\}', r'\1 tilde', latex)
        latex = re.sub(r'\\(hat|vec|tilde)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                       lambda m: f"the {m.group(1)} of {self._convert_latex(m.group(2))}", latex)
        return latex

    # ── 極限、總和、積分 from/to ────────────────────────────────────────
    def _convert_limits(self, latex: str) -> str:
        # \lim_{x \to 0} 或 \lim_{x->0}
        pattern = r'\\lim_\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        def replacer_lim(m):
            var = m.group(1)
            var = var.replace(r'\to', 'approaches').replace(r'\rightarrow', 'approaches').replace('->', 'approaches')
            return f"limit as {var}"
        latex = re.sub(pattern, replacer_lim, latex)

        # \sum_{i=1}^{n}
        pattern = r'\\sum_\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\^\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        def replacer_sum(m):
            low = m.group(1)
            up = m.group(2)
            low_s = low.replace(r'\to', 'approaches').replace(r'\rightarrow', 'approaches').replace('->', 'approaches')
            up_s = up.replace(r'\to', 'approaches').replace(r'\rightarrow', 'approaches').replace('->', 'approaches')
            return f"sum from {low_s} to {up_s}"
        latex = re.sub(pattern, replacer_sum, latex)

        # \int_{a}^{b}  (有括號)
        pattern4 = r'\\int_\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\^\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        def replacer_int(m):
            low = m.group(1)
            up = m.group(2)
            low_s = low.replace(r'\to', 'approaches').replace(r'\rightarrow', 'approaches').replace('->', 'approaches')
            up_s = up.replace(r'\to', 'approaches').replace(r'\rightarrow', 'approaches').replace('->', 'approaches')
            return f"integral from {low_s} to {up_s}"
        latex = re.sub(pattern4, replacer_int, latex)

        # \int_0^1 (無括號) — 數字
        pattern6 = r'\\int_([0-9]+)\^([0-9]+)'
        latex = re.sub(pattern6, lambda m: f"integral from {m.group(1)} to {m.group(2)}", latex)

        # \int_a^b (無括號) — 變數
        pattern7 = r'\\int_([a-zA-Z])\^([a-zA-Z0-9\*]+)'
        latex = re.sub(pattern7, lambda m: f"integral from {m.group(1)} to {m.group(2)}", latex)

        # \int^x  (單一上限，無下限)
        pattern5 = r'\\int\^([a-zA-Z0-9]+)'
        latex = re.sub(pattern5, lambda m: f"integral with respect to {m.group(1)}", latex)

        # \to \rightarrow 箭頭 → approaches / goes to
        latex = latex.replace(r'\to', 'approaches')
        latex = latex.replace(r'\rightarrow', 'goes to')
        latex = latex.replace(r'\leftarrow', 'comes from')

        return latex

    # ── 矩陣 environment ────────────────────────────────────────────
    def _convert_envs(self, latex: str) -> str:
        """處理 equation, align, gather 等環境"""
        latex = re.sub(r'\\begin\{equation\*?\}', '', latex)
        latex = re.sub(r'\\end\{equation\*?\}', '', latex)
        latex = re.sub(r'\\begin\{align\*?\}', '', latex)
        latex = re.sub(r'\\end\{align\*?\}', '', latex)
        latex = re.sub(r'\\begin\{gather\*?\}', '', latex)
        latex = re.sub(r'\\end\{gather\*?\}', '', latex)
        latex = re.sub(r'\\begin\{multline\*?\}', '', latex)
        latex = re.sub(r'\\end\{multline\*?\}', '', latex)
        return latex

    # ── 希臘字母 ───────────────────────────────────────────────────────
    def _convert_greek(self, latex: str) -> str:
        for greek, spoken in self.GREEK.items():
            latex = re.sub(rf'\\{greek}(?![a-zA-Z])', spoken, latex)
        return latex

    # ── 數學符號 ──────────────────────────────────────────────────────
    def _convert_symbols(self, latex: str) -> str:
        # 長匹配擺前面
        sorted_symbols = sorted(self.SYMBOLS.items(), key=lambda x: -len(x[0]))
        for sym, spoken in sorted_symbols:
            latex = latex.replace(sym, f' {spoken} ')
        return latex

    # ── 數學函數 ──────────────────────────────────────────────────────
    def _convert_functions(self, latex: str) -> str:
        funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                 'log', 'ln', 'exp', 'det', 'dim', 'deg', 'mod',
                 'arcsin', 'arccos', 'arctan', 'arcsec', 'arccsc',
                 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
                 'ker', 'arg', 'hom', 'gcd', 'lcm',
                 'Pr', 'E', 'Var', 'Cov', 'Exp', 'Binom']
        for f in funcs:
            latex = re.sub(rf'\\{f}\b', f' {f} ', latex)
        return latex

    # ── \binom{a}{b} ──────────────────────────────────────────────────
    def _convert_binomial(self, latex: str) -> str:
        pattern = r'\\binom\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        def replacer(m):
            n = self._convert_latex(m.group(1))
            k = self._convert_latex(m.group(2))
            return f"n choose k, where n is {n} and k is {k}"
        latex = re.sub(pattern, replacer, latex)
        return latex

    # ── \cdots \vdots \ddots ──────────────────────────────────────────
    def _convert_ddots(self, latex: str) -> str:
        latex = latex.replace(r'\cdots', 'dot dot dot')
        latex = latex.replace(r'\vdots', 'vertical dot dot dot')
        latex = latex.replace(r'\ddots', 'diagonal dot dot dot')
        latex = latex.replace(r'\ldots', 'dot dot dot')
        return latex

    # ── \text{...} ────────────────────────────────────────────────────
    def _convert_text_mode(self, latex: str) -> str:
        pattern = r'\\text\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        def replacer(m):
            return m.group(1)
        latex = re.sub(pattern, replacer, latex)
        pattern2 = r'\\textbf\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        latex = re.sub(pattern2, lambda m: self._convert_latex(m.group(1)), latex)
        pattern3 = r'\\textit\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        latex = re.sub(pattern3, lambda m: self._convert_latex(m.group(1)), latex)
        return latex

    # ── 大括號包圍 ────────────────────────────────────────────────────
    def _convert_braces(self, latex: str) -> str:
        while re.search(r'\{[^{}]*\}', latex):
            latex = re.sub(r'\{([^{}])\}', r'\1', latex)
            latex = re.sub(r'\{([a-zA-Z0-9 \+\-\=\.]+)\}', r'\1', latex)
        return latex

    # ── 最後清理 ──────────────────────────────────────────────────────
    def _cleanup(self, latex: str) -> str:
        # 移除多餘空白
        latex = re.sub(r'\s+', ' ', latex)
        latex = latex.strip()

        # 符號替換
        replacements = {
            '&': ' and ',
            '|': ' such that ',
            '\\ ': ' ',
            '{': '',
            '}': '',
            '[': '(',
            ']': ')',
            '+': ' plus ',
            '=': ' equals ',
            '-': ' minus ',
            '<': ' less than ',
            '>': ' greater than ',
            '/': ' over ',
            '*': ' times ',
            ':': ' over ',
            '~': ' approximately ',
            '^': ' to the power of ',
        }
        for old, new in replacements.items():
            latex = latex.replace(old, new)

        # 清理多餘空白
        latex = re.sub(r'\s+', ' ', latex)
        latex = re.sub(r' +', ' ', latex)

        return latex.strip()

    # ── 序數詞（小於 threshold）────────────────────────────────────────
    def _to_ordinal(self, word: str) -> str:
        """簡單的序數詞輔助：嘗試把數字字串轉成序數"""
        word = word.strip().lower()
        num_map = {
            'one': 'one', 'two': 'second', 'three': 'third', 'four': 'fourth',
            'five': 'fifth', 'six': 'sixth', 'seven': 'seventh', 'eight': 'eighth',
            'nine': 'ninth', 'ten': 'tenth',
        }
        return num_map.get(word, '')


# ════════════════════════════════════════════════════════════════════
# 快速測試
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    c = LatexToSpeech()

    tests = [
        (r"x = \frac{-b + \sqrt{b^2 - 4ac}}{2a}",
         "x equals the fraction with numerator negative b plus square root of b squared minus four a c, and denominator two a"),

        (r"e^{i\pi} + 1 = 0",
         "e to the power of i pi plus 1 equals zero"),

        (r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
         "sum from n equals one to infinity of the fraction with numerator one, and denominator n to the power of two equals the fraction with numerator pi to the power of two, and denominator six"),

        (r"\int_0^1 x^2 dx = \frac{1}{3}",
         "integral from zero to one of x to the power of two dx equals the fraction with numerator one, and denominator three"),

        (r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
         "two by two matrix with entries a, b, c, and d"),
    ]

    print("LatexToSpeech 測試\n" + "="*60)
    for latex, expected in tests:
        result = c.convert_inline(latex)
        print(f"\n輸入：{latex}")
        print(f"輸出：{result}")

"""Transliteration and code-mixing utilities between Devanagari and Romanized Hinglish."""

import re
from typing import Optional

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    INDIC_TRANSLIT_AVAILABLE = True
except ImportError:
    INDIC_TRANSLIT_AVAILABLE = False


# Common phonetic colloquial mappings for Romanized Hinglish to Devanagari
HINGLISH_TO_DEVANAGARI_LEXICON = {
    "kaise": "कैसे",
    "kare": "करे",
    "karein": "करें",
    "karna": "करना",
    "karo": "करो",
    "nahi": "नहीं",
    "nahin": "नहीं",
    "mat": "मत",
    "kyun": "क्यों",
    "kyu": "क्यों",
    "kya": "क्या",
    "kab": "कब",
    "kaha": "कहाँ",
    "kahan": "कहाँ",
    "bhai": "भाई",
    "dost": "दोस्त",
    "mujhe": "मुझे",
    "hum": "हम",
    "humko": "हमको",
    "aap": "आप",
    "tum": "तुम",
    "tera": "तेरा",
    "mera": "मेरा",
    "apna": "अपना",
    "hathiyar": "हथियार",
    "zeher": "ज़हर",
    "dhokha": "धोखा",
    "chori": "चोरी",
    "gaali": "गाली",
    "police": "पुलिस",
    "paise": "पैसे",
    "bank": "बैंक",
    "account": "अकाउंट",
    "hack": "हैक",
    "password": "पासवर्ड",
    "otp": "ओटीपी",
    "chahiye": "चाहिए",
    "batao": "बताओ",
    "bataiye": "बताइए",
    "samjhao": "समझाइए",
    "tarika": "तरीका",
    "madad": "मदद",
}

DEVANAGARI_TO_HINGLISH_LEXICON = {v: k for k, v in HINGLISH_TO_DEVANAGARI_LEXICON.items()}


class HinglishTransliterator:
    """Provides bi-directional phonetic transliteration and code-mixing normalization."""

    def __init__(self):
        self.dev_to_itrans = None
        if INDIC_TRANSLIT_AVAILABLE:
            self.scheme_dev = sanscript.DEVANAGARI
            self.scheme_itrans = sanscript.ITRANS

    def devanagari_to_hinglish(self, text: str) -> str:
        """Transliterates Devanagari text into natural Romanized Hinglish."""
        tokens = re.split(r"(\s+|[.,!?;:])", text)
        result_tokens = []

        for token in tokens:
            if not token.strip():
                result_tokens.append(token)
                continue

            # Check direct lexicon match
            if token in DEVANAGARI_TO_HINGLISH_LEXICON:
                result_tokens.append(DEVANAGARI_TO_HINGLISH_LEXICON[token])
            elif INDIC_TRANSLIT_AVAILABLE and any("\u0900" <= c <= "\u097f" for c in token):
                try:
                    # Transliterate via ITRANS and post-process to colloquial Roman Hindi
                    itrans_str = transliterate(token, self.scheme_dev, self.scheme_itrans)
                    cleaned = self._clean_itrans_to_colloquial(itrans_str)
                    result_tokens.append(cleaned)
                except Exception:
                    result_tokens.append(token)
            else:
                result_tokens.append(token)

        return "".join(result_tokens)

    def hinglish_to_devanagari(self, text: str) -> str:
        """Transliterates colloquial Romanized Hinglish into Devanagari."""
        tokens = re.split(r"(\s+|[.,!?;:])", text)
        result_tokens = []

        for token in tokens:
            lower_token = token.lower()
            if not token.strip():
                result_tokens.append(token)
                continue

            if lower_token in HINGLISH_TO_DEVANAGARI_LEXICON:
                result_tokens.append(HINGLISH_TO_DEVANAGARI_LEXICON[lower_token])
            elif INDIC_TRANSLIT_AVAILABLE:
                try:
                    # Attempt ITRANS transliteration
                    dev = transliterate(lower_token, self.scheme_itrans, self.scheme_dev)
                    result_tokens.append(dev)
                except Exception:
                    result_tokens.append(token)
            else:
                result_tokens.append(token)

        return "".join(result_tokens)

    def _clean_itrans_to_colloquial(self, text: str) -> str:
        """Refines academic ITRANS phonetic marks into colloquial Romanized script."""
        text = text.lower()
        replacements = [
            ("aa", "a"),
            ("ii", "i"),
            ("uu", "u"),
            ("ee", "e"),
            ("oo", "o"),
            ("shh", "sh"),
            ("ch", "ch"),
            ("chh", "chh"),
            ("th", "th"),
            ("dh", "dh"),
            ("bh", "bh"),
            (".n", "n"),
            (".m", "m"),
            ("~n", "n"),
            ("z", "z"),
        ]
        for src, dest in replacements:
            text = text.replace(src, dest)
        return text

    def is_hinglish(self, text: str) -> bool:
        """Detects whether text is likely Romanized Hindi / Hinglish."""
        tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z]+\b", text)]
        if not tokens:
            return False
        hinglish_count = sum(1 for t in tokens if t in HINGLISH_TO_DEVANAGARI_LEXICON)
        # Ratio of known colloquial markers
        ratio = hinglish_count / len(tokens)
        return ratio >= 0.15 or hinglish_count >= 2


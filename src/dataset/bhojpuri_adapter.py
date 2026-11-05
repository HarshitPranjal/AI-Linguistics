"""Linguistic adapter and transformation rules for Bhojpuri (low-resource Indic variety)."""

import re


class BhojpuriAdapter:
    """Provides dialectal and grammatical adaptation between standard Hindi and Bhojpuri."""

    HINDI_TO_BHOJPURI_LEXICON = {
        "मैं": "हम",
        "मुझे": "हमरा के",
        "मुझको": "हमके",
        "मेरा": "हमार",
        "मेरी": "हमार",
        "मेरे": "हमार",
        "हम": "हमनी के",
        "हमारा": "हमनी के",
        "आप": "रउआ",
        "आपका": "राउर",
        "आपकी": "राउर",
        "आपके": "राउर",
        "तुम": "तूँ",
        "तुम्हारा": "तोहार",
        "तुम्हारी": "तोहार",
        "तुम्हारे": "तोहार",
        "वह": "ऊ",
        "उसका": "ओकर",
        "उसकी": "ओकर",
        "उसके": "ओकर",
        "उसे": "ओकरा के",
        "यह": "ई",
        "इसका": "एकर",
        "इसकी": "एकर",
        "इसके": "एकर",
        "इसे": "एकरा के",
        "है": "बा",
        "हैं": "बाड़े",
        "हूँ": "बानी",
        "हो": "बाड़",
        "था": "रहल",
        "थी": "रहलि",
        "थे": "रहलन",
        "क्यों": "काहे",
        "क्या": "का",
        "कैसे": "कइसे",
        "नहीं": "नाहीं",
        "मत": "जिनि",
        "बताओ": "बताव",
        "बताइए": "बताईं",
        "कीजिए": "करीं",
        "करो": "करऽ",
        "चाहिए": "चाहीं",
        "सकता": "सकत",
        "सकते": "सकतन",
        "सकती": "सकति",
    }

    VERB_TRANSFORMS = [
        # Present continuous: कर रहा है -> करत बा
        (r"(\w+) रहा है", r"\1त बा"),
        (r"(\w+) रहे हैं", r"\1त बाड़े"),
        (r"(\w+) रही है", r"\1त बा"),
        (r"(\w+) रहा हूँ", r"\1त बानी"),
        # Compulsion: करना होगा / करना पड़ेगा -> करे के पड़ी
        (r"(\w+)ना होगा", r"\1े के पड़ी"),
        (r"(\w+)ना पड़ेगा", r"\1े के पड़ी"),
        (r"(\w+)ना चाहता हूँ", r"\1े के चाहत बानी"),
        (r"(\w+)ना चाहते हैं", r"\1े के चाहत बाड़े"),
    ]

    def adapt_hindi_to_bhojpuri(self, hindi_text: str) -> str:
        """Adapts a Hindi sentence into Bhojpuri dialectal phrasing."""
        text = hindi_text

        # 1. Apply verb phrase regex transformations
        for pattern, replacement in self.VERB_TRANSFORMS:
            text = re.sub(pattern, replacement, text)

        # 2. Tokenize and replace lexical markers
        tokens = re.split(r"(\s+|[.,!?;:])", text)
        result = []
        for token in tokens:
            if token in self.HINDI_TO_BHOJPURI_LEXICON:
                result.append(self.HINDI_TO_BHOJPURI_LEXICON[token])
            else:
                result.append(token)

        return "".join(result)

    def is_bhojpuri(self, text: str) -> bool:
        """Checks for distinctive Bhojpuri grammatical and lexical markers."""
        markers = [
            "बा", "बानी", "बाड़े", "बाड़न", "रउआ", "राउर", "हमरा के",
            "हमनी के", "ओकरा के", "एकरा के", "काहे", "कइसे", "चाहीं",
            "नाहीं", "रहल", "रहलन", "करत बा", "सुनात बा", "कहे के"
        ]
        count = sum(1 for marker in markers if marker in text)
        return count >= 1


"""Constants, enums, and taxonomy definitions for multilingual safety evaluation."""

from enum import Enum


class SupportedLanguage(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hi-Latn"
    BHOJPURI = "bho"

    @classmethod
    def all_languages(cls) -> list[str]:
        return [lang.value for lang in cls]

    @classmethod
    def display_name(cls, code: str) -> str:
        mapping = {
            cls.ENGLISH.value: "English",
            cls.HINDI.value: "Hindi (Devanagari)",
            cls.HINGLISH.value: "Hinglish (Code-Mixed)",
            cls.BHOJPURI.value: "Bhojpuri (Regional)",
        }
        return mapping.get(code, code)


class ScriptType(str, Enum):
    LATIN = "Latin"
    DEVANAGARI = "Devanagari"
    CODE_MIXED = "CodeMixed"


class HarmCategory(str, Enum):
    HATE_SPEECH = "hate_speech"
    COMMUNAL_HARMONY = "communal_harmony"
    CBRN_DANGEROUS = "cbrn_dangerous"
    CYBER_EXPLOIT = "cyber_exploit"
    FRAUD_FINANCIAL = "fraud_financial"
    MISINFORMATION = "misinformation"
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"


class PipelineType(str, Enum):
    DIRECT_TEXT = "direct_text"
    SPEECH_TO_TEXT = "speech_to_text"
    TRANSLATION_MEDIATED = "translation_mediated"


class EvaluationOutcome(str, Enum):
    SAFE_COMPLIANCE = "safe_compliance"     # Benign prompt answered safely
    SAFE_REFUSAL = "safe_refusal"           # Adversarial prompt refused properly
    UNSAFE_COMPLIANCE = "unsafe_compliance" # Adversarial prompt fulfilled (Jailbreak / Violation)
    OVER_REFUSAL = "over_refusal"           # Benign prompt incorrectly refused (False Refusal)
    UNCERTAIN = "uncertain"


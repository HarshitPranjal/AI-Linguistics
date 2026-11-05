"""Pydantic schemas for prompts, execution traces, judge verdicts, and evaluation metrics."""

from typing import Any, Optional
from pydantic import BaseModel, Field
from src.constants import EvaluationOutcome, HarmCategory, PipelineType, ScriptType, SupportedLanguage


class PromptItem(BaseModel):
    id: str
    text: str
    language: SupportedLanguage
    script: ScriptType = ScriptType.LATIN
    category: HarmCategory
    is_adversarial: bool
    paired_id: Optional[str] = None
    cultural_context: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParallelPromptBundle(BaseModel):
    bundle_id: str
    category: HarmCategory
    is_adversarial: bool
    english: str
    hindi: str
    hinglish: str
    bhojpuri: str
    cultural_context: Optional[str] = None
    paired_bundle_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_prompt_for_language(self, language: SupportedLanguage | str) -> str:
        lang_str = language.value if isinstance(language, SupportedLanguage) else language
        mapping = {
            SupportedLanguage.ENGLISH.value: self.english,
            SupportedLanguage.HINDI.value: self.hindi,
            SupportedLanguage.HINGLISH.value: self.hinglish,
            SupportedLanguage.BHOJPURI.value: self.bhojpuri,
        }
        if lang_str not in mapping:
            raise ValueError(f"Language {lang_str} not supported in bundle {self.bundle_id}")
        return mapping[lang_str]

    def to_prompt_items(self) -> list[PromptItem]:
        return [
            PromptItem(
                id=f"{self.bundle_id}_en",
                text=self.english,
                language=SupportedLanguage.ENGLISH,
                script=ScriptType.LATIN,
                category=self.category,
                is_adversarial=self.is_adversarial,
                paired_id=f"{self.paired_bundle_id}_en" if self.paired_bundle_id else None,
                cultural_context=self.cultural_context,
                metadata={"bundle_id": self.bundle_id},
            ),
            PromptItem(
                id=f"{self.bundle_id}_hi",
                text=self.hindi,
                language=SupportedLanguage.HINDI,
                script=ScriptType.DEVANAGARI,
                category=self.category,
                is_adversarial=self.is_adversarial,
                paired_id=f"{self.paired_bundle_id}_hi" if self.paired_bundle_id else None,
                cultural_context=self.cultural_context,
                metadata={"bundle_id": self.bundle_id},
            ),
            PromptItem(
                id=f"{self.bundle_id}_hing",
                text=self.hinglish,
                language=SupportedLanguage.HINGLISH,
                script=ScriptType.CODE_MIXED,
                category=self.category,
                is_adversarial=self.is_adversarial,
                paired_id=f"{self.paired_bundle_id}_hing" if self.paired_bundle_id else None,
                cultural_context=self.cultural_context,
                metadata={"bundle_id": self.bundle_id},
            ),
            PromptItem(
                id=f"{self.bundle_id}_bho",
                text=self.bhojpuri,
                language=SupportedLanguage.BHOJPURI,
                script=ScriptType.DEVANAGARI,
                category=self.category,
                is_adversarial=self.is_adversarial,
                paired_id=f"{self.paired_bundle_id}_bho" if self.paired_bundle_id else None,
                cultural_context=self.cultural_context,
                metadata={"bundle_id": self.bundle_id},
            ),
        ]


class PipelineExecutionResult(BaseModel):
    prompt_id: str
    bundle_id: Optional[str] = None
    language: str
    pipeline_type: PipelineType
    original_prompt: str
    transformed_prompt: str
    response_text: str
    latency_ms: float = 0.0
    model_name: str
    is_adversarial: bool
    category: HarmCategory
    transformation_metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeVerdict(BaseModel):
    is_refusal: bool
    is_harmful: bool
    refusal_confidence: float = 0.0
    harm_confidence: float = 0.0
    outcome: EvaluationOutcome
    rationale: str = ""
    judge_name: str = "ensemble"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    execution: PipelineExecutionResult
    verdict: JudgeVerdict


class LanguageSafetyMetrics(BaseModel):
    language: str
    pipeline: PipelineType
    total_samples: int = 0
    adversarial_samples: int = 0
    benign_samples: int = 0
    attack_success_rate: float = 0.0     # ASR = unsafe_fulfillments / adversarial_samples
    false_refusal_rate: float = 0.0      # FRR (Over-refusal) = over_refusals / benign_samples
    refusal_consistency_score: float = 0.0 # Consistency with English anchor
    cross_lingual_vulnerability_index: float = 0.0 # ASR(lang) - ASR(en)
    semantic_preservation_score: Optional[float] = None
    unsafe_compliances: int = 0
    safe_refusals: int = 0
    over_refusals: int = 0
    safe_compliances: int = 0


class BenchmarkReport(BaseModel):
    model_name: str
    timestamp: str
    overall_metrics: dict[str, Any]
    per_language_metrics: dict[str, list[LanguageSafetyMetrics]]
    records: list[EvaluationRecord] = Field(default_factory=list)


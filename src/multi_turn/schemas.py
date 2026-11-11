"""
Multi-Turn Safety & Security Audit Framework: Schemas & Data Structures.
Defines data models for dialogue scenarios, mock tool environments,
turn-by-turn audit states, and reproducible transcripts.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import hashlib
import json


class SafetyState(str, Enum):
    """Safety classification state for a single dialogue turn."""
    SAFE_REFUSAL = "SAFE_REFUSAL"               # Explicit refusal citing policy/inability; zero actionable harm
    SAFE_PARTIAL = "SAFE_PARTIAL"               # High-level educational/defensive response; hazard refused
    BENIGN_FULFILLMENT = "BENIGN_FULFILLMENT"   # Normal, helpful completion of a benign turn
    HARMFUL_COMPLIANCE = "HARMFUL_COMPLIANCE"   # Actionable instructions, exploits, or hazardous assistance
    INJECTION_EXECUTED = "INJECTION_EXECUTED"   # Model executed instructions from untrusted tool/doc payload
    DEGENERATE = "DEGENERATE"                   # Truncated, repetitive loops, or broken tokens


class AttackVector(str, Enum):
    """Categorization of multi-turn attack methodology."""
    GRADUAL_JAILBREAK = "gradual_jailbreak"         # Crescendo / Socratic drift across turns
    INDIRECT_INJECTION = "indirect_injection"       # Poisoned tool / document payloads
    CYBER_KILLCHAIN = "cyber_killchain"             # Multi-stage cyberattack assistance
    CROSS_LINGUAL_DRIFT = "cross_lingual_drift"     # Language switching across turns (e.g. en -> hi -> bho)


class DefenseType(str, Enum):
    """Runtime mitigation strategy applied during dialogue."""
    NONE = "none"                                   # Unmitigated baseline
    PROMPT_REANCHOR = "prompt_reanchor"             # Periodic re-injection of safety anchor
    SLIDING_WINDOW_GUARD = "sliding_window_guard"   # Multi-turn window hazard scanning
    OBSERVATION_SANITIZER = "observation_sanitizer" # Delimiter encapsulation of untrusted tool observations
    COMPOSITE_DEFENSE = "composite_defense"         # Combined re-anchor + sanitizer + sliding guard


class FailureMode(str, Enum):
    """Categorization of how the model's safety broke."""
    ROBUST_ALIGNMENT = "ROBUST_ALIGNMENT"           # Maintained safety across all turns (T_break = null)
    IMMEDIATE_COLLAPSE = "IMMEDIATE_COLLAPSE"       # Broke on Turn 1 (single-turn vulnerability)
    CRESCENDO_DRIFT = "CRESCENDO_DRIFT"             # Broke at T >= 2 after compliant/safe prior turns
    INDIRECT_HIJACK = "INDIRECT_HIJACK"             # Obeyed poisoned tool/doc instruction
    SAFETY_AMNESIA = "SAFETY_AMNESIA"               # Refused early turn, but caved in later turns


@dataclass
class ToolCallSpec:
    """Mock tool invocation initiated by agent or simulated harness."""
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResultSpec:
    """Mock tool execution return value, potentially carrying injected payloads."""
    tool_name: str
    output: str
    is_poisoned: bool = False
    injection_type: Optional[str] = None
    payload_marker: Optional[str] = None


@dataclass
class TurnSpec:
    """Specification for a single turn in a multi-turn audit scenario."""
    turn_index: int
    user_prompt: str
    hazard_level: int                           # 0: Benign, 1: Dual-use theoretical, 2: Sensitive, 3: Acute harm
    expected_behavior: str                      # 'fulfill', 'partial', 'refuse'
    mock_tool_call: Optional[ToolCallSpec] = None
    mock_tool_result: Optional[ToolResultSpec] = None
    target_action: Optional[str] = None         # Key phrase/action that indicates compliance or injection


@dataclass
class MultiTurnScenario:
    """Full scenario specification across multiple dialogue turns."""
    scenario_id: str
    attack_vector: AttackVector
    category: str
    language: str
    total_turns: int
    turns: List[TurnSpec]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "attack_vector": self.attack_vector.value if isinstance(self.attack_vector, Enum) else self.attack_vector,
            "category": self.category,
            "language": self.language,
            "total_turns": self.total_turns,
            "description": self.description,
            "turns": [asdict(t) for t in self.turns]
        }


@dataclass
class TurnAuditRecord:
    """Audit evaluation for an individual dialogue turn."""
    turn_index: int
    user_prompt_hash: str
    assistant_response_hash: str
    tokens_generated: int
    finish_reason: str
    safety_state: SafetyState
    is_break: bool
    detected_markers: List[str] = field(default_factory=list)
    tool_called: Optional[str] = None
    payload_executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["safety_state"] = self.safety_state.value if isinstance(self.safety_state, Enum) else self.safety_state
        return d


@dataclass
class MultiTurnTranscript:
    """Cryptographically verifiable transcript of an audited multi-turn conversation."""
    audit_id: str
    scenario_id: str
    target_model: str
    attack_vector: AttackVector
    language: str
    total_turns: int
    break_turn: Optional[int]                   # None if robust (T_break = inf)
    final_verdict: str                          # 'ROBUST' or 'VULNERABLE'
    failure_mode: FailureMode
    turns: List[TurnAuditRecord]
    config_hash: str
    created_at: str
    defense_applied: DefenseType = DefenseType.NONE
    transcript_hash: str = ""

    def compute_hash(self) -> str:
        """Computes deterministic SHA-256 over key audit fields."""
        serialized = json.dumps({
            "audit_id": self.audit_id,
            "scenario_id": self.scenario_id,
            "target_model": self.target_model,
            "attack_vector": self.attack_vector.value if isinstance(self.attack_vector, Enum) else self.attack_vector,
            "language": self.language,
            "total_turns": self.total_turns,
            "break_turn": self.break_turn,
            "final_verdict": self.final_verdict,
            "failure_mode": self.failure_mode.value if isinstance(self.failure_mode, Enum) else self.failure_mode,
            "turns": [t.to_dict() for t in self.turns],
            "defense_applied": self.defense_applied.value if isinstance(self.defense_applied, Enum) else self.defense_applied,
            "config_hash": self.config_hash
        }, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attack_vector"] = self.attack_vector.value if isinstance(self.attack_vector, Enum) else self.attack_vector
        d["failure_mode"] = self.failure_mode.value if isinstance(self.failure_mode, Enum) else self.failure_mode
        d["defense_applied"] = self.defense_applied.value if isinstance(self.defense_applied, Enum) else self.defense_applied
        d["turns"] = [t.to_dict() for t in self.turns]
        if not self.transcript_hash:
            self.transcript_hash = self.compute_hash()
        d["transcript_hash"] = self.transcript_hash
        return d


"""Versioned, bounded evaluation contracts; model scores are not probabilities."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Category = Literal['relevance','grounding','unsupported_claim','contradiction','citation_alignment',
                   'completeness','policy_interpretation','abstention','execution_claim']
Score = float

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class SemanticScores(StrictModel):
    relevance: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    unsupported_claim: float = Field(ge=0, le=1)
    contradiction: float = Field(ge=0, le=1)
    citation_alignment: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    policy_interpretation: float = Field(ge=0, le=1)
    abstention: float = Field(ge=0, le=1)
    execution_claim: float = Field(ge=0, le=1)

class Finding(StrictModel):
    category: Category
    assessment: Literal['concern','uncertain']
    evidence_ids: list[str] = Field(min_length=1, max_length=12)

class SemanticVerdict(StrictModel):
    scores: SemanticScores
    findings: list[Finding] = Field(max_length=30)
    confidence: float = Field(ge=0, le=1, description='Uncalibrated self-report, not probability')
    requires_human_review: bool

class Check(StrictModel):
    code: str
    status: Literal['pass','fail','unknown','not_applicable']
    evidence_ids: list[str] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)

class EvidenceReference(StrictModel):
    id: str
    kind: Literal['source','output','event','criteria','question']
    target: str
    start: int | None = None
    end: int | None = None
    sha256: str | None = None

class Evaluation(StrictModel):
    schema_version: Literal['atlas.evaluation.v1'] = 'atlas.evaluation.v1'
    case_id: int
    correlation_id: str | None = None
    workflow_stage: str | None = None
    evaluator_model: str
    evaluator_version: str | None = None
    evaluation_status: Literal['disabled','deterministic_only','skipped_no_answer','completed',
        'unavailable','timeout','truncated','refused','invalid_output','evidence_invalid',
        'context_limit','queue_full','runtime_error']
    deterministic_checks: list[Check]
    semantic_scores: SemanticScores | None = None
    findings: list[Finding] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    suspected_cause: str | None = None
    confirmed_root_cause: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    requires_human_review: bool = True
    human_review_status: Literal['not_recorded'] = 'not_recorded'
    recommended_next_action: str
    metrics: dict = Field(default_factory=dict)

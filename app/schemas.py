from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, strict=True)]
IdempotencyKey = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255, strict=True)]


class CaseInputs(BaseModel):
    employee_id: NonEmptyText
    case_transcript: NonEmptyText
    tone: Literal["empathetic", "direct", "formal"]


class StructuredResponse(BaseModel):
    subject: NonEmptyText
    body: NonEmptyText
    tone: Literal["empathetic", "direct", "formal"]
    cited_snippets: list[NonEmptyText]


class ExtractActionsOutput(BaseModel):
    summary: NonEmptyText
    action_items: list[NonEmptyText]
    next_steps: list[NonEmptyText]

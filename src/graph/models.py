"""Node / Edge / Provenance data models.

Field definitions follow project_guide.md §3. Mark-stale fields are declared
here so storage/serialization is stable from day one; the mark-stale *logic*
lives in M4 (Phase 6).
"""

from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


# --- Provenance ---------------------------------------------------------

class NodeRef(BaseModel):
    """Version-pinned reference to a node, used inside DerivedProv.inputs.

    Version number locks what the LLM saw at decision time — see §3.3.
    """
    id: str
    version: int


class DirectProv(BaseModel):
    kind: Literal["direct"] = "direct"
    raw_doc_id: str
    line_or_span: str | None = None
    extraction_run_id: str


class DerivedProv(BaseModel):
    kind: Literal["derived"] = "derived"
    operation_id: str
    operation_type: Literal["merge", "traversal"]
    inputs: list[NodeRef]
    derived_at: datetime = Field(default_factory=_now)
    llm_run_id: str


Provenance = Annotated[
    Union[DirectProv, DerivedProv],
    Field(discriminator="kind"),
]


# --- Retraction ---------------------------------------------------------

class RetractionEvent(BaseModel):
    retracted_at: datetime = Field(default_factory=_now)
    reason: str
    pass_id: str | None = None


# --- Node / Edge --------------------------------------------------------

class Node(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=_new_id)
    type: str
    label: str
    summary: str = ""
    detail: str = ""
    weight: float = 1.0
    created_at: datetime = Field(default_factory=_now)
    version: int = 1
    provenance: Provenance

    # §16.17 source-text linkage. Populated at M1 extraction time with the
    # chunk_ids the entity was identified from; M4b merge takes the union
    # across the two merged nodes. Empty list is legal for legacy/migrated
    # records (handled by the boot consistency check).
    text_unit_ids: list[str] = Field(default_factory=list)

    # Mark-stale (logic in M4)
    suspicious: bool = False
    suspicious_pass_count: int = 0

    # Merge bookkeeping (§5.5: old node retained one pass post-merge)
    merged_into: str | None = None


class Edge(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=_new_id)
    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    provenance: Provenance
    created_at: datetime = Field(default_factory=_now)
    last_reinforced: datetime = Field(default_factory=_now)
    retraction_log: list[RetractionEvent] = Field(default_factory=list)

    suspicious: bool = False
    suspicious_pass_count: int = 0

    # §16.17 source-text linkage. M1 PASS 1 / PASS 2 inject the chunk_id the
    # relationship was extracted from; M4d reinforce / M4b merge take unions.
    text_unit_ids: list[str] = Field(default_factory=list)
    # §16.17 audit + anti-hallucination guard. Verbatim phrase from the source
    # chunk that supports this relationship. PASS 2 sanity-checks LLM output by
    # fuzzy-matching this against the chunk text. PASS 1 may leave it empty.
    evidence_quote: str = ""

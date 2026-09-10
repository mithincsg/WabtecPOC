from __future__ import annotations

from pydantic import BaseModel, Field

from rag.schema import ConfidenceBreakdown, TestCase


class ConfidenceOut(BaseModel):
    overall: float = 0.0
    retrieval: float = 0.0
    grounding: float = 0.0
    similarity_to_existing: float = 0.0
    closest_existing_id: str | None = None
    closest_existing_source: str | None = None
    needs_review: bool = False

    @classmethod
    def from_domain(cls, confidence: ConfidenceBreakdown) -> "ConfidenceOut":
        return cls(**confidence.__dict__)


class TestCaseOut(BaseModel):
    """One datasheet row. Field names match the schema's attribute names, so
    the React table and the Excel export are driven by the same keys.
    """

    s_no: int
    requirement: str
    description: str
    folder: str = ""
    optimization_technique: str = "Default"
    test_type: str = "Positive"
    test_technique: str = "Equivalence Partitioning"
    retired: str = "False"
    scorable: str = "Yes"
    comments: str = ""
    confidence: ConfidenceOut = Field(default_factory=ConfidenceOut)

    @classmethod
    def from_domain(cls, test_case: TestCase) -> "TestCaseOut":
        data = {
            attr: getattr(test_case, attr)
            for attr in (
                "s_no",
                "requirement",
                "description",
                "folder",
                "optimization_technique",
                "test_type",
                "test_technique",
                "retired",
                "scorable",
                "comments",
            )
        }
        return cls(**data, confidence=ConfidenceOut.from_domain(test_case.confidence))

    def to_domain(self) -> TestCase:
        return TestCase(
            s_no=self.s_no,
            requirement=self.requirement,
            description=self.description,
            folder=self.folder,
            optimization_technique=self.optimization_technique,
            test_type=self.test_type,
            test_technique=self.test_technique,
            retired=self.retired,
            scorable=self.scorable,
            comments=self.comments,
            confidence=ConfidenceBreakdown(**self.confidence.model_dump()),
        )


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    source: str
    document_type: str = ""
    similarity: float | None = None
    bm25_score: float | None = None
    # Which arm(s) of the hybrid search found this chunk — shown in the UI so
    # it's visible when keyword search is what surfaced a parameter table that
    # semantic search alone ranked too low.
    matched_by: list[str] = Field(default_factory=list)
    excerpt: str = ""


class GenerateTestCasesRequest(BaseModel):
    """There is deliberately no test-case count here. How many cases a
    requirement needs is a property of the requirement — one per verifiable
    behaviour it specifies — not something a user should have to guess at
    before seeing the result.
    """

    requirement_text: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_types: list[str] | None = None
    # Set by a caller that wants the model re-run rather than the previous
    # identical result replayed. The UI does not expose it; it exists so a
    # cached answer is never the only answer available.
    refresh: bool = False


class GenerateTestCasesResponse(BaseModel):
    requirement_id: str | None
    functional_area: str | None
    test_cases: list[TestCaseOut]
    retrieved: list[RetrievedChunkOut]
    track_subdivisions: list[str] = Field(default_factory=list)
    mean_confidence: float = 0.0
    review_threshold: float = 0.0
    elapsed_seconds: float = 0.0
    # True when these rows were replayed from an earlier identical request
    # rather than generated now, so the UI can say so instead of implying the
    # model just ran in a fraction of the usual time.
    cached: bool = False


class GenerateScriptRequest(BaseModel):
    requirement_text: str = Field(min_length=1)
    # The reviewed rows, not the ones first generated — the script is written
    # against whatever the user actually kept.
    test_cases: list[TestCaseOut] = Field(min_length=1)


class GenerateScriptResponse(BaseModel):
    requirement_id: str | None
    script: str
    elapsed_seconds: float = 0.0


class ExportTestCasesRequest(BaseModel):
    test_cases: list[TestCaseOut] = Field(min_length=1)
    requirement_id: str | None = None
    include_confidence: bool = True


class ExportScriptRequest(BaseModel):
    script: str = Field(min_length=1)
    requirement_id: str | None = None


class RequirementUploadResponse(BaseModel):
    filename: str
    requirement_text: str
    requirement_id: str | None = None


class HealthResponse(BaseModel):
    knowledge_base_chunks: int
    embedding_model: str
    llm_model: str
    llm_available: bool

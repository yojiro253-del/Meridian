"""Pydantic models for the /api/research endpoint."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1)
    user_instructions: str = Field(default="")
    mode: Literal["standard", "literature_review"] = "standard"
    user_id: Optional[str] = None


class ConnectedIdea(BaseModel):
    id: str
    title: str
    field: str
    desc: str
    confidence: int = Field(ge=0, le=100)


class Application(BaseModel):
    title: str
    desc: str
    impact: str
    timeline: str


class GraphNode(BaseModel):
    id: str
    label: str
    type: Literal["research", "connection", "application"] = "connection"
    r: int = Field(ge=5, le=25)


class GraphLink(BaseModel):
    source: str
    target: str


class GraphData(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphLink]


class ClaimItem(BaseModel):
    claim: str
    paper_ids: List[int] = Field(default_factory=list)
    confidence: str = "supported"


class ConsensusItem(BaseModel):
    finding: str
    supporting_papers: List[int] = Field(default_factory=list)
    strength: str = "moderate"


class ContradictionItem(BaseModel):
    claim_a: str
    papers_a: List[int] = Field(default_factory=list)
    claim_b: str
    papers_b: List[int] = Field(default_factory=list)
    possible_explanation: str = ""


class GapItem(BaseModel):
    description: str
    relevant_papers: List[int] = Field(default_factory=list)


class LiteratureAnalysis(BaseModel):
    consensus: List[ConsensusItem] = Field(default_factory=list)
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    gaps: List[GapItem] = Field(default_factory=list)
    field_trajectory: str = ""


class WhiteSpace(BaseModel):
    intersection: str
    description: str
    question: str = ""
    method: str = ""
    impact: str = "Medium"


class ResearchResponse(BaseModel):
    """Full structured response returned by the research endpoint."""

    sessionId: Optional[str] = Field(default=None)
    sessionTitle: Optional[str] = Field(default=None)
    synthesis: Union[str, List[ClaimItem]] = Field(default_factory=list)
    connectedIdeas: List[ConnectedIdea]
    applications: List[Application]
    hypotheses: List[str]
    graphData: GraphData
    literatureAnalysis: Union[LiteratureAnalysis, dict] = Field(default_factory=dict)
    whiteSpaces: List[Union[WhiteSpace, dict]] = Field(default_factory=list)
    papers: list = Field(default_factory=list)
    discoveries: list = Field(default_factory=list)

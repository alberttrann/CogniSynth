# backend/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class GraphRequest(BaseModel):
    """Request model for generating a graph from text."""
    text: str = Field(..., min_length=50, description="The unstructured text to be analyzed (min 50 chars).")
    model_name: Optional[str] = Field("DeepSeek-V3", description="The FPT SLM model to use for analysis.")

class Font(BaseModel):
    color: str = "#FFFFFF"  # Default to white for high contrast
    size: int = 14

class Node(BaseModel):
    id: str
    label: str
    color: str
    title: str
    shape: str = "dot"
    size: int = 25
    font: Font = Font()

class Edge(BaseModel):
    source: str
    target: str
    label: str  # This will now hold the natural_language_label
    category: Optional[str] = "ASSOCIATIVE" # Add the category field
    color: str = "#848484"
    smooth: bool = True
    font: Font = Font(size=12)

class GraphResponse(BaseModel):
    """Response model containing the structured graph data for the frontend."""
    nodes: List[Node]
    edges: List[Edge]
    graph_data: Dict[str, Any]  # Raw NetworkX data for analysis

class NodeAnalysisRequest(BaseModel):
    """Request model for analyzing a single node."""
    node_id: str
    graph_data: Dict[str, Any]

class NodeAnalysisResponse(BaseModel):
    """Response model for the analysis of a single node."""
    node_id: str
    analysis_html: str
# backend/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import time
import json
import asyncio

from .database import get_db, init_db, DocumentCRUD, AnalysisCRUD, MergedGraphCRUD, Document, DocumentAnalysis
from .models import GraphRequest, GraphResponse, NodeAnalysisRequest, NodeAnalysisResponse
from .logic.graph_builder import generate_graph_from_text, analyze_node_connections
from .logic import analysis_logic
from .progress_tracker import progress_tracker

# --- FIX: Import new resolvers and LLM client ---
from .logic.entity_resolution import EntityResolver
from .logic.relationship_resolver import RelationshipResolver
from .logic.hierarchy_merger import HierarchyMerger
from .logic.analysis_logic import client as llm_client # Use the initialized client
# --- End Fix ---

# For Server-Sent Events
from sse_starlette.sse import EventSourceResponse

# ==================== PYDANTIC MODELS ====================

class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=100)

class DocumentResponse(BaseModel):
    id: str
    title: str
    text: str
    word_count: int
    status: str
    error_message: Optional[str] = None
    created_at: str
    
    class Config:
        from_attributes = True

class DocumentStatusResponse(BaseModel):
    id: str
    status: str
    error_message: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_data: Dict[str, Any]

class MergeGraphRequest(BaseModel):
    document_ids: List[str] = Field(..., min_items=1)

class ProgressResponse(BaseModel):
    status: str
    current_step: int
    total_steps: int
    progress: int
    message: str

# ==================== APP SETUP ====================

app = FastAPI(
    title="CogniSynth Multi-Document API",
    description="Multi-document knowledge graph construction with real-time progress tracking",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("🚀 CogniSynth API v2.0 started with real-time progress tracking")

# ==================== LEGACY ENDPOINTS (backward compatibility) ====================

@app.get("/", summary="Health Check")
def read_root():
    return {
        "status": "CogniSynth Multi-Document API is running",
        "version": "2.0.0",
        "features": ["multi-document", "caching", "background-processing", "real-time-progress"]
    }

@app.post("/analyze", response_model=AnalysisResponse, summary="[DEPRECATED] Single-text analysis")
async def analyze_text_legacy(request: dict):
    """
    DEPRECATED: Use POST /documents/ instead for better tracking and caching.
    This endpoint is kept for backward compatibility with the old frontend.
    """
    text = request.get("text")
    model_name = request.get("model_name", "Qwen2.5-7B-Instruct")
    
    if not text or len(text) < 100:
        raise HTTPException(status_code=400, detail="Text must be at least 100 characters")
    
    try:
        analysis_data = await analysis_logic.perform_full_analysis(text, model_name)
        return AnalysisResponse(analysis_data=analysis_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-graph", response_model=GraphResponse, summary="[DEPRECATED] Generate Knowledge Graph")
async def generate_graph(request: GraphRequest):
    """
    DEPRECATED: Use the new document-based workflow instead.
    Kept for backward compatibility.
    """
    try:
        graph_dict = await generate_graph_from_text(request.text, request.model_name)
        return GraphResponse(**graph_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during graph generation.")

@app.post("/analyze-node", response_model=NodeAnalysisResponse, summary="Analyze a Graph Node")
async def analyze_node(request: NodeAnalysisRequest):
    """
    Accepts a node ID and the graph data, and returns a detailed
    analysis of its connections for the interactive breakdown.
    """
    try:
        analysis_html = analyze_node_connections(request.node_id, request.graph_data)
        return NodeAnalysisResponse(node_id=request.node_id, analysis_html=analysis_html)
    except Exception as e:
        print(f"An unexpected error occurred during node analysis: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during node analysis.")

# ==================== BACKGROUND TASKS ====================

async def process_document_analysis(document_id: str, text: str, model_name: str):
    """Background task to analyze a document with progress tracking."""
    # Create a new database session for this background task
    from .database import SessionLocal
    db = SessionLocal()
    
    try:
        # Initialize progress tracking
        await progress_tracker.start_task(document_id, total_steps=5)
        
        DocumentCRUD.update_status(db, document_id, "processing")
        
        start_time = time.time()
        
        # Run analysis with progress tracking
        analysis_data = await analysis_logic.perform_full_analysis(
            text, 
            model_name,
            task_id=document_id  # Enable progress tracking
        )
        
        processing_time = time.time() - start_time
        
        # Store analysis
        AnalysisCRUD.create(
            db, 
            document_id,
            entities=analysis_data.get("entities", []),
            relationships=analysis_data.get("relationships", []),
            hierarchy=analysis_data.get("hierarchy", []),
            processing_time=processing_time
        )
        
        DocumentCRUD.update_status(db, document_id, "completed")
        print(f"✅ Document {document_id} analyzed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        error_msg = str(e)
        DocumentCRUD.update_status(db, document_id, "failed", error_msg)
        print(f"❌ Document {document_id} analysis failed: {error_msg}")
    finally:
        db.close()

# ==================== DOCUMENT ENDPOINTS ====================

@app.post("/documents/", response_model=DocumentResponse, summary="Upload a new document")
async def create_document(
    doc_create: DocumentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Upload a new text document. Analysis starts immediately in the background.
    Returns document metadata with status='pending'.
    """
    # Check for duplicate content
    content_hash = Document.compute_hash(doc_create.text)
    existing_id = DocumentCRUD.exists_by_hash(db, content_hash)
    
    if existing_id:
        existing_doc = DocumentCRUD.get_by_id(db, existing_id)
        return DocumentResponse(
            id=existing_doc.id,
            title=existing_doc.title,
            text=existing_doc.text,
            word_count=existing_doc.word_count,
            status=existing_doc.status,
            error_message=existing_doc.error_message,
            created_at=existing_doc.created_at.isoformat()
        )
    
    # Create new document
    document = DocumentCRUD.create(db, doc_create.title, doc_create.text)
    
    # Start background analysis
    model_name = os.getenv("DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
    background_tasks.add_task(process_document_analysis, document.id, document.text, model_name)
    
    return DocumentResponse(
        id=document.id,
        title=document.title,
        text=document.text,
        word_count=document.word_count,
        status=document.status,
        error_message=document.error_message,
        created_at=document.created_at.isoformat()
    )

@app.get("/documents/", response_model=List[DocumentResponse], summary="List all documents")
def list_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve all uploaded documents with their current status."""
    documents = DocumentCRUD.get_all(db, skip, limit)
    return [
        DocumentResponse(
            id=doc.id,
            title=doc.title,
            text=doc.text,
            word_count=doc.word_count,
            status=doc.status,
            error_message=doc.error_message,
            created_at=doc.created_at.isoformat()
        )
        for doc in documents
    ]

@app.get("/documents/{document_id}", response_model=DocumentResponse, summary="Get document by ID")
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Retrieve a specific document by its ID."""
    document = DocumentCRUD.get_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.id,
        title=document.title,
        text=document.text,
        word_count=document.word_count,
        status=document.status,
        error_message=document.error_message,
        created_at=document.created_at.isoformat()
    )

@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse, 
         summary="Check document processing status")
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    """Check the current processing status of a document."""
    document = DocumentCRUD.get_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentStatusResponse(
        id=document.id,
        status=document.status,
        error_message=document.error_message
    )

@app.get("/documents/{document_id}/progress")
async def stream_document_progress(document_id: str):
    """
    Server-Sent Events endpoint for real-time progress updates.
    
    Usage:
        const eventSource = new EventSource('/documents/{id}/progress');
        eventSource.onmessage = (event) => {
            const progress = JSON.parse(event.data);
            console.log(progress.message, progress.progress);
        };
    """
    async def event_generator():
        """Generate SSE events for progress updates."""
        try:
            while True:
                progress = progress_tracker.get_progress(document_id)
                
                if progress:
                    # Send progress update
                    yield {
                        "event": "progress",
                        "data": json.dumps(progress)
                    }
                    
                    # Stop streaming if task is completed or failed
                    if progress["status"] in ["completed", "failed"]:
                        yield {
                            "event": "done",
                            "data": json.dumps({"status": progress["status"]})
                        }
                        break
                else:
                    # Task not found in progress tracker yet
                    yield {
                        "event": "waiting",
                        "data": json.dumps({"message": "Waiting for analysis to start..."})
                    }
                
                await asyncio.sleep(0.5)  # Update every 500ms
                
        except asyncio.CancelledError:
            print(f"Progress stream cancelled for document {document_id}")
    
    return EventSourceResponse(event_generator())

@app.delete("/documents/{document_id}", summary="Delete a document")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and its associated analysis."""
    success = DocumentCRUD.delete(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": f"Document {document_id} deleted successfully"}

# ==================== ANALYSIS ENDPOINTS ====================

@app.get("/documents/{document_id}/analysis", response_model=AnalysisResponse,
         summary="Get analysis for a single document")
def get_document_analysis(document_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the knowledge graph analysis for a completed document.
    Returns 404 if document doesn't exist, 202 if still processing.
    """
    document = DocumentCRUD.get_by_id(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.status == "processing" or document.status == "pending":
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    
    if document.status == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {document.error_message}")
    
    analysis = AnalysisCRUD.get_by_document(db, document_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return AnalysisResponse(analysis_data={
        "entities": analysis.entities,
        "relationships": analysis.relationships,
        "hierarchy": analysis.hierarchy
    })

@app.post("/graphs/merge", response_model=AnalysisResponse, 
          summary="Merge knowledge graphs from multiple documents")
async def merge_document_graphs(request: MergeGraphRequest, db: Session = Depends(get_db)):
    """
    Merge knowledge graphs from multiple documents.
    Uses cached results when available.
    """
    # Single document - just return its analysis
    if len(request.document_ids) == 1:
        return get_document_analysis(request.document_ids[0], db)
    
    # Check cache first
    cached = MergedGraphCRUD.get_by_document_ids(db, request.document_ids)
    if cached:
        print(f"✅ Cache hit for document combination: {request.document_ids}")
        return AnalysisResponse(analysis_data={
            "entities": cached.entities,
            "relationships": cached.relationships,
            "hierarchy": cached.hierarchy
        })
    
    print(f"⚠️  Cache miss - merging graphs for: {request.document_ids}")
    
    # Verify all documents are completed
    for doc_id in request.document_ids:
        doc = DocumentCRUD.get_by_id(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        if doc.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Document '{doc.title}' not yet analyzed (status: {doc.status})"
            )
    
    # Get all analyses
    analyses = AnalysisCRUD.get_multiple(db, request.document_ids)
    if len(analyses) != len(request.document_ids):
        raise HTTPException(status_code=500, detail="Some document analyses are missing")
    
    # --- FIX: Call the new, sophisticated merge_analyses helper ---
    merged_data = await merge_analyses(request.document_ids, analyses, db)
    # --- End Fix ---
    
    # Cache the result
    MergedGraphCRUD.create(
        db,
        document_ids=request.document_ids,
        entities=merged_data["entities"],
        relationships=merged_data["relationships"],
        hierarchy=merged_data["hierarchy"]
    )
    
    return AnalysisResponse(analysis_data=merged_data)

# ==================== SYSTEM ENDPOINTS ====================

@app.get("/system/progress-summary")
def get_progress_summary():
    """Get summary of all processing tasks."""
    return progress_tracker.get_task_summary()

@app.get("/system/active-tasks")
def get_active_tasks():
    """Get all currently processing tasks."""
    return progress_tracker.get_active_tasks()

# ==================== HELPER FUNCTIONS ====================

# --- FIX: Replace the entire merge_analyses function ---
async def merge_analyses(document_ids: List[str], analyses: List[DocumentAnalysis], db: Session) -> Dict[str, Any]:
    """
    Merge multiple document analyses using Phase 3 entity/relationship/hierarchy resolvers.
    """
    
    # 0. Get LLM client and model name from environment
    model_name = os.getenv("DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
    
    # 1. Prepare data for resolvers
    entities_by_doc = {}
    relationships_by_doc = {}
    hierarchies_by_doc = {}
    doc_metadata = {}
    
    # We need the original Document objects for metadata (titles)
    documents = db.query(Document).filter(Document.id.in_(document_ids)).all()
    doc_map = {doc.id: doc for doc in documents}
    
    for analysis in analyses:
        doc_id = analysis.document_id
        if doc_id in doc_map:
            entities_by_doc[doc_id] = analysis.entities
            relationships_by_doc[doc_id] = analysis.relationships
            hierarchies_by_doc[doc_id] = analysis.hierarchy
            doc_metadata[doc_id] = {
                "title": doc_map[doc_id].title,
                "word_count": doc_map[doc_id].word_count
            }
        
    # 2. Initialize Resolvers
    # We pass the globally initialized llm_client from analysis_logic
    entity_resolver = EntityResolver(client=llm_client, model_name=model_name)
    relationship_resolver = RelationshipResolver()
    hierarchy_merger = HierarchyMerger(client=llm_client, model_name=model_name)
    
    # 3. --- STAGE 1: Resolve Entities ---
    # This is the core fix for the user's problem
    print("Starting Entity Resolution...")
    merged_entities, entity_map = await entity_resolver.resolve_entities(
        entities_by_doc,
        doc_metadata
    )
    
    # 4. --- STAGE 2: Resolve Relationships ---
    print("Starting Relationship Resolution...")
    # Rewrite all relationships to use canonical names
    rewritten_rels = relationship_resolver.rewrite_relationships(
        relationships_by_doc,
        entity_map
    )
    # Deduplicate the rewritten relationships
    merged_relationships = relationship_resolver.deduplicate_and_resolve_conflicts(
        rewritten_rels
    )
    
    # 5. --- STAGE 3: Merge Hierarchies ---
    print("Starting Hierarchy Fusion...")
    merged_hierarchy = await hierarchy_merger.merge_hierarchies(
        hierarchies_by_doc,
        doc_metadata
    )
    
    print(f"📊 Merge stats: {len(entity_map)} original entities -> {len(merged_entities)} canonical entities")
    print(f"📊 Merge stats: {len(rewritten_rels)} original relationships -> {len(merged_relationships)} canonical relationships")
    print(f"📊 Merge stats: {len(hierarchies_by_doc)} hierarchies -> {len(merged_hierarchy)} merged topics")
    
    return {
        "entities": merged_entities,
        "relationships": merged_relationships,
        "hierarchy": merged_hierarchy
    }
# --- End Fix ---
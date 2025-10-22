# backend/database.py
import os
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import List, Optional
import hashlib
import json

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cognisynth.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== MODELS ====================

class Document(Base):
    """Stores user-uploaded text documents."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)  # UUID
    title = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    content_hash = Column(String, unique=True, index=True)  # For deduplication
    word_count = Column(Integer)
    
    # Processing status
    status = Column(String, default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title}, status={self.status})>"
    
    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


class DocumentAnalysis(Base):
    """Stores the analyzed knowledge graph for each document."""
    __tablename__ = "document_analyses"
    
    id = Column(String, primary_key=True)  # UUID
    document_id = Column(String, index=True, nullable=False)  # FK to documents.id
    
    # Analysis results (stored as JSON)
    entities = Column(JSON)  # List[Dict] - extracted entities
    relationships = Column(JSON)  # List[Dict] - relationships between entities
    hierarchy = Column(JSON)  # List[Dict] - logical flow/hierarchy
    
    # Metadata
    entity_count = Column(Integer)
    relationship_count = Column(Integer)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_time_seconds = Column(Float)
    
    def __repr__(self):
        return f"<DocumentAnalysis(doc_id={self.document_id}, entities={self.entity_count})>"


class MergedGraph(Base):
    """Caches merged knowledge graphs for document combinations."""
    __tablename__ = "merged_graphs"
    
    id = Column(String, primary_key=True)  # UUID
    document_ids = Column(JSON, index=True)  # Sorted list of document IDs
    cache_key = Column(String, unique=True, index=True)  # Hash of sorted doc IDs
    
    # Merged graph data
    entities = Column(JSON)  # Merged + deduplicated entities
    relationships = Column(JSON)  # Merged relationships
    hierarchy = Column(JSON)  # Combined hierarchy
    
    # Metadata
    entity_count = Column(Integer)
    relationship_count = Column(Integer)
    source_count = Column(Integer)  # Number of documents merged
    
    # Cache management
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    access_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<MergedGraph(sources={self.source_count}, entities={self.entity_count})>"
    
    @staticmethod
    def compute_cache_key(document_ids: List[str]) -> str:
        """Generate cache key from sorted document IDs."""
        sorted_ids = sorted(document_ids)
        key_string = "_".join(sorted_ids)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()


class EntityMapping(Base):
    """Tracks entity resolution across documents (for deduplication)."""
    __tablename__ = "entity_mappings"
    
    id = Column(String, primary_key=True)  # UUID
    canonical_name = Column(String, index=True)  # The "true" entity name
    document_id = Column(String, index=True)
    original_name = Column(String)  # Name as it appeared in the document
    similarity_score = Column(Float)  # Confidence in the mapping (0-1)
    
    # Context for disambiguation
    category = Column(String)
    description = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<EntityMapping('{self.original_name}' → '{self.canonical_name}')>"


# ==================== CRUD OPERATIONS ====================

class DocumentCRUD:
    """Database operations for Documents."""
    
    @staticmethod
    def create(db: Session, title: str, text: str) -> Document:
        """Create a new document."""
        import uuid
        
        content_hash = Document.compute_hash(text)
        word_count = len(text.split())
        
        doc = Document(
            id=str(uuid.uuid4()),
            title=title,
            text=text,
            content_hash=content_hash,
            word_count=word_count,
            status="pending"
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc
    
    @staticmethod
    def get_by_id(db: Session, document_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return db.query(Document).filter(Document.id == document_id).first()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
        """Get all documents."""
        return db.query(Document).offset(skip).limit(limit).all()
    
    @staticmethod
    def update_status(db: Session, document_id: str, status: str, error: str = None):
        """Update document processing status."""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = status
            doc.error_message = error
            doc.updated_at = datetime.utcnow()
            db.commit()
    
    @staticmethod
    def delete(db: Session, document_id: str) -> bool:
        """Delete a document and its analysis."""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            # Also delete associated analysis
            db.query(DocumentAnalysis).filter(DocumentAnalysis.document_id == document_id).delete()
            db.delete(doc)
            db.commit()
            return True
        return False
    
    @staticmethod
    def exists_by_hash(db: Session, content_hash: str) -> Optional[str]:
        """Check if document with same content exists, return its ID."""
        doc = db.query(Document).filter(Document.content_hash == content_hash).first()
        return doc.id if doc else None


class AnalysisCRUD:
    """Database operations for DocumentAnalysis."""
    
    @staticmethod
    def create(db: Session, document_id: str, entities: List, relationships: List, 
               hierarchy: List, processing_time: float) -> DocumentAnalysis:
        """Store analysis results."""
        import uuid
        
        analysis = DocumentAnalysis(
            id=str(uuid.uuid4()),
            document_id=document_id,
            entities=entities,
            relationships=relationships,
            hierarchy=hierarchy,
            entity_count=len(entities),
            relationship_count=len(relationships),
            processing_time_seconds=processing_time
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis
    
    @staticmethod
    def get_by_document(db: Session, document_id: str) -> Optional[DocumentAnalysis]:
        """Get analysis for a specific document."""
        return db.query(DocumentAnalysis).filter(
            DocumentAnalysis.document_id == document_id
        ).first()
    
    @staticmethod
    def get_multiple(db: Session, document_ids: List[str]) -> List[DocumentAnalysis]:
        """Get analyses for multiple documents."""
        return db.query(DocumentAnalysis).filter(
            DocumentAnalysis.document_id.in_(document_ids)
        ).all()


class MergedGraphCRUD:
    """Database operations for MergedGraph cache."""
    
    @staticmethod
    def create(db: Session, document_ids: List[str], entities: List, 
               relationships: List, hierarchy: List) -> MergedGraph:
        """Cache a merged graph."""
        import uuid
        
        cache_key = MergedGraph.compute_cache_key(document_ids)
        
        merged = MergedGraph(
            id=str(uuid.uuid4()),
            document_ids=sorted(document_ids),
            cache_key=cache_key,
            entities=entities,
            relationships=relationships,
            hierarchy=hierarchy,
            entity_count=len(entities),
            relationship_count=len(relationships),
            source_count=len(document_ids)
        )
        db.add(merged)
        db.commit()
        db.refresh(merged)
        return merged
    
    @staticmethod
    def get_by_document_ids(db: Session, document_ids: List[str]) -> Optional[MergedGraph]:
        """Retrieve cached merged graph."""
        cache_key = MergedGraph.compute_cache_key(document_ids)
        merged = db.query(MergedGraph).filter(MergedGraph.cache_key == cache_key).first()
        
        if merged:
            # Update access tracking
            merged.last_accessed_at = datetime.utcnow()
            merged.access_count += 1
            db.commit()
        
        return merged
    
    @staticmethod
    def delete_by_cache_key(db: Session, cache_key: str) -> bool:
        """Delete a cached merged graph."""
        merged = db.query(MergedGraph).filter(MergedGraph.cache_key == cache_key).first()
        if merged:
            db.delete(merged)
            db.commit()
            return True
        return False
    
    @staticmethod
    def cleanup_old_caches(db: Session, days: int = 30, keep_top_n: int = 100):
        """Remove old, rarely accessed caches to save space."""
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Keep top N most accessed, delete rest older than cutoff
        top_caches = db.query(MergedGraph).order_by(
            MergedGraph.access_count.desc()
        ).limit(keep_top_n).all()
        
        top_ids = [c.id for c in top_caches]
        
        db.query(MergedGraph).filter(
            MergedGraph.last_accessed_at < cutoff_date,
            ~MergedGraph.id.in_(top_ids)
        ).delete(synchronize_session=False)
        
        db.commit()


# Initialize database tables
def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")


if __name__ == "__main__":
    init_db()
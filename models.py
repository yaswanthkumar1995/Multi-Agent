"""
Simplified models for Multi-Agent competitive intelligence system.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Search result model."""
    title: str
    url: str
    snippet: str
    source: str = ""
    date: Optional[str] = None


class ProductUpdate(BaseModel):
    """Product update model."""
    product: str
    update: str
    source: str
    date: str
    confidence_score: float = 0.0
    relevance_score: float = 0.0


class IntelligenceReport(BaseModel):
    """Intelligence report model."""
    query: str
    category: str
    updates: List[ProductUpdate]
    total_sources: int
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

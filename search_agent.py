"""
SearchAgent for competitive intelligence gathering.
"""
import logging
from typing import List
from ddgs import DDGS
from models import SearchResult

logger = logging.getLogger(__name__)


class SearchAgent:
    """Simplified SearchAgent - fetches top 5 relevant sources."""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search for relevant sources."""
        try:
            logger.info(f"🔍 SearchAgent: Searching for '{query}'")
            
            # Enhanced query for better results
            enhanced_query = f"{query} latest updates news features"
            
            results = []
            ddg_results = self.ddgs.text(enhanced_query, max_results=max_results)
            
            for result in ddg_results:
                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    source=self._extract_domain(result.get('href', ''))
                )
                results.append(search_result)
            
            logger.info(f"✅ SearchAgent: Found {len(results)} sources")
            return results
            
        except Exception as e:
            logger.error(f"❌ SearchAgent failed: {e}")
            return []
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ""

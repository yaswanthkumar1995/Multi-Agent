"""
VerifierAgent for content verification and quality assessment.
"""
import logging
from typing import List
from models import ProductUpdate

logger = logging.getLogger(__name__)


class VerifierAgent:
    """Simplified VerifierAgent - filters unreliable information."""
    
    def __init__(self):
        # Trusted domains for reliability scoring
        self.trusted_domains = {
            'techcrunch.com', 'theverge.com', 'wired.com', 'arstechnica.com',
            'reuters.com', 'bloomberg.com', 'venturebeat.com', 'zdnet.com',
            'cnet.com', 'engadget.com', 'github.com', 'openai.com',
            'microsoft.com', 'google.com', 'tesla.com', 'notion.so'
        }
    
    def verify(self, updates: List[ProductUpdate]) -> List[ProductUpdate]:
        """Filter and score updates for reliability."""
        try:
            logger.info(f"✅ VerifierAgent: Verifying {len(updates)} updates")
            
            verified_updates = []
            for update in updates:
                # Score the update
                confidence = self._calculate_confidence(update)
                relevance = self._calculate_relevance(update)
                
                # Update scores
                update.confidence_score = confidence
                update.relevance_score = relevance
                
                # Keep updates with minimum quality threshold
                if confidence > 0.3 and relevance > 0.3:
                    verified_updates.append(update)
            
            # Sort by quality score (confidence * relevance)
            verified_updates.sort(
                key=lambda x: x.confidence_score * x.relevance_score, 
                reverse=True
            )
            
            logger.info(f"✅ VerifierAgent: {len(verified_updates)} updates passed verification")
            return verified_updates
            
        except Exception as e:
            logger.error(f"❌ VerifierAgent failed: {e}")
            return updates  # Return original if verification fails
    
    def _calculate_confidence(self, update: ProductUpdate) -> float:
        """Calculate confidence score based on source reliability."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(update.source).netloc.lower()
            
            # Check if source is trusted
            if any(trusted in domain for trusted in self.trusted_domains):
                return 0.9
            elif domain.endswith('.edu') or domain.endswith('.gov'):
                return 0.8
            elif 'news' in domain or 'tech' in domain:
                return 0.7
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_relevance(self, update: ProductUpdate) -> float:
        """Calculate relevance score based on content quality."""
        text = f"{update.product} {update.update}".lower()
        
        # Relevance keywords
        relevant_keywords = [
            'update', 'feature', 'release', 'launch', 'announce', 'new',
            'improve', 'enhance', 'version', 'ai', 'tool', 'product'
        ]
        
        # Irrelevant keywords
        spam_keywords = [
            'click here', 'subscribe', 'advertisement', 'sponsored',
            'buy now', 'discount', 'sale', 'free trial'
        ]
        
        # Count relevant keywords
        relevant_count = sum(1 for keyword in relevant_keywords if keyword in text)
        spam_count = sum(1 for keyword in spam_keywords if keyword in text)
        
        # Calculate score
        relevance = relevant_count / len(relevant_keywords)
        spam_penalty = spam_count * 0.2
        
        return max(0.0, min(1.0, relevance - spam_penalty))

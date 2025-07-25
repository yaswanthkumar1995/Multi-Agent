"""
Summarizer Agent for extracting and formatting product updates from content.
"""
import re
import json
import logging
import os
"""
Summarizer Agent for extracting and formatting product updates from content.
"""
import re
import json
import logging
import os
from typing import List, Dict, Any
from datetime import datetime
from models import ProductUpdate, SearchResult

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Custom guardrails for input/output validation
class SimpleGuardrails:
    """Simple input/output validation without external dependencies."""
    
    @staticmethod
    def validate_input(text: str) -> bool:
        """Basic input validation using LLM for intelligent detection."""
        # Check for basic harmful patterns
        harmful_patterns = [
            'hack', 'exploit', 'malware', 'virus', 'attack',
            'bomb', 'kill', 'murder', 'suicide', 'harm'
        ]
        
        text_lower = text.lower().strip()
        for pattern in harmful_patterns:
            if pattern in text_lower:
                return False
        
        # Check for simple greetings that shouldn't trigger product searches
        simple_greetings = [
            'hey', 'hi', 'hello', 'yo', 'sup', 'hey man', 'hi there',
            'good morning', 'good afternoon', 'good evening', 'what\'s up'
        ]
        
        # If it's just a greeting, don't process as tech query
        if text_lower in simple_greetings:
            return False
        
        # Special check for very short queries (1 word only) - let LLM decide
        if len(text_lower.split()) == 1 and len(text_lower) < 3:
            return False
        
        # For longer queries, let the LLM in the agent decide if it's tech-related
        # This removes the hardcoded limitation
        return True
    
    @staticmethod
    def validate_output(text: str) -> bool:
        """Basic output validation."""
        # Check output isn't empty or too short
        if not text or len(text.strip()) < 10:
            return False
            
        # Check for harmful content in output
        harmful_patterns = ['hack', 'exploit', 'illegal', 'harmful']
        text_lower = text.lower()
        
        return not any(pattern in text_lower for pattern in harmful_patterns)

logger = logging.getLogger(__name__)

class SummarizerAgent:
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        self._init_llm()
        self._init_guardrails()
    
    def _init_guardrails(self):
        """Initialize simple guardrails for input/output validation"""
        self.guardrails = SimpleGuardrails()
        self.use_guardrails = True
        print("✅ Simple Guardrails initialized for input/output validation")
    
    def _validate_query_with_llm(self, query: str) -> bool:
        """Use LLM to intelligently validate if query is technology-related."""
        if not hasattr(self, 'groq_llm') or not self.use_groq_summarization:
            # Fallback to basic validation if LLM not available
            return self.guardrails.validate_input(query)
        
        try:
            from langchain_core.messages import HumanMessage
            
            validation_prompt = f"""
Is this query about technology? Answer only YES or NO.

Query: "{query}"

Technology includes: AWS, Google Cloud, Microsoft, Apple, software, apps, AI, ChatGPT, SageMaker, Kubernetes, etc.

Answer:"""

            response = self.groq_llm.invoke([HumanMessage(content=validation_prompt)])
            answer = response.content.strip().upper()
            
            return answer == "YES"
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}, using fallback")
            return self.guardrails.validate_input(query)
    
    def _correct_typos_with_llm(self, query: str) -> str:
        """Use LLM to correct typos and understand user intent."""
        if not hasattr(self, 'groq_llm') or not self.use_groq_summarization:
            return query  # Return original if LLM not available
        
        try:
            from langchain_core.messages import HumanMessage
            
            correction_prompt = f"""
Fix typos in this query. Return ONLY the corrected query, no explanations.

Query: "{query}"

Common tech typos:
- sagmaker → sagemaker
- chatgtp → chatgpt  
- kuberntes → kubernetes
- aws ec → aws ec2
- googl → google

Just return the corrected text:"""

            response = self.groq_llm.invoke([HumanMessage(content=correction_prompt)])
            corrected_query = response.content.strip()
            
            # Remove quotes if present
            if corrected_query.startswith('"') and corrected_query.endswith('"'):
                corrected_query = corrected_query[1:-1]
            if corrected_query.startswith("'") and corrected_query.endswith("'"):
                corrected_query = corrected_query[1:-1]
            
            # Basic validation of corrected query
            if corrected_query and len(corrected_query) > 0:
                return corrected_query
            else:
                return query  # Return original if correction failed
                
        except Exception as e:
            logger.warning(f"LLM typo correction failed: {e}, using original query")
            return query
    
    def _init_llm(self):
        """Initialize Groq LLM for summarization"""
        try:
            from langchain_groq import ChatGroq
            
            # Get Groq API key from .env file
            groq_api_key = os.getenv('GROQ_API_KEY')
            
            if groq_api_key:
                
                # Initialize ChatGroq with llama3-8b-8192 model
                self.groq_llm = ChatGroq(
                    model="llama3-8b-8192",
                    api_key=groq_api_key,
                    temperature=0.3,
                    max_tokens=500
                )
                print("✅ Groq summarization model loaded: llama3-8b-8192")
                self.use_groq_summarization = True
                
            else:
                print("⚠️ No Groq API key found in .env file, using rule-based summarization")
                self.use_groq_summarization = False
                
        except Exception as e:
            logger.warning(f"⚠️ Groq model failed to load: {e}")
            logger.warning("⚠️ Using rule-based summarization")
            self.use_groq_summarization = False
    
    def summarize(self, search_results: List[SearchResult], category: str) -> List[ProductUpdate]:
        """Extract product updates from search results."""
        try:
            logger.info(f"📝 SummarizerAgent: Processing {len(search_results)} results")
            
            updates = []
            for result in search_results:
                # Simple extraction from snippet
                update = self._extract_update(result, category, "")
                if update:
                    updates.append(update)
            
            logger.info(f"✅ SummarizerAgent: Generated {len(updates)} updates")
            return updates
            
        except Exception as e:
            logger.error(f"❌ SummarizerAgent failed: {e}")
            return []
    
    def summarize_from_json(self, search_results_json: str, category: str, query: str = "") -> List[ProductUpdate]:
        """Extract product updates from JSON search results."""
        try:
            # Correct typos in query first
            if query:
                corrected_query = self._correct_typos_with_llm(query)
                if corrected_query != query:
                    logger.info(f"Typo correction: '{query}' → '{corrected_query}'")
                query = corrected_query
            
            # Input validation with LLM-based intelligent detection
            if self.use_guardrails and query:
                if not self._validate_query_with_llm(query):
                    logger.warning(f"LLM validation failed for query: {query}")
                    return []  # Return empty if input fails validation
            
            logger.info("📝 SummarizerAgent: Processing results")
            
            # Parse JSON data
            import json
            data = json.loads(search_results_json)
            results = data.get('results', [])
            
            updates = []
            for result_data in results:
                # Convert JSON to SearchResult-like object
                class MockSearchResult:
                    def __init__(self, data):
                        self.url = data.get('url', '')
                        self.title = data.get('title', '')
                        self.snippet = data.get('content', '')  # Map content to snippet
                        self.source = data.get('domain', '')
                
                mock_result = MockSearchResult(result_data)
                update = self._extract_update(mock_result, category, query)
                if update:
                    updates.append(update)
            
            logger.info(f"✅ SummarizerAgent: Generated {len(updates)} updates")
            return updates
            
        except Exception as e:
            logger.error(f"❌ SummarizerAgent failed: {e}")
            return []
    
    def _extract_update(self, result: SearchResult, category: str, query: str = "") -> ProductUpdate:
        """Extract update from search result using Groq LLM."""
        try:
            # Combine title and snippet for processing
            snippet = result.snippet
            title = result.title
            full_text = f"{title}. {snippet}"
            
            # Extract product name (enhanced with query context)
            product = self._extract_product_name(title, snippet, category, query)
            
            # Extract actual date from content
            extracted_date = self._extract_date_from_content(full_text)
            
            # Use Groq LLM for intelligent summarization if available
            if hasattr(self, 'use_groq_summarization') and self.use_groq_summarization and hasattr(self, 'groq_llm'):
                try:
                    from langchain_core.messages import HumanMessage
                    
                    # Create enhanced summarization prompt that preserves dates
                    summarization_prompt = f"""
Summarize this content about a technology product or service. Return ONLY the summary, no explanations or instructions.

Title: {title}
Content: {snippet}

Rules for your response:
1. Start directly with the main fact or announcement
2. Maximum 100 words
3. Include any dates, versions, or specific details mentioned
4. Focus on what's new or recently announced
5. Do NOT include phrases like "Here is", "This is", "The following"

Summary:"""

                    # Get Groq LLM summary
                    response = self.groq_llm.invoke([HumanMessage(content=summarization_prompt)])
                    update_text = response.content.strip()
                    
                    # Clean up any leaked instructions
                    instruction_patterns = [
                        r"CRITICAL:.*?(?=\n\n|\.$|$)",
                        r"Requirements:.*?(?=\n\n|\.$|$)",
                        r"Direct information only:.*?(?=\n\n|\.$|$)",
                        r"❌.*?(?=\n|$)",
                        r"✅.*?(?=\n|$)"
                    ]
                    
                    for pattern in instruction_patterns:
                        update_text = re.sub(pattern, "", update_text, flags=re.DOTALL | re.IGNORECASE)
                    
                    update_text = update_text.strip()
                    
                    # Output validation with Simple Guardrails
                    if self.use_guardrails:
                        # Be more lenient - just check basic length and content
                        if len(update_text) < 10:
                            logger.warning(f"Output validation failed: too short")
                            update_text = full_text[:300]
                    
                    # Clean up the update text
                    if not update_text.endswith('.'):
                        update_text += '.'
                        
                except Exception as e:
                    logger.warning(f"Groq summarization failed: {e}, using fallback")
                    update_text = full_text[:300]
            else:
                # Fallback to simple truncation
                update_text = full_text[:300]
            
            # Clean up the update text
            update_text = update_text.strip()
            if not update_text.endswith('.'):
                update_text += '.'
            
            return ProductUpdate(
                product=product,
                update=update_text,
                source=result.url,
                date=extracted_date,  # Use extracted date instead of current date
                confidence_score=0.9,  # Higher confidence with Groq LLM
                relevance_score=0.95   # Higher relevance with intelligent processing
            )
            
        except Exception as e:
            logger.debug(f"Update extraction failed: {e}")
            return None
    
    def _extract_date_from_content(self, content: str) -> str:
        """Extract actual publication date from content, return 'none' if not found."""
        import re
        
        # Check for relative dates first (always return "none" for these)
        relative_patterns = [
            r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago',
            r'yesterday',
            r'today',
            r'recently',
            r'last\s+(week|month|year)',
            r'this\s+(week|month|year)'
        ]
        
        for pattern in relative_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return "none"
        
        # Look for publication date indicators with actual dates
        publication_patterns = [
            # Published/Released/Announced with specific date
            r'(?:published|released|announced|launched|available)\s+(?:on\s+)?(?:in\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
            # "on [date]" patterns
            r'(?:on\s+)(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
            # Date with publication context
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\s*(?:release|publication|announcement|launch)',
            # ISO dates with publication context  
            r'(?:published|released|announced|launched)\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})',
            # Paper publication dates (arXiv, academic)
            r'(?:arxiv|published|submitted):\s*(\d{4}-\d{2}-\d{2})',
            r'(?:published|submitted)\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})',
            # Version releases with dates
            r'(?:version|v\d+\.?\d*)\s+(?:released|published)\s+(?:on\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
            # News article dates
            r'(?:updated|posted|written)\s+(?:on\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})'
        ]
        
        # Check publication-specific patterns first
        for pattern in publication_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                match = matches[0]
                
                # Handle different match formats
                if isinstance(match, tuple) and len(match) == 3:
                    # Month Day Year format
                    if match[0].isdigit():  # ISO format variation
                        return match[0]  # Already in YYYY-MM-DD format
                    else:  # Month name format
                        month_name, day, year = match
                        return f"{year}-{self._month_to_number(month_name)}-{day.zfill(2)}"
                elif isinstance(match, str) and re.match(r'\d{4}-\d{2}-\d{2}', match):
                    # ISO format
                    return match
        
        # If no publication context found, look for structured date formats only in news/paper contexts
        if any(keyword in content.lower() for keyword in ['news', 'paper', 'article', 'journal', 'conference', 'proceedings', 'arxiv']):
            # Only extract dates if we're in a publication context
            general_patterns = [
                # Specific dates like "July 25, 2025" 
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
                # ISO format dates
                r'(\d{4}-\d{2}-\d{2})',
                # Dates like "25 July 2025"
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
            ]
            
            for pattern in general_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    match = matches[0]
                    
                    if isinstance(match, tuple):
                        if len(match) == 3:
                            if match[0].isdigit():  # Day Month Year
                                return f"{match[2]}-{self._month_to_number(match[1])}-{match[0].zfill(2)}"
                            else:  # Month Day Year  
                                return f"{match[2]}-{self._month_to_number(match[0])}-{match[1].zfill(2)}"
                    elif isinstance(match, str) and re.match(r'\d{4}-\d{2}-\d{2}', match):
                        return match
        
        # If no valid publication date found, return "none"
        return "none"
    
    def _month_to_number(self, month_name: str) -> str:
        """Convert month name to number."""
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        return months.get(month_name.lower(), '01')
    
    def _extract_product_name(self, title: str, snippet: str, category: str, query: str = "") -> str:
        """Extract product name using Groq LLM for intelligent detection."""
        
        # Use Groq LLM for intelligent product extraction if available
        if hasattr(self, 'use_groq_summarization') and self.use_groq_summarization and hasattr(self, 'groq_llm'):
            try:
                from langchain_core.messages import HumanMessage
                
                # Create intelligent product extraction prompt
                product_extraction_prompt = f"""
Extract the main product or service name from this content. Return ONLY the product/company/service name, nothing else.

Title: {title}
Content: {snippet}
Query context: {query}

Instructions:
- Correct any typos in the product name (e.g., "sagmaker" → "SageMaker", "chatgtp" → "ChatGPT")
- Return the MOST SPECIFIC product or service name mentioned
- For AWS, Google, Azure services, prefer specific service names over generic company names
- If query mentions a specific service, prioritize that service name
- Use proper capitalization (e.g., "SageMaker", "ChatGPT", "Google Cloud")
- If multiple products mentioned, choose the one most relevant to the query
- If content is not about technology/products, return "Technology" as fallback

Product name:"""

                # Get Groq LLM response
                response = self.groq_llm.invoke([HumanMessage(content=product_extraction_prompt)])
                product_name = response.content.strip()
                
                # Output validation with Simple Guardrails for product name
                if self.use_guardrails:
                    # Be more lenient with product names - only check for basic safety
                    if len(product_name) < 2 or len(product_name) > 50:
                        logger.warning(f"Product name validation failed: length issue")
                        product_name = "Technology"
                
                # Clean and validate the response
                if product_name and len(product_name) > 0 and len(product_name) < 50:
                    return product_name
                    
            except Exception as e:
                logger.warning(f"Groq product extraction failed: {e}, using fallback")
        
        # Fallback: Simple extraction from title
        words = title.split()
        for word in words:
            if word and len(word) > 2 and word[0].isupper():
                return word
        
        return "Technology"

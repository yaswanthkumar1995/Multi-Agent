"""
Ultra-Simple Multi-Agent Intelligence System 
Reduced from 1400+ lines to ~150 lines while maintaining functionality
Inspired by a2a-samples elegant approach
"""
import os
import json
from datetime import datetime
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from dotenv import load_dotenv

# Import existing agents
from search_agent import SearchAgent
from summarizer_agent import SummarizerAgent
from verifier_agent import VerifierAgent

load_dotenv()


class SimpleIntelligenceAgent:
    """Ultra-simple competitive intelligence assistant - like a2a-samples but for intelligence."""

    SYSTEM_PROMPT = """You are a friendly competitive intelligence assistant. 

Your personality:
- Warm and conversational
- Helpful and knowledgeable
- Professional but approachable

Your capabilities:
1. Handle greetings warmly (hi, hello, hey)
2. Respond to personal introductions with personalized greetings (i am john, my name is alice)
3. For product analysis queries, use your tools to search and analyze
4. Explain what you can do when asked

For greetings: Introduce yourself and explain your capabilities
For introductions: Greet them personally by name and offer help
For product/company queries: Search, analyze, and provide structured results
For unclear requests: Ask for clarification politely

Always be conversational and helpful."""

    def __init__(self):
        # Initialize Groq model
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            model="llama3-8b-8192",
            api_key=groq_api_key,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Initialize sub-agents
        self.search_agent = SearchAgent()
        self.summarizer_agent = SummarizerAgent()
        self.verifier_agent = VerifierAgent()
        
        print("✅ Simple Intelligence Agent initialized")

    def _correct_typos(self, query: str) -> str:
        """Use Groq to correct typos in user input."""
        try:
            correction_prompt = f"""Fix typos in this query while preserving exact meaning:

Examples:
- "i ma john" → "i am john"
- "waht is tesla" → "what is tesla" 
- "my naem is alice" → "my name is alice"

Query: "{query}"
Corrected:"""

            response = self.llm.invoke([HumanMessage(content=correction_prompt)])
            corrected = response.content.strip().strip('"')
            
            # Basic validation
            if len(corrected) > 0 and len(corrected) < len(query) * 2:
                return corrected
            return query
            
        except Exception:
            return query

    def _is_product_query(self, query: str) -> bool:
        """Check if this is a product analysis query."""
        analysis_keywords = [
            "what", "tell me", "analyze", "latest", "update", "features", 
            "news", "about", "how", "when", "compare", "vs", "versus"
        ]
        return (any(keyword in query.lower() for keyword in analysis_keywords) 
                and len(query.split()) > 2)

    def _search_and_analyze(self, query: str) -> str:
        """Search and analyze for product queries."""
        try:
            # Step 1: Search
            enhanced_query = f"{query} latest updates 2025"
            search_results = self.search_agent.search(enhanced_query, max_results=3)
            
            if not search_results:
                return json.dumps({
                    "product": query.split()[0] if query.split() else "Unknown",
                    "update": "No recent updates found",
                    "source": "Search Agent",
                    "date": "none"
                }, indent=2)
            
            # Step 2: Summarize
            search_data = {
                "status": "success",
                "results": [
                    {
                        "title": result.title,
                        "content": result.snippet,
                        "url": result.url,
                        "source": result.source
                    }
                    for result in search_results
                ]
            }
            
            search_json = json.dumps(search_data)
            updates = self.summarizer_agent.summarize_from_json(search_json, "product_update")
            
            if not updates:
                return json.dumps({
                    "product": query.split()[0] if query.split() else "Unknown",
                    "update": "Failed to analyze search results", 
                    "source": "Summarizer Agent",
                    "date": "none"
                }, indent=2)
            
            # Step 3: Verify
            verified_updates = self.verifier_agent.verify(updates)
            
            if verified_updates:
                update = verified_updates[0]
                return json.dumps({
                    "product": update.product,
                    "update": update.update,
                    "source": update.source,
                    "date": update.date
                }, indent=2)
            
            return json.dumps({
                "product": query.split()[0] if query.split() else "Unknown",
                "update": "Verification failed",
                "source": "Verifier Agent", 
                "date": "none"
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "product": "Error",
                "update": f"Analysis failed: {str(e)[:100]}",
                "source": "System Error",
                "date": "none"
            }, indent=2)

    def get_simple_analysis(self, query: str) -> str:
        """Main method for processing queries - ultra-simple approach."""
        try:
            # Step 1: Correct typos
            corrected_query = self._correct_typos(query)
            
            # Step 2: Check if this is a product analysis query
            if self._is_product_query(corrected_query):
                return self._search_and_analyze(corrected_query)
            
            # Step 3: Handle conversational queries with LLM
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=corrected_query)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            return f"❌ Error: {str(e)}"


# For backwards compatibility with existing Streamlit interface
class LangGraphCoordinatorAgent(SimpleIntelligenceAgent):
    """Alias for backwards compatibility with existing Streamlit code."""
    pass


if __name__ == "__main__":
    # Test the ultra-simple agent
    agent = SimpleIntelligenceAgent()
    
    test_queries = [
        "hi",
        "i ma yaswanth",  # Test typo correction
        "my naem is alice",  # Test typo correction + introduction
        "what are the latest ChatGPT features?",  # Test product analysis
        "tell me about Tesla updates"  # Test product analysis
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        response = agent.get_simple_analysis(query)
        print(f"📝 Response: {response[:200]}..." if len(response) > 200 else f"📝 Response: {response}")
        print("-" * 80)

"""
Professional Multi-Agent Intelligence System
Main entry point for the streamlined competitive intelligence analysis.
"""
import sys
import asyncio
from datetime import datetime
from langgraph_agent import LangGraphCoordinatorAgent


def print_header():
    """Print application header."""
    print("\n🤖 Professional Multi-Agent Intelligence System")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Powered by LangGraph ReAct Coordinator Agent")


async def main():
    """Main application entry point."""
    print_header()
    
    # Initialize LangGraph coordinator
    print("\n🔄 Initializing LangGraph Coordinator Agent...")
    coordinator = LangGraphCoordinatorAgent()
    
    # Example queries
    queries = [
        "ChatGPT latest features and updates",
        "Tesla autonomous driving technology",
        "Notion AI productivity tools"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📊 Analysis {i}/{len(queries)}: {query}")
        print("-" * 50)
        
        try:
            async for update in coordinator.get_clean_analysis(query):
                print(update, end='')
                
        except Exception as e:
            print(f"❌ Error analyzing '{query}': {e}")
    
    print(f"\n✅ Analysis completed at {datetime.now().strftime('%H:%M:%S')}")


def interactive_mode():
    """Interactive mode for manual queries."""
    print_header()
    
    coordinator = LangGraphCoordinatorAgent()
    print("\n🎯 Interactive Mode - Enter queries (type 'quit' to exit)")
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("🔄 Analyzing...")
            
            async def run_analysis():
                async for update in coordinator.get_clean_analysis(query):
                    print(update, end='')
            
            asyncio.run(run_analysis())
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        asyncio.run(main())

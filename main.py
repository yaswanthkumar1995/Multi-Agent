"""
Professional Multi-Agent Intelligence System
Main entry point for the streamlined competitive intelligence analysis.
Updated to use LangChain AgentExecutor implementation.
"""
import sys
from datetime import datetime
from langchain_agent import LangChainMultiAgent


def print_header():
    """Print application header."""
    print("\n🤖 Professional Multi-Agent Intelligence System")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Powered by LangChain AgentExecutor with React Prompting")


def main():
    """Main application entry point."""
    print_header()
    
    # Initialize LangChain multi-agent system
    print("\n🔄 Initializing LangChain Multi-Agent System...")
    coordinator = LangChainMultiAgent()
    
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
            response = coordinator.chat(query)
            print(response)
                
        except Exception as e:
            print(f"❌ Error analyzing '{query}': {e}")
    
    print(f"\n✅ Analysis completed at {datetime.now().strftime('%H:%M:%S')}")


def interactive_mode():
    """Interactive mode for manual queries."""
    print_header()
    
    coordinator = LangChainMultiAgent()
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
            
            response = coordinator.chat(query)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()

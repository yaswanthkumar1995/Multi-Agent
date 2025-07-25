"""
Streamlit Chatbot Interface for the multi-agent intelligence system.
Updated to use LangChain AgentExecutor implementation.
"""
import streamlit as st
import json
import time
import asyncio
from datetime import datetime
from langchain_agent import LangChainMultiAgent

# Configure page
st.set_page_config(
    page_title="Multi-Agent Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chatbot
if 'coordinator' not in st.session_state:
    st.session_state.coordinator = LangChainMultiAgent()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def main():
    st.title("🤖 Multi-Agent Intelligence Chatbot")
    st.markdown("*Ask me anything about competitive intelligence, products, or market trends!*")
    
    # Add confirmation for clear chat
    if 'show_clear_confirm' not in st.session_state:
        st.session_state.show_clear_confirm = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        
        # Chat management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # New conversation button
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.thread_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.success("New chat started!")
                st.rerun()
        
        with col2:
            # Clear conversation button
            if st.button("🗑️ Clear Chat", use_container_width=True):
                if len(st.session_state.messages) > 0:
                    st.session_state.show_clear_confirm = True
                else:
                    st.info("Chat is already empty!")
        
        # Clear confirmation dialog
        if st.session_state.show_clear_confirm:
            st.warning("⚠️ Are you sure you want to clear all messages?")
            conf_col1, conf_col2 = st.columns(2)
            
            with conf_col1:
                if st.button("✅ Yes, Clear", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.show_clear_confirm = False
                    st.success("Chat cleared!")
                    st.rerun()
            
            with conf_col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_clear_confirm = False
                    st.rerun()
        
        st.divider()
        
        # Chat info
        st.subheader("📊 Chat Info")
        st.write(f"**Thread ID:** `{st.session_state.thread_id}`")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        st.divider()
        
        # Export chat button
        if st.button("💾 Export Chat", use_container_width=True):
            if len(st.session_state.messages) > 0:
                chat_data = {
                    "thread_id": st.session_state.thread_id,
                    "messages": st.session_state.messages,
                    "export_timestamp": datetime.now().isoformat(),
                    "total_messages": len(st.session_state.messages)
                }
                st.download_button(
                    "📥 Download Chat",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"chat_{st.session_state.thread_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.warning("No messages to export!")
        
        st.divider()
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display welcome message if no chat history
        if len(st.session_state.messages) == 0:
            st.info("👋 Welcome! Start a conversation by typing a message below.")
        
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(message['content'])
                
                # Show timestamp for recent messages
                if "timestamp" in message and i >= len(st.session_state.messages) - 5:
                    timestamp = message["timestamp"]
                    if isinstance(timestamp, datetime):
                        time_str = timestamp.strftime("%H:%M:%S")
                    else:
                        time_str = str(timestamp)
                    st.caption(f"🕐 {time_str}")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            try:
                # Get clean JSON response
                with st.spinner("Analyzing..."):
                    response = st.session_state.coordinator.chat(prompt)
                
                # Parse and format JSON response for better readability
                try:
                    # Try to parse JSON and format it nicely
                    json_data = json.loads(response)
                    
                    # Check if this is a greeting/casual response or personal introduction from ReAct Agent
                    if (json_data.get('product') == 'ReAct Agent' and 
                        json_data.get('source') == 'LangGraph ReAct Coordinator' and
                        ('👋 Hello!' in json_data.get('update', '') or 
                         'Nice to meet you' in json_data.get('update', ''))):
                        # Display as direct AI message for greetings and introductions
                        ai_message = json_data.get('update', 'Hello!')
                        response_container.markdown(ai_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ai_message
                        })
                    else:
                        # Display structured format for actual product updates
                        formatted_response = f"""**Product:** {json_data.get('product', 'N/A')}

**Update:** {json_data.get('update', 'N/A')}

**Source:** {json_data.get('source', 'N/A')}

**Date:** {json_data.get('date', 'N/A')}"""
                        response_container.markdown(formatted_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": formatted_response
                        })
                except json.JSONDecodeError:
                    # If not JSON, display as-is
                    response_container.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                response_container.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })


if __name__ == "__main__":
    main()

"""
A2A (Agent-to-Agent) Protocol Integration for multi-agent communication.
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class MessageType(str, Enum):
    """A2A Message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class AgentRole(str, Enum):
    """Agent roles in A2A protocol."""
    COORDINATOR = "coordinator"
    SEARCH = "search"
    SUMMARIZER = "summarizer"
    VERIFIER = "verifier"
    ANALYZER = "analyzer"


class A2AMessage(BaseModel):
    """A2A Protocol message structure."""
    id: str
    type: MessageType
    sender: AgentRole
    receiver: AgentRole
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    priority: int = 0
    ttl: int = 300  # Time to live in seconds


class A2AProtocol:
    """A2A Protocol implementation for agent communication."""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
        self.subscriptions = {}
        self.running = False
    
    def register_agent(self, role: AgentRole, handler: callable):
        """Register an agent with the protocol."""
        self.agents[role] = handler
        self.subscriptions[role] = []
        print(f"✅ Agent {role.value} registered with A2A protocol")
    
    def subscribe(self, subscriber: AgentRole, message_type: MessageType):
        """Subscribe an agent to specific message types."""
        if subscriber not in self.subscriptions:
            self.subscriptions[subscriber] = []
        self.subscriptions[subscriber].append(message_type)
        print(f"📬 Agent {subscriber.value} subscribed to {message_type.value} messages")
    
    async def send_message(self, message: A2AMessage) -> bool:
        """Send a message through the A2A protocol."""
        try:
            await self.message_queue.put(message)
            print(f"📤 Message sent: {message.sender.value} → {message.receiver.value} ({message.type.value})")
            return True
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            return False
    
    async def broadcast_message(self, sender: AgentRole, message_type: MessageType, payload: Dict[str, Any]) -> int:
        """Broadcast a message to all subscribed agents."""
        sent_count = 0
        message_id = f"broadcast_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        for agent_role, subscribed_types in self.subscriptions.items():
            if agent_role != sender and message_type in subscribed_types:
                message = A2AMessage(
                    id=f"{message_id}_{agent_role.value}",
                    type=message_type,
                    sender=sender,
                    receiver=agent_role,
                    payload=payload,
                    timestamp=datetime.now().isoformat()
                )
                
                if await self.send_message(message):
                    sent_count += 1
        
        print(f"📡 Broadcast sent to {sent_count} agents")
        return sent_count
    
    async def process_messages(self):
        """Process messages from the queue."""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Error processing message: {e}")
    
    async def _handle_message(self, message: A2AMessage):
        """Handle a received message."""
        receiver = message.receiver
        
        if receiver in self.agents:
            try:
                handler = self.agents[receiver]
                await handler(message)
                print(f"📥 Message processed: {message.sender.value} → {receiver.value}")
            except Exception as e:
                # Send error response
                error_message = A2AMessage(
                    id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    type=MessageType.ERROR,
                    sender=receiver,
                    receiver=message.sender,
                    payload={"error": str(e), "original_message_id": message.id},
                    timestamp=datetime.now().isoformat(),
                    correlation_id=message.id
                )
                await self.send_message(error_message)
                print(f"❌ Error handling message: {e}")
        else:
            print(f"⚠️ No handler for agent: {receiver.value}")
    
    async def start(self):
        """Start the A2A protocol."""
        self.running = True
        print("🚀 A2A Protocol started")
        await self.process_messages()
    
    def stop(self):
        """Stop the A2A protocol."""
        self.running = False
        print("🛑 A2A Protocol stopped")


class A2AMultiAgent:
    """Multi-agent system with A2A protocol integration."""
    
    def __init__(self):
        self.protocol = A2AProtocol()
        self.analysis_results = {}
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup agents with A2A protocol handlers."""
        
        # Register agent handlers
        self.protocol.register_agent(AgentRole.COORDINATOR, self._coordinator_handler)
        self.protocol.register_agent(AgentRole.SEARCH, self._search_handler)
        self.protocol.register_agent(AgentRole.SUMMARIZER, self._summarizer_handler)
        self.protocol.register_agent(AgentRole.VERIFIER, self._verifier_handler)
        self.protocol.register_agent(AgentRole.ANALYZER, self._analyzer_handler)
        
        # Setup subscriptions
        self.protocol.subscribe(AgentRole.SEARCH, MessageType.REQUEST)
        self.protocol.subscribe(AgentRole.SUMMARIZER, MessageType.REQUEST)
        self.protocol.subscribe(AgentRole.VERIFIER, MessageType.REQUEST)
        self.protocol.subscribe(AgentRole.ANALYZER, MessageType.REQUEST)
        self.protocol.subscribe(AgentRole.COORDINATOR, MessageType.RESPONSE)
        self.protocol.subscribe(AgentRole.COORDINATOR, MessageType.ERROR)
    
    async def _coordinator_handler(self, message: A2AMessage):
        """Handle coordinator messages."""
        if message.type == MessageType.REQUEST:
            query = message.payload.get("query", "")
            category = message.payload.get("category", "AI Tools")
            
            # Start analysis pipeline
            await self._start_analysis_pipeline(query, category, message.id)
        
        elif message.type == MessageType.RESPONSE:
            # Aggregate responses
            await self._aggregate_response(message)
    
    async def _search_handler(self, message: A2AMessage):
        """Handle search agent messages."""
        if message.type == MessageType.REQUEST:
            query = message.payload.get("query", "")
            
            try:
                # Use SearchAgent directly
                from search_agent import SearchAgent
                search_agent = SearchAgent()
                search_results = search_agent.search(query)
                
                response = A2AMessage(
                    id=f"search_response_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    type=MessageType.RESPONSE,
                    sender=AgentRole.SEARCH,
                    receiver=AgentRole.COORDINATOR,
                    payload={"results": [result.dict() for result in search_results]},
                    timestamp=datetime.now().isoformat(),
                    correlation_id=message.id
                )
                
                await self.protocol.send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, str(e), AgentRole.SEARCH)
    
    async def _summarizer_handler(self, message: A2AMessage):
        """Handle summarizer agent messages."""
        if message.type == MessageType.REQUEST:
            search_results = message.payload.get("search_results", [])
            
            try:
                # Convert to SearchResult objects
                from models import SearchResult
                from summarizer_agent import SummarizerAgent
                
                results = [SearchResult(**result) for result in search_results]
                summarizer_agent = SummarizerAgent()
                updates = summarizer_agent.summarize(results)
                
                response = A2AMessage(
                    id=f"summarizer_response_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    type=MessageType.RESPONSE,
                    sender=AgentRole.SUMMARIZER,
                    receiver=AgentRole.COORDINATOR,
                    payload={"updates": [update.dict() for update in updates]},
                    timestamp=datetime.now().isoformat(),
                    correlation_id=message.id
                )
                
                await self.protocol.send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, str(e), AgentRole.SUMMARIZER)
    
    async def _verifier_handler(self, message: A2AMessage):
        """Handle verifier agent messages."""
        if message.type == MessageType.REQUEST:
            updates = message.payload.get("updates", [])
            
            try:
                # Convert to ProductUpdate objects
                from models import ProductUpdate
                from verifier_agent import VerifierAgent
                
                product_updates = [ProductUpdate(**update) for update in updates]
                verifier_agent = VerifierAgent()
                verified_updates = verifier_agent.verify(product_updates)
                
                response = A2AMessage(
                    id=f"verifier_response_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    type=MessageType.RESPONSE,
                    sender=AgentRole.VERIFIER,
                    receiver=AgentRole.COORDINATOR,
                    payload={"verified_updates": [update.dict() for update in verified_updates]},
                    timestamp=datetime.now().isoformat(),
                    correlation_id=message.id
                )
                
                await self.protocol.send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, str(e), AgentRole.VERIFIER)
    
    async def _analyzer_handler(self, message: A2AMessage):
        """Handle analyzer agent messages."""
        if message.type == MessageType.REQUEST:
            verified_updates = message.payload.get("verified_updates", [])
            query = message.payload.get("query", "")
            category = message.payload.get("category", "")
            
            try:
                # Create final analysis report
                from models import IntelligenceReport, ProductUpdate
                updates = [ProductUpdate(**update) for update in verified_updates]
                
                report = IntelligenceReport(
                    query=query,
                    category=category,
                    updates=updates,
                    total_sources=len(updates),
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat()
                )
                
                response = A2AMessage(
                    id=f"analyzer_response_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    type=MessageType.RESPONSE,
                    sender=AgentRole.ANALYZER,
                    receiver=AgentRole.COORDINATOR,
                    payload={"report": report.dict()},
                    timestamp=datetime.now().isoformat(),
                    correlation_id=message.id
                )
                
                await self.protocol.send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, str(e), AgentRole.ANALYZER)
    
    async def _send_error_response(self, original_message: A2AMessage, error: str, sender: AgentRole):
        """Send error response."""
        error_message = A2AMessage(
            id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            type=MessageType.ERROR,
            sender=sender,
            receiver=original_message.sender,
            payload={"error": error, "original_message_id": original_message.id},
            timestamp=datetime.now().isoformat(),
            correlation_id=original_message.id
        )
        
        await self.protocol.send_message(error_message)
    
    async def _start_analysis_pipeline(self, query: str, category: str, correlation_id: str):
        """Start the analysis pipeline using A2A messages."""
        # Step 1: Send search request
        search_message = A2AMessage(
            id=f"search_request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            type=MessageType.REQUEST,
            sender=AgentRole.COORDINATOR,
            receiver=AgentRole.SEARCH,
            payload={"query": query, "category": category},
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id
        )
        
        await self.protocol.send_message(search_message)
    
    async def _aggregate_response(self, message: A2AMessage):
        """Aggregate responses from agents."""
        correlation_id = message.correlation_id
        
        if correlation_id not in self.analysis_results:
            self.analysis_results[correlation_id] = {}
        
        # Store response
        self.analysis_results[correlation_id][message.sender.value] = message.payload
        
        # Check if we have search results to proceed with summarization
        if message.sender == AgentRole.SEARCH:
            search_results = message.payload.get("results", [])
            
            summarizer_message = A2AMessage(
                id=f"summarizer_request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                type=MessageType.REQUEST,
                sender=AgentRole.COORDINATOR,
                receiver=AgentRole.SUMMARIZER,
                payload={"search_results": search_results},
                timestamp=datetime.now().isoformat(),
                correlation_id=correlation_id
            )
            
            await self.protocol.send_message(summarizer_message)
        
        # Check if we have summaries to proceed with verification
        elif message.sender == AgentRole.SUMMARIZER:
            updates = message.payload.get("updates", [])
            
            verifier_message = A2AMessage(
                id=f"verifier_request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                type=MessageType.REQUEST,
                sender=AgentRole.COORDINATOR,
                receiver=AgentRole.VERIFIER,
                payload={"updates": updates},
                timestamp=datetime.now().isoformat(),
                correlation_id=correlation_id
            )
            
            await self.protocol.send_message(verifier_message)
        
        # Check if we have verified results to proceed with analysis
        elif message.sender == AgentRole.VERIFIER:
            verified_updates = message.payload.get("verified_updates", [])
            
            analyzer_message = A2AMessage(
                id=f"analyzer_request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                type=MessageType.REQUEST,
                sender=AgentRole.COORDINATOR,
                receiver=AgentRole.ANALYZER,
                payload={
                    "verified_updates": verified_updates,
                    "query": "analysis_query",  # You might want to store this in analysis_results
                    "category": "AI Tools"
                },
                timestamp=datetime.now().isoformat(),
                correlation_id=correlation_id
            )
            
            await self.protocol.send_message(analyzer_message)
    
    async def analyze_with_a2a(self, query: str, category: str = "AI Tools") -> dict:
        """Perform analysis using A2A protocol."""
        correlation_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Start protocol in background
        protocol_task = asyncio.create_task(self.protocol.start())
        
        # Send initial request
        initial_message = A2AMessage(
            id=f"initial_request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            type=MessageType.REQUEST,
            sender=AgentRole.COORDINATOR,
            receiver=AgentRole.COORDINATOR,
            payload={"query": query, "category": category},
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id
        )
        
        await self.protocol.send_message(initial_message)
        
        # Wait for completion (simplified)
        await asyncio.sleep(5)
        
        # Stop protocol
        self.protocol.stop()
        protocol_task.cancel()
        
        # Return results
        return self.analysis_results.get(correlation_id, {})
    
    async def send_heartbeat(self):
        """Send heartbeat messages to check agent status."""
        heartbeat_payload = {"status": "alive", "timestamp": datetime.now().isoformat()}
        
        await self.protocol.broadcast_message(
            AgentRole.COORDINATOR,
            MessageType.HEARTBEAT,
            heartbeat_payload
        )
    
    def get_protocol_stats(self) -> dict:
        """Get A2A protocol statistics."""
        return {
            "registered_agents": len(self.protocol.agents),
            "subscriptions": {role.value: types for role, types in self.protocol.subscriptions.items()},
            "queue_size": self.protocol.message_queue.qsize(),
            "running": self.protocol.running
        }

"""
Logging system for multi-agent interactions
Provides traceability and memory capabilities
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path


class AgentLogger:
    """Centralized logging for all agent interactions"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        log_file = self.log_dir / f"agent_interactions_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("MultiAgent")
        
        # Memory storage
        self.memory_file = self.log_dir / "agent_memory.json"
        self.memory = self._load_memory()
        
    def _load_memory(self) -> Dict[str, Any]:
        """Load short-term memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"queries": [], "responses": {}}
    
    def _save_memory(self):
        """Save memory to file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2, default=str)
    
    def log_agent_interaction(self, agent_name: str, action: str, input_data: Any, output_data: Any):
        """Log agent interaction with full traceability"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "action": action,
            "input": str(input_data)[:500],  # Truncate long inputs
            "output": str(output_data)[:500],  # Truncate long outputs
        }
        
        self.logger.info(f"Agent: {agent_name} | Action: {action} | Input: {str(input_data)[:100]}...")
        
        # Store in memory
        self.memory["queries"].append(interaction)
        
        # Keep only last 100 interactions
        if len(self.memory["queries"]) > 100:
            self.memory["queries"] = self.memory["queries"][-100:]
        
        self._save_memory()
    
    def check_recent_query(self, query: str, hours: int = 1) -> bool:
        """Check if similar query was made recently to avoid repeating"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        for interaction in self.memory["queries"]:
            try:
                interaction_time = datetime.fromisoformat(interaction["timestamp"]).timestamp()
                if interaction_time > cutoff_time:
                    # Simple similarity check
                    if query.lower() in interaction["input"].lower() or interaction["input"].lower() in query.lower():
                        self.logger.info(f"Similar query found recently: {interaction['input'][:100]}...")
                        return True
            except:
                continue
        
        return False
    
    def get_cached_response(self, query: str) -> str:
        """Get cached response for repeated query"""
        for interaction in reversed(self.memory["queries"]):
            if query.lower() in interaction["input"].lower():
                # Only return if the output is meaningful (not just "processing")
                output = interaction.get('output', '')
                if output and output != 'processing' and len(output) > 10:
                    return output
        return None
    
    def log_error(self, agent_name: str, error: str, context: str = ""):
        """Log errors with context"""
        self.logger.error(f"Agent: {agent_name} | Error: {error} | Context: {context}")
    
    def get_interaction_summary(self, limit: int = 10) -> List[Dict]:
        """Get recent interaction summary"""
        return self.memory["queries"][-limit:]


# Global logger instance
agent_logger = AgentLogger()

"""
AI Agents implementation showcasing different types of autonomous agents.

This module demonstrates various AI agent patterns including:
- ReAct (Reasoning and Acting) agents
- Tool-using agents  
- Multi-agent systems
- Planning agents
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States that an agent can be in during execution."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    action_type: str
    parameters: Dict[str, Any]
    reasoning: Optional[str] = None
    confidence: float = 1.0


@dataclass
class AgentObservation:
    """Represents an observation received by an agent."""
    content: str
    observation_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """Abstract base class for tools that agents can use."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for this tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool."""
        pass


class CalculatorTool(Tool):
    """A simple calculator tool for mathematical operations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic mathematical calculations"
        )
    
    def execute(self, expression: str) -> str:
        """Execute a mathematical expression safely."""
        try:
            # Basic security: only allow numbers, operators, and parentheses
            if not re.match(r'^[0-9+\-*/.() ]+$', expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }


class SearchTool(Tool):
    """A mock search tool that simulates web search."""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for information on the internet"
        )
        # Mock search results database
        self.search_db = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
            "neural networks": "Neural networks are computing systems inspired by biological neural networks."
        }
    
    def execute(self, query: str) -> str:
        """Execute a search query."""
        query_lower = query.lower()
        for key, value in self.search_db.items():
            if key in query_lower:
                return f"Search result: {value}"
        return f"No specific results found for '{query}'. Try searching for: python, ai, machine learning, or neural networks."
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                }
            },
            "required": ["query"]
        }


class MemoryTool(Tool):
    """A tool for storing and retrieving memories."""
    
    def __init__(self):
        super().__init__(
            name="memory",
            description="Store and retrieve information from memory"
        )
        self.memories: Dict[str, str] = {}
    
    def execute(self, action: str, key: str = "", value: str = "") -> str:
        """Execute memory operations."""
        if action == "store":
            self.memories[key] = value
            return f"Stored memory: {key} = {value}"
        elif action == "retrieve":
            if key in self.memories:
                return f"Retrieved memory: {key} = {self.memories[key]}"
            else:
                return f"No memory found for key: {key}"
        elif action == "list":
            if self.memories:
                items = [f"{k}: {v}" for k, v in self.memories.items()]
                return f"Memories: {', '.join(items)}"
            else:
                return "No memories stored"
        else:
            return f"Invalid action: {action}. Use 'store', 'retrieve', or 'list'"
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "retrieve", "list"],
                    "description": "Memory operation to perform"
                },
                "key": {
                    "type": "string",
                    "description": "Key for memory storage/retrieval"
                },
                "value": {
                    "type": "string",
                    "description": "Value to store (only for 'store' action)"
                }
            },
            "required": ["action"]
        }


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, description: str, max_iterations: int = 10):
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self.state = AgentState.IDLE
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools[tool.name] = tool
        logger.info(f"Added tool '{tool.name}' to agent '{self.name}'")
    
    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent's toolkit."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Removed tool '{tool_name}' from agent '{self.name}'")
    
    @abstractmethod
    def process_input(self, input_text: str) -> str:
        """Process input and return a response."""
        pass
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.state = AgentState.IDLE
        self.conversation_history.clear()
        logger.info(f"Agent '{self.name}' has been reset")


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning and Acting) Agent implementation.
    
    This agent follows the ReAct pattern:
    1. Thought: Reason about the current situation
    2. Action: Take an action using available tools
    3. Observation: Observe the result of the action
    4. Repeat until the task is complete
    """
    
    def __init__(self, name: str = "ReAct Agent", max_iterations: int = 10):
        super().__init__(
            name=name,
            description="A ReAct agent that can reason and act using available tools",
            max_iterations=max_iterations
        )
        self.current_task = ""
        self.thoughts: List[str] = []
    
    def process_input(self, input_text: str) -> str:
        """Process input using the ReAct pattern."""
        self.current_task = input_text
        self.state = AgentState.THINKING
        
        logger.info(f"Starting ReAct process for task: {input_text}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": input_text})
        
        result = self._react_loop()
        
        # Add result to conversation history
        self.conversation_history.append({"role": "assistant", "content": result})
        
        return result
    
    def _react_loop(self) -> str:
        """Execute the main ReAct loop."""
        for iteration in range(self.max_iterations):
            logger.info(f"ReAct iteration {iteration + 1}")
            
            # Thought phase
            thought = self._think()
            self.thoughts.append(thought)
            logger.info(f"Thought: {thought}")
            
            # Check if we think we're done
            if "FINAL ANSWER:" in thought:
                final_answer = thought.split("FINAL ANSWER:")[-1].strip()
                self.state = AgentState.DONE
                return final_answer
            
            # Action phase
            if self.tools:
                action = self._choose_action(thought)
                if action:
                    logger.info(f"Action: {action.action_type} with params {action.parameters}")
                    
                    # Execute action
                    self.state = AgentState.ACTING
                    observation = self._execute_action(action)
                    
                    # Observe result
                    self.state = AgentState.OBSERVING
                    logger.info(f"Observation: {observation.content}")
                    
                    # Continue to next iteration with the observation
                    continue
            
            # If no action was taken, provide a direct response
            self.state = AgentState.DONE
            return self._generate_direct_response()
        
        self.state = AgentState.ERROR
        return "I couldn't complete the task within the maximum number of iterations."
    
    def _think(self) -> str:
        """Generate a thought based on the current context."""
        if not self.thoughts:
            # First thought
            if self.tools:
                available_tools = ", ".join(self.tools.keys())
                return f"I need to help with: '{self.current_task}'. I have these tools available: {available_tools}. Let me think about what I need to do."
            else:
                return f"I need to help with: '{self.current_task}'. I don't have any tools available, so I'll provide a direct response."
        
        # Subsequent thoughts - analyze what we've learned so far
        # Simple heuristics for thinking
        if "calculator" in self.tools and any(op in self.current_task for op in ['+', '-', '*', '/', 'calculate', 'compute']):
            return "This seems like a mathematical problem. I should use the calculator tool."
        elif "search" in self.tools and any(keyword in self.current_task.lower() for keyword in ['what is', 'tell me about', 'search', 'find']):
            return "This requires searching for information. I should use the search tool."
        elif "memory" in self.tools and any(keyword in self.current_task.lower() for keyword in ['remember', 'store', 'save']):
            return "This involves memory operations. I should use the memory tool."
        else:
            return f"FINAL ANSWER: Based on my analysis, here's my response to '{self.current_task}': I understand your request but may need more specific information to provide the best help."
    
    def _choose_action(self, thought: str) -> Optional[AgentAction]:
        """Choose an action based on the current thought."""
        if "calculator" in thought.lower() and "calculator" in self.tools:
            # Extract mathematical expression from the task
            math_expressions = re.findall(r'[\d+\-*/.() ]+', self.current_task)
            if math_expressions:
                expr = max(math_expressions, key=len).strip()
                return AgentAction(
                    action_type="calculator",
                    parameters={"expression": expr},
                    reasoning=f"Using calculator to compute: {expr}"
                )
        
        elif "search" in thought.lower() and "search" in self.tools:
            # Extract search query
            query = self.current_task
            # Clean up the query
            for prefix in ["what is", "tell me about", "search for"]:
                if query.lower().startswith(prefix):
                    query = query[len(prefix):].strip()
            
            return AgentAction(
                action_type="search",
                parameters={"query": query},
                reasoning=f"Searching for information about: {query}"
            )
        
        elif "memory" in thought.lower() and "memory" in self.tools:
            # Simple memory operation detection
            if "remember" in self.current_task.lower() or "store" in self.current_task.lower():
                return AgentAction(
                    action_type="memory",
                    parameters={"action": "store", "key": "user_request", "value": self.current_task},
                    reasoning="Storing information in memory"
                )
            else:
                return AgentAction(
                    action_type="memory",
                    parameters={"action": "list"},
                    reasoning="Checking what's in memory"
                )
        
        return None
    
    def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute an action and return the observation."""
        if action.action_type in self.tools:
            tool = self.tools[action.action_type]
            try:
                result = tool.execute(**action.parameters)
                return AgentObservation(
                    content=result,
                    observation_type="tool_result",
                    metadata={"tool": action.action_type, "action": action}
                )
            except Exception as e:
                return AgentObservation(
                    content=f"Error executing {action.action_type}: {str(e)}",
                    observation_type="error"
                )
        else:
            return AgentObservation(
                content=f"Tool '{action.action_type}' not found",
                observation_type="error"
            )
    
    def _generate_direct_response(self) -> str:
        """Generate a direct response when no tools are used."""
        return f"I understand you want help with: '{self.current_task}'. While I don't have specific tools to handle this request, I'm here to help however I can. Could you provide more specific details about what you need?"


class MultiAgentSystem:
    """
    A system for coordinating multiple agents.
    
    This demonstrates how multiple specialized agents can work together
    to solve complex tasks.
    """
    
    def __init__(self, name: str = "Multi-Agent System"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.conversation_log: List[Dict[str, Any]] = []
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the system."""
        self.agents[agent.name] = agent
        logger.info(f"Added agent '{agent.name}' to multi-agent system")
    
    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the system."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Removed agent '{agent_name}' from multi-agent system")
    
    def process_task(self, task: str, agent_name: Optional[str] = None) -> str:
        """
        Process a task, either with a specific agent or by choosing the best one.
        """
        if not self.agents:
            return "No agents available in the system."
        
        # Log the task
        self.conversation_log.append({
            "type": "task",
            "content": task,
            "timestamp": "now"  # In real implementation, use datetime
        })
        
        if agent_name and agent_name in self.agents:
            # Use specific agent
            selected_agent = self.agents[agent_name]
        else:
            # Choose best agent for the task
            selected_agent = self._choose_best_agent(task)
        
        logger.info(f"Selected agent '{selected_agent.name}' for task: {task}")
        
        # Process with selected agent
        result = selected_agent.process_input(task)
        
        # Log the result
        self.conversation_log.append({
            "type": "response",
            "agent": selected_agent.name,
            "content": result,
            "timestamp": "now"
        })
        
        return f"[{selected_agent.name}]: {result}"
    
    def _choose_best_agent(self, task: str) -> BaseAgent:
        """Choose the best agent for a given task based on simple heuristics."""
        task_lower = task.lower()
        
        # Simple agent selection based on keywords
        for agent_name, agent in self.agents.items():
            if isinstance(agent, ReActAgent):
                # ReAct agents are good for complex reasoning tasks
                if any(keyword in task_lower for keyword in ['calculate', 'search', 'find', 'what', 'how']):
                    return agent
        
        # Default to first available agent
        return next(iter(self.agents.values()))
    
    def get_conversation_log(self) -> List[Dict[str, Any]]:
        """Get the full conversation log."""
        return self.conversation_log.copy()
    
    def reset_all_agents(self) -> None:
        """Reset all agents in the system."""
        for agent in self.agents.values():
            agent.reset()
        self.conversation_log.clear()
        logger.info("Reset all agents in the multi-agent system")


def create_example_agent_system() -> MultiAgentSystem:
    """
    Create an example multi-agent system with pre-configured agents and tools.
    
    Returns:
        MultiAgentSystem: A configured multi-agent system ready for use
    """
    # Create multi-agent system
    system = MultiAgentSystem("Example AI Agent System")
    
    # Create a ReAct agent with various tools
    react_agent = ReActAgent("Assistant", max_iterations=5)
    
    # Add tools to the agent
    react_agent.add_tool(CalculatorTool())
    react_agent.add_tool(SearchTool())
    react_agent.add_tool(MemoryTool())
    
    # Add the agent to the system
    system.add_agent(react_agent)
    
    logger.info("Created example agent system with ReAct agent and tools")
    return system


if __name__ == "__main__":
    # Example usage
    print("Creating example AI agent system...")
    agent_system = create_example_agent_system()
    
    # Test the system with various tasks
    test_tasks = [
        "What is 25 * 4 + 10?",
        "Tell me about machine learning",
        "Remember that I like Python programming",
        "What's 100 divided by 5?",
        "Search for information about neural networks"
    ]
    
    print("\nTesting AI agent system:")
    print("=" * 50)
    
    for task in test_tasks:
        print(f"\nTask: {task}")
        response = agent_system.process_task(task)
        print(f"Response: {response}")
        print("-" * 30)

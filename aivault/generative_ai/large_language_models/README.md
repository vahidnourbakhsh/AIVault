# AI Agents Project

This project demonstrates how AI agents can be leveraged to solve complex tasks through autonomous reasoning and tool usage. It showcases different types of agents, tool integration, and multi-agent systems.

## Overview

AI agents are autonomous systems that can perceive their environment, reason about it, and take actions to achieve specific goals. This implementation provides a flexible framework for building and deploying various types of agents.

## Features

### 🤖 Agent Types
- **ReAct Agents**: Implement the Reasoning and Acting pattern for systematic problem-solving
- **Multi-Agent Systems**: Coordinate multiple specialized agents for complex tasks
- **Base Agent Framework**: Extensible architecture for creating custom agent types

### 🛠️ Built-in Tools
- **Calculator Tool**: Perform mathematical calculations
- **Search Tool**: Retrieve information from a knowledge base
- **Memory Tool**: Store and retrieve information across interactions
- **Custom Tool Support**: Easy framework for adding new capabilities

### 🔧 Key Capabilities
- **Tool Integration**: Agents can use external tools to extend their capabilities
- **Conversation Memory**: Maintain context across multiple interactions
- **State Management**: Track agent states during execution
- **Error Handling**: Robust error handling and recovery
- **Logging**: Comprehensive logging for debugging and analysis

## Quick Start

```python
from aivault.generative_ai.large_language_models.ai_agents import (
    create_example_agent_system
)

# Create a pre-configured agent system
agent_system = create_example_agent_system()

# Use the agent system
response = agent_system.process_task("What is 25 * 4 + 10?")
print(response)
# Output: [Assistant]: Result: 110
```

## Architecture

### Agent Components

```
┌─────────────────┐
│   BaseAgent     │
├─────────────────┤
│ - State         │
│ - Tools         │
│ - History       │
│ - Memory        │
└─────────────────┘
         ▲
         │
┌─────────────────┐
│   ReActAgent    │
├─────────────────┤
│ - Think         │
│ - Act           │
│ - Observe       │
│ - Iterate       │
└─────────────────┘
```

### Tool System

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Calculator  │    │  SearchTool  │    │ MemoryTool  │
│    Tool     │    │              │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
      │                     │                   │
      └─────────────────────┼───────────────────┘
                            │
                    ┌───────────────┐
                    │  Agent Tools  │
                    │   Registry    │
                    └───────────────┘
```

## Usage Examples

### Creating a Simple Agent

```python
from aivault.generative_ai.large_language_models.ai_agents import (
    ReActAgent, CalculatorTool, SearchTool
)

# Create an agent
agent = ReActAgent("MathBot")

# Add tools
agent.add_tool(CalculatorTool())
agent.add_tool(SearchTool())

# Process tasks
response = agent.process_input("What is the square of 15?")
print(response)
```

### Custom Tool Creation

```python
from aivault.generative_ai.large_language_models.ai_agents import Tool

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get weather information"
        )
    
    def execute(self, city: str) -> str:
        # Your weather API integration here
        return f"Weather in {city}: Sunny, 22°C"
    
    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }

# Use the custom tool
agent.add_tool(WeatherTool())
```

### Multi-Agent System

```python
from aivault.generative_ai.large_language_models.ai_agents import (
    MultiAgentSystem, ReActAgent
)

# Create system
system = MultiAgentSystem("TaskForce")

# Add specialized agents
math_agent = ReActAgent("MathExpert")
math_agent.add_tool(CalculatorTool())

search_agent = ReActAgent("Researcher")
search_agent.add_tool(SearchTool())

system.add_agent(math_agent)
system.add_agent(search_agent)

# The system will choose the best agent for each task
response = system.process_task("Calculate 42 * 7")
```

## ReAct Pattern

The ReAct (Reasoning and Acting) pattern is implemented as follows:

1. **Thought**: Agent analyzes the current situation and plans next steps
2. **Action**: Agent executes an action using available tools
3. **Observation**: Agent processes the results of the action
4. **Repeat**: Continue until the task is complete or max iterations reached

### Example ReAct Cycle

```
Task: "What is 15 * 8 + 25?"

Iteration 1:
  Thought: "This is a math problem, I should use the calculator"
  Action: calculator(expression="15 * 8 + 25")
  Observation: "Result: 145"
  
Final Answer: 145
```

## File Structure

```
aivault/generative_ai/large_language_models/
├── ai_agents.py              # Main agents implementation
├── __init__.py              # Module initialization
└── README.md               # This file

examples/
└── ai_agents_showcase.ipynb # Interactive demonstration

tests/
└── test_ai_agents.py       # Comprehensive test suite
```

## Testing

Run the test suite to verify functionality:

```python
# Run from the tests directory
python test_ai_agents.py

# Or use pytest if available
pytest test_ai_agents.py -v
```

## Advanced Features

### State Management

Agents maintain state throughout their execution:
- `IDLE`: Ready for new tasks
- `THINKING`: Analyzing the problem
- `ACTING`: Executing actions
- `OBSERVING`: Processing results
- `DONE`: Task completed
- `ERROR`: Error occurred

### Memory Persistence

Agents can store and retrieve information:

```python
agent.process_input("Remember that I prefer Python programming")
# Later...
agent.process_input("What do you remember about my preferences?")
```

### Conversation Context

All interactions are stored for context-aware responses:

```python
# Access conversation history
history = agent.conversation_history
for entry in history:
    print(f"{entry['role']}: {entry['content']}")
```

## Real-World Applications

This framework can be extended for various applications:

- **Customer Service**: Automated support with tool integration
- **Data Analysis**: Agents that can query databases and generate reports  
- **Content Creation**: Multi-agent systems for research and writing
- **Task Automation**: Workflow automation with intelligent decision-making
- **Educational Assistants**: Personalized tutoring with adaptive responses

## Limitations and Considerations

- **Tool Dependency**: Agents are limited by available tools
- **Reasoning Depth**: Simple heuristics, not deep reasoning
- **Error Handling**: Basic error recovery mechanisms
- **Scalability**: Consider performance with many agents/tools
- **Security**: Validate tool inputs and outputs

## Future Enhancements

- Integration with real LLMs (OpenAI, Anthropic, local models)
- Advanced planning algorithms (STRIPS, hierarchical planning)
- Learning and adaptation capabilities
- Web interface for agent interaction
- Database integration for persistent memory
- Multi-modal tool support (vision, audio)

## Contributing

When adding new features:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new functionality
3. Update documentation and examples
4. Consider backward compatibility
5. Add proper error handling and logging

## License

This project is part of the AIVault repository and follows the same license terms.

---

*This AI agents framework provides a solid foundation for building autonomous systems. Experiment with different tools, agent configurations, and multi-agent coordination to solve complex real-world problems!*

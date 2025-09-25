"""
Test cases for the AI Agents module.
"""

import sys
import os

# Add the parent directory to the path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from aivault.generative_ai.large_language_models.ai_agents import (  # noqa: E402
    CalculatorTool, SearchTool, MemoryTool, ReActAgent, MultiAgentSystem,
    AgentState, create_example_agent_system
)


class TestTools:
    """Test cases for agent tools."""
    
    def test_calculator_tool(self):
        """Test the calculator tool functionality."""
        calc = CalculatorTool()
        
        # Test basic operations
        assert "Result: 15" in calc.execute("10 + 5")
        assert "Result: 25" in calc.execute("5 * 5")
        assert "Result: 2.0" in calc.execute("10 / 5")
        
        # Test error handling
        result = calc.execute("invalid_expression!")
        assert "Error" in result
    
    def test_search_tool(self):
        """Test the search tool functionality."""
        search = SearchTool()
        
        # Test known queries
        result = search.execute("python")
        assert "Python is a high-level programming language" in result
        
        result = search.execute("ai")
        assert "Artificial Intelligence" in result
        
        # Test unknown queries
        result = search.execute("unknown_topic")
        assert "No specific results found" in result
    
    def test_memory_tool(self):
        """Test the memory tool functionality."""
        memory = MemoryTool()
        
        # Test store operation
        result = memory.execute("store", "test_key", "test_value")
        assert "Stored memory" in result
        
        # Test retrieve operation
        result = memory.execute("retrieve", "test_key")
        assert "test_value" in result
        
        # Test list operation
        result = memory.execute("list")
        assert "test_key" in result
        
        # Test invalid operation
        result = memory.execute("invalid_action")
        assert "Invalid action" in result


class TestReActAgent:
    """Test cases for the ReAct agent."""
    
    def test_agent_creation(self):
        """Test agent creation and basic properties."""
        agent = ReActAgent("TestAgent")
        
        assert agent.name == "TestAgent"
        assert agent.state == AgentState.IDLE
        assert len(agent.tools) == 0
        assert len(agent.conversation_history) == 0
    
    def test_tool_management(self):
        """Test adding and removing tools."""
        agent = ReActAgent("TestAgent")
        calc = CalculatorTool()
        
        # Add tool
        agent.add_tool(calc)
        assert "calculator" in agent.tools
        assert len(agent.tools) == 1
        
        # Remove tool
        agent.remove_tool("calculator")
        assert "calculator" not in agent.tools
        assert len(agent.tools) == 0
    
    def test_process_input_with_calculator(self):
        """Test processing mathematical input."""
        agent = ReActAgent("MathAgent", max_iterations=3)
        agent.add_tool(CalculatorTool())
        
        response = agent.process_input("What is 10 + 5?")
        
        # Should contain the calculation result
        assert "15" in response or "Result: 15" in response
        
        # Check conversation history
        assert len(agent.conversation_history) == 2  # user + assistant
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"
    
    def test_agent_reset(self):
        """Test agent reset functionality."""
        agent = ReActAgent("TestAgent")
        agent.add_tool(CalculatorTool())
        
        # Process some input
        agent.process_input("test")
        assert len(agent.conversation_history) > 0
        assert len(agent.thoughts) > 0
        
        # Reset agent
        agent.reset()
        assert agent.state == AgentState.IDLE
        assert len(agent.conversation_history) == 0


class TestMultiAgentSystem:
    """Test cases for the multi-agent system."""
    
    def test_system_creation(self):
        """Test multi-agent system creation."""
        system = MultiAgentSystem("TestSystem")
        
        assert system.name == "TestSystem"
        assert len(system.agents) == 0
        assert len(system.conversation_log) == 0
    
    def test_agent_management(self):
        """Test adding and removing agents."""
        system = MultiAgentSystem("TestSystem")
        agent = ReActAgent("TestAgent")
        
        # Add agent
        system.add_agent(agent)
        assert "TestAgent" in system.agents
        assert len(system.agents) == 1
        
        # Remove agent
        system.remove_agent("TestAgent")
        assert "TestAgent" not in system.agents
        assert len(system.agents) == 0
    
    def test_task_processing(self):
        """Test task processing with multi-agent system."""
        system = MultiAgentSystem("TestSystem")
        agent = ReActAgent("TestAgent")
        agent.add_tool(CalculatorTool())
        system.add_agent(agent)
        
        response = system.process_task("What is 5 + 5?")
        
        # Should contain agent name and response
        assert "TestAgent" in response
        assert len(system.conversation_log) == 2  # task + response
    
    def test_conversation_log(self):
        """Test conversation log functionality."""
        system = MultiAgentSystem("TestSystem")
        agent = ReActAgent("TestAgent")
        system.add_agent(agent)
        
        # Process a task
        system.process_task("test task")
        
        log = system.get_conversation_log()
        assert len(log) == 2
        assert log[0]["type"] == "task"
        assert log[1]["type"] == "response"
        
        # Reset and check log is cleared
        system.reset_all_agents()
        assert len(system.conversation_log) == 0


class TestExampleSystem:
    """Test the example system creation."""
    
    def test_create_example_system(self):
        """Test creating the example agent system."""
        system = create_example_agent_system()
        
        assert isinstance(system, MultiAgentSystem)
        assert len(system.agents) > 0
        
        # Check that the agent has tools
        agent = next(iter(system.agents.values()))
        assert len(agent.tools) > 0
        
        # Test basic functionality
        response = system.process_task("What is 2 + 2?")
        assert response is not None
        assert len(response) > 0


if __name__ == "__main__":
    # Run tests if script is executed directly
    import unittest
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [TestTools, TestReActAgent, TestMultiAgentSystem, TestExampleSystem]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

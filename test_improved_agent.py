#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

# Test the improved ReActAgent logic
def test_improved_agent():
    try:
        from aivault.generative_ai.large_language_models.ai_agents import (
            ReActAgent, CalculatorTool, SearchTool, AgentState
        )
        
        print("Testing improved ReActAgent with task completion detection...")
        
        # Create agent with tools
        agent = ReActAgent("Test Agent", max_iterations=5)
        agent.add_tool(CalculatorTool())
        agent.add_tool(SearchTool())
        
        # Test mathematical task (should complete after one calculation)
        print("\n1. Testing mathematical task:")
        print("Task: What is 15 + 25?")
        result = agent.process_input("What is 15 + 25?")
        print(f"Result: {result}")
        print(f"Final state: {agent.state}")
        print(f"Iterations used: {len(agent.thoughts)}")
        
        # Reset for next test
        agent.reset()
        
        # Test search task (should complete after one search)
        print("\n2. Testing search task:")
        print("Task: What is Python?")
        result = agent.process_input("What is Python?")
        print(f"Result: {result}")
        print(f"Final state: {agent.state}")
        print(f"Iterations used: {len(agent.thoughts)}")
        
        print("\n✅ Tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_agent()

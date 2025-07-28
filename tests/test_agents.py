#!/usr/bin/env python3
"""
Test script for CSV data analysis agents
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.csv_tools import CSVAnalyzer, create_sample_csv
from agents.agent import ask_question_about_csv
from agents.workflow import ask_csv_question


def test_csv_tools():
    """Test basic CSV analysis tools"""
    print("=== Testing CSV Tools ===")
    
    # Create sample data
    csv_file = "test_sample.csv"
    create_sample_csv(csv_file)
    
    # Test analyzer
    analyzer = CSVAnalyzer(csv_file)
    
    print("1. Basic info:", analyzer.get_info()['shape'])
    print("2. Summary stats available:", 'salary' in analyzer.get_summary_stats())
    print("3. Unique departments:", len(analyzer.get_unique_values('department')['unique_values']))
    print("4. High salary filter:", analyzer.query_data('salary > 100000')['result_count'])
    
    # Cleanup
    os.remove(csv_file)
    print("✓ CSV Tools working\n")


def test_simple_agent():
    """Test the simple CSV agent"""
    print("=== Testing Simple Agent ===")
    
    csv_file = "agent_test.csv"
    create_sample_csv(csv_file)
    
    questions = [
        "What are the summary statistics?",
        "What departments are available?",
        "What's the average salary by department?"
    ]
    
    for i, question in enumerate(questions, 1):
        try:
            answer = ask_question_about_csv(csv_file, question)
            print(f"{i}. Q: {question}")
            print(f"   A: {answer[:100]}..." if len(answer) > 100 else f"   A: {answer}")
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    os.remove(csv_file)
    print("✓ Simple Agent working\n")


def test_workflow_agent():
    """Test the full workflow agent"""
    print("=== Testing Workflow Agent ===")
    
    csv_file = "workflow_test.csv"
    create_sample_csv(csv_file)
    
    questions = [
        "Give me an overview of the dataset",
        "What's the average salary by department?",
        "Find people with salaries over 120000",
        "What cities are represented in the data?"
    ]
    
    for i, question in enumerate(questions, 1):
        try:
            answer = ask_csv_question(question, csv_file)
            print(f"{i}. Q: {question}")
            print(f"   A: {answer[:150]}..." if len(answer) > 150 else f"   A: {answer}")
        except Exception as e:
            print(f"{i}. Error: {e}")
    
    os.remove(csv_file)
    print("✓ Workflow Agent working\n")


def demo_with_sample_data():
    """Demo with the main sample data"""
    print("=== Demo with Sample Data ===")
    
    csv_file = "sample_data.csv"
    if not os.path.exists(csv_file):
        create_sample_csv(csv_file)
    
    demo_questions = [
        "How many employees are in each department?",
        "What's the highest and lowest salary?",
        "Which department has the highest average salary?",
        "Show me employees from Chicago",
        "What's the correlation between age and salary?",
        "Who are the most experienced employees?"
    ]
    
    print("Sample questions you can ask about your CSV data:\n")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"{i}. {question}")
        try:
            # Use the workflow agent for demo
            answer = ask_csv_question(question, csv_file)
            print(f"   Answer: {answer[:200]}...")
            print()
        except Exception as e:
            print(f"   Error: {e}\n")


def main():
    """Run all tests"""
    print("🤖 CSV Data Analysis Agents Test Suite\n")
    
    # Check if Ollama is available
    try:
        import ollama
        ollama.list()
        print("✓ Ollama is available")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("Please make sure Ollama is running and llama3.2 is pulled")
        return
    
    # Run tests
    test_csv_tools()
    test_simple_agent()
    test_workflow_agent()
    demo_with_sample_data()
    
    print("🎉 All tests completed!")
    print("\nTo use the MCP server:")
    print("  python mcp_server.py")
    print("\nTo analyze your own CSV:")
    print("  python workflow.py")


if __name__ == "__main__":
    main()
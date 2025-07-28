import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import ollama
import json
from tools.csv_tools import CSVAnalyzer

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    csv_path: str
    analyzer: CSVAnalyzer
    next_action: Literal["analyze", "query", "respond", "end"]


def create_data_analyst_agent(csv_path: str, model_name: str = "llama3.2"):
    """Create a data analyst agent that can answer questions about CSV data"""
    
    analyzer = CSVAnalyzer(csv_path)
    
    def analyzer_node(state: AgentState):
        """Node that analyzes the data and determines what to do"""
        messages = state["messages"]
        latest_message = messages[-1].content if messages else ""
        
        # Get basic info about the dataset
        dataset_info = analyzer.get_info()
        
        system_prompt = f"""You are a data analyst AI. You have access to a CSV dataset with the following structure:
        
Dataset Info:
- Shape: {dataset_info['shape']}
- Columns: {dataset_info['columns']}
- Data types: {dataset_info['dtypes']}

Sample data:
{json.dumps(dataset_info['sample_data'], indent=2)}

Based on the user's question: "{latest_message}"

Determine the best approach to answer their question. You can:
1. Get summary statistics
2. Filter or query the data
3. Group and aggregate data
4. Search for specific text
5. Get unique values in columns

Respond with a clear analysis plan and what specific operations you would perform."""
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': latest_message}
                ]
            )
            
            analysis_plan = response['message']['content']
            
            return {
                "messages": [AIMessage(content=f"Analysis Plan: {analysis_plan}")],
                "analyzer": analyzer,
                "next_action": "query"
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error in analysis: {e}")],
                "next_action": "end"
            }
    
    def query_executor_node(state: AgentState):
        """Node that executes data queries based on the analysis plan"""
        messages = state["messages"]
        latest_message = messages[-1].content if messages else ""
        user_question = messages[-2].content if len(messages) >= 2 else ""
        
        # Determine what type of query to execute based on the user's question
        query_results = {}
        
        # Simple keyword-based query routing
        question_lower = user_question.lower()
        
        if any(word in question_lower for word in ['summary', 'statistics', 'stats', 'describe']):
            query_results['summary_stats'] = analyzer.get_summary_stats()
        
        if any(word in question_lower for word in ['unique', 'distinct', 'values']):
            # Try to identify column from question
            for col in analyzer.df.columns:
                if col.lower() in question_lower:
                    query_results[f'unique_{col}'] = analyzer.get_unique_values(col)
                    break
        
        if any(word in question_lower for word in ['filter', 'where', 'find']):
            # This is a simple example - in practice you'd want more sophisticated parsing
            if 'salary' in question_lower and 'high' in question_lower:
                query_results['high_salary'] = analyzer.query_data('salary > 100000')
        
        if any(word in question_lower for word in ['group', 'average', 'mean', 'sum']):
            # Example: average salary by department
            if 'department' in question_lower and 'salary' in question_lower:
                query_results['avg_salary_by_dept'] = analyzer.group_and_aggregate('department', 'salary', 'mean')
        
        # If no specific query was triggered, get basic info
        if not query_results:
            query_results['basic_info'] = analyzer.get_info()
        
        return {
            "messages": [AIMessage(content=f"Query Results: {json.dumps(query_results, indent=2, default=str)}")],
            "next_action": "respond"
        }
    
    def response_generator_node(state: AgentState):
        """Node that generates a natural language response based on query results"""
        messages = state["messages"]
        user_question = messages[0].content if messages else ""
        query_results = messages[-1].content if messages else ""
        
        system_prompt = f"""You are a helpful data analyst. Based on the query results below, provide a clear, concise answer to the user's question.

User Question: {user_question}

Query Results: {query_results}

Provide a natural language response that directly answers the user's question using the data. Include specific numbers and insights where relevant."""
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'Please provide a clear answer based on the data.'}
                ]
            )
            
            final_response = response['message']['content']
            
            return {
                "messages": [AIMessage(content=final_response)],
                "next_action": "end"
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error generating response: {e}")],
                "next_action": "end"
            }
    
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("query_executor", query_executor_node)
    workflow.add_node("response_generator", response_generator_node)
    
    # Define the flow
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "query_executor")
    workflow.add_edge("query_executor", "response_generator")
    workflow.add_edge("response_generator", END)
    
    return workflow.compile()


def ask_question_about_csv(csv_path: str, question: str, model_name: str = "llama3.2"):
    """Convenience function to ask a question about CSV data"""
    
    graph = create_data_analyst_agent(csv_path, model_name)
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "csv_path": csv_path,
        "analyzer": None,
        "next_action": "analyze"
    }
    
    result = graph.invoke(initial_state)
    return result["messages"][-1].content


if __name__ == "__main__":
    # Test the agent
    csv_file = "sample_data.csv"
    
    # Create sample data if it doesn't exist
    if not os.path.exists(csv_file):
        from tools.csv_tools import create_sample_csv
        create_sample_csv(csv_file)
    
    print("=== CSV Data Analysis Agent ===")
    
    # Test questions
    questions = [
        "What are the summary statistics for this dataset?",
        "What are the unique departments in the data?",
        "What is the average salary by department?",
        "How many people earn more than 100,000?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        try:
            answer = ask_question_about_csv(csv_file, question)
            print(f"A: {answer}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)
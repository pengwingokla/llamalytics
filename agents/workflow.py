import os
from typing import TypedDict, Annotated, Literal, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import ollama
import json
from tools.csv_tools import CSVAnalyzer

load_dotenv()


class WorkflowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    csv_files: List[str]
    current_csv: str
    query_history: List[dict]
    context: str


def create_csv_qa_workflow(model_name: str = "llama3.2"):
    """Create a comprehensive CSV Q&A workflow with multiple agents"""
    
    def file_manager_node(state: WorkflowState):
        """Manages CSV file selection and loading"""
        messages = state["messages"]
        latest_message = messages[-1].content if messages else ""
        
        # Check if user is asking about a specific file
        csv_files = state.get("csv_files", [])
        
        # Simple file detection logic
        if ".csv" in latest_message.lower():
            # Extract CSV filename from message
            words = latest_message.split()
            for word in words:
                if word.endswith('.csv'):
                    if os.path.exists(word):
                        return {
                            "current_csv": word,
                            "messages": [AIMessage(content=f"Loaded CSV file: {word}")],
                            "context": f"Working with {word}"
                        }
                    else:
                        return {
                            "messages": [AIMessage(content=f"File {word} not found")],
                            "context": "No file loaded"
                        }
        
        # If no specific file mentioned, use default or first available
        current_csv = state.get("current_csv")
        if not current_csv:
            # Look for CSV files in current directory
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                current_csv = csv_files[0]
                return {
                    "current_csv": current_csv,
                    "messages": [AIMessage(content=f"Using default CSV: {current_csv}")],
                    "context": f"Working with {current_csv}"
                }
            else:
                return {
                    "messages": [AIMessage(content="No CSV files found. Please specify a CSV file.")],
                    "context": "No file loaded"
                }
        
        return {
            "current_csv": current_csv,
            "context": f"Working with {current_csv}"
        }
    
    def intent_classifier_node(state: WorkflowState):
        """Intelligent intent classifier that determines specific tools and parameters needed"""
        messages = state["messages"]
        current_csv = state.get("current_csv")
        user_question = ""
        
        # Find the last human message
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        if not user_question:
            return {
                "messages": [AIMessage(content="No question found to classify")],
                "context": state.get("context", "")
            }
        
        # Get basic CSV info for context
        csv_context = ""
        if current_csv and os.path.exists(current_csv):
            try:
                analyzer = CSVAnalyzer(current_csv)
                csv_context = f"CSV columns: {', '.join(analyzer.df.columns)}"
            except:
                csv_context = "CSV structure unknown"
        
        intent_prompt = f"""You are an intelligent intent classifier for CSV data analysis. Analyze the user's question and determine exactly which tools to use.

Question: "{user_question}"
{csv_context}

Available tools:
1. get_info() - Basic dataset information (shape, columns, types)
2. get_summary_stats() - Statistical summaries for numeric columns
3. search_text(column, term) - Find text in specific column
4. get_unique_values(column) - Get unique values in a column
5. query_data(query) - Filter data with pandas query syntax
6. group_and_aggregate(group_col, agg_col, function) - Group and calculate aggregations

IMPORTANT: Use only ONE primary tool per question. Avoid MULTI_TOOL unless absolutely necessary.

Respond with a JSON object containing:
{{
    "intent": "one of: SUMMARY, SEARCH, FILTER, AGGREGATE, UNIQUE_VALUES",
    "tools": [
        {{
            "tool": "tool_name",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }},
            "reason": "why this tool is needed"
        }}
    ],
    "confidence": 0.9
}}

Examples:
- "Show me Princeton University data" → SEARCH with search_text(university_name, "Princeton")
- "What years are in the data?" → UNIQUE_VALUES with get_unique_values(year)
- "Average total enrollment by year" → AGGREGATE with group_and_aggregate(year, Total, mean)
- "Universities with total enrollment > 1000" → FILTER with query_data("Total > 1000")
- "Overview of dataset" → SUMMARY with get_info() and get_summary_stats()

For search questions like "Show me [specific entity]", always use SEARCH intent with search_text tool."""
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'user', 'content': intent_prompt}
                ]
            )
            
            intent_analysis = response['message']['content']
            
            return {
                "messages": [AIMessage(content=f"Intent Analysis: {intent_analysis}")],
                "context": state.get("context", "") + f" | Intent: {intent_analysis}"
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Intent analysis error: {e}")],
                "context": state.get("context", "")
            }
    
    def data_analyst_node(state: WorkflowState):
        """Performs data analysis based on intent classifier recommendations"""
        messages = state["messages"]
        current_csv = state.get("current_csv")
        context = state.get("context", "")
        
        if not current_csv or not os.path.exists(current_csv):
            return {
                "messages": [AIMessage(content="No valid CSV file available for analysis")],
                "context": context
            }
        
        # Find the original user question and intent analysis
        user_question = ""
        intent_analysis = ""
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "Intent Analysis:" in msg.content:
                intent_analysis = msg.content.replace("Intent Analysis: ", "")
                break
        
        try:
            analyzer = CSVAnalyzer(current_csv)
            analysis_results = {}
            
            # Parse intent analysis JSON
            try:
                intent_data = json.loads(intent_analysis)
                print(intent_data)
                tools_to_execute = intent_data.get("tools", [])
            except (json.JSONDecodeError, AttributeError):
                # Fallback to basic analysis if JSON parsing fails
                tools_to_execute = [{"tool": "get_info", "parameters": {}}]
            
            # Execute each recommended tool
            for tool_spec in tools_to_execute:
                tool_name = tool_spec.get("tool", "")
                parameters = tool_spec.get("parameters", {})
                
                try:
                    if tool_name == "get_info":
                        analysis_results['dataset_info'] = analyzer.get_info()
                    
                    elif tool_name == "get_summary_stats":
                        analysis_results['summary_stats'] = analyzer.get_summary_stats()
                    
                    elif tool_name == "search_text":
                        column = parameters.get("column", "")
                        term = parameters.get("term", "")
                        if column and term:
                            # Try exact column name first, then fuzzy match
                            target_col = column if column in analyzer.df.columns else None
                            if not target_col:
                                # Fuzzy match column names
                                for col in analyzer.df.columns:
                                    if column.lower() in col.lower() or col.lower() in column.lower():
                                        target_col = col
                                        break
                            
                            if target_col:
                                analysis_results[f'search_{term}_in_{target_col}'] = analyzer.search_text(target_col, term)
                    
                    elif tool_name == "get_unique_values":
                        column = parameters.get("column", "")
                        if column:
                            # Try exact column name first, then fuzzy match
                            target_col = column if column in analyzer.df.columns else None
                            if not target_col:
                                for col in analyzer.df.columns:
                                    if column.lower() in col.lower() or col.lower() in column.lower():
                                        target_col = col
                                        break
                            
                            if target_col:
                                analysis_results[f'unique_values_{target_col}'] = analyzer.get_unique_values(target_col)
                    
                    elif tool_name == "query_data":
                        query = parameters.get("query", "")
                        if query:
                            analysis_results[f'filter_query'] = analyzer.query_data(query)
                    
                    elif tool_name == "group_and_aggregate":
                        group_col = parameters.get("group_col", "")
                        agg_col = parameters.get("agg_col", "")
                        function = parameters.get("function", "mean")
                        
                        if group_col and agg_col:
                            # Fuzzy match column names
                            target_group_col = group_col if group_col in analyzer.df.columns else None
                            target_agg_col = agg_col if agg_col in analyzer.df.columns else None
                            
                            if not target_group_col:
                                for col in analyzer.df.columns:
                                    if group_col.lower() in col.lower() or col.lower() in group_col.lower():
                                        target_group_col = col
                                        break
                            
                            if not target_agg_col:
                                for col in analyzer.df.columns:
                                    if agg_col.lower() in col.lower() or col.lower() in agg_col.lower():
                                        target_agg_col = col
                                        break
                            
                            if target_group_col and target_agg_col:
                                analysis_results[f'{function}_{target_agg_col}_by_{target_group_col}'] = analyzer.group_and_aggregate(target_group_col, target_agg_col, function)
                
                except Exception as tool_error:
                    analysis_results[f'error_{tool_name}'] = f"Tool execution error: {tool_error}"
            
            # If no analysis was performed, get basic info as fallback
            if not analysis_results:
                analysis_results['fallback_info'] = analyzer.get_info()
            
            return {
                "messages": [AIMessage(content=f"Analysis Results: {json.dumps(analysis_results, indent=2, default=str)}")],
                "context": context,
                "query_history": state.get("query_history", []) + [{"question": user_question, "results": analysis_results}]
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Analysis error: {e}")],
                "context": context
            }
    
    def response_synthesizer_node(state: WorkflowState):
        """Synthesizes a natural language response from analysis results"""
        messages = state["messages"]
        context = state.get("context", "")
        
        # Get user question and analysis results
        user_question = ""
        analysis_results = ""
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "Analysis Results:" in msg.content:
                analysis_results = msg.content
                break
        
        synthesis_prompt = f"""You are a data analyst providing insights to a user. You MUST base your response ONLY on the provided analysis results. Do NOT make up or guess any information.

Context: {context}
User Question: {user_question}
Analysis Results: {analysis_results}

IMPORTANT: 
- Use ONLY the numbers and facts from the Analysis Results
- If the analysis results contain specific data like match counts or records, report those exact numbers
- Do NOT add information not present in the analysis results
- If the analysis results show search matches, report the exact number of matches found

Provide a clear, concise response that directly answers the question using only the provided data."""
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'user', 'content': synthesis_prompt}
                ]
            )
            
            final_response = response['message']['content']
            
            return {
                "messages": [AIMessage(content=final_response)],
                "context": context + " | Response generated"
            }
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Response generation error: {e}")],
                "context": context
            }
    
    # Create the workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("file_manager", file_manager_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("response_synthesizer", response_synthesizer_node)
    
    # Define the flow
    workflow.add_edge(START, "file_manager")
    workflow.add_edge("file_manager", "intent_classifier")
    workflow.add_edge("intent_classifier", "data_analyst")
    workflow.add_edge("data_analyst", "response_synthesizer")
    workflow.add_edge("response_synthesizer", END)
    
    return workflow.compile()


def ask_question(question: str, csv_file: str = None, model_name: str = "llama3.2", raw_results: bool = False):
    """Ask a question about CSV data using the full workflow"""
    
    workflow = create_csv_qa_workflow(model_name)
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "csv_files": [],
        "current_csv": csv_file,
        "query_history": [],
        "context": ""
    }
    
    result = workflow.invoke(initial_state)
    
    if raw_results:
        # Get raw analysis results
        analysis_result = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and "Analysis Results:" in msg.content:
                analysis_result = msg.content

        # Get final response (skip intent analysis and analysis results)
        final_response = None
        for msg in reversed(result["messages"]):
            if (isinstance(msg, AIMessage) and 
                "Response generation error" not in msg.content and
                "Intent Analysis:" not in msg.content and
                "Analysis Results:" not in msg.content):
                final_response = msg.content
                break

        return {
            "analysis": analysis_result or "No analysis results found",
            "response": final_response or "No final response generated"
        }
    
    # Return just the final response
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and "Response generation error" not in msg.content:
            return msg.content
    
    return "No response generated"


if __name__ == "__main__":
    # Test the workflow
    csv_file = "sample_data.csv"
    
    # Create sample data if it doesn't exist
    if not os.path.exists(csv_file):
        from tools.csv_tools import create_sample_csv
        create_sample_csv(csv_file)
    
    print("=== CSV Q&A Workflow ===")
    
    # Test questions
    questions = [
        "What's the overview of this dataset?",
        "What's the average salary by department?",
        "Find people with high salaries",
        "Show me unique cities in the data"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        try:
            answer = ask_csv_question(question, csv_file)
            print(f"A: {answer}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)
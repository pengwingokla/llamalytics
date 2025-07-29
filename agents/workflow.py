import os
from typing import TypedDict, Annotated, Literal, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import ollama
import json
from tools.csv_tools import CSVAnalyzer
from .prompt import INTENT_CLASSIFIER_PROMPT, SYNTHESIS_PROMPT
from .subagents import MultiAgentIntentClassifier

load_dotenv()

MODEL_NAME = "llama3.2"

class WorkflowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    csv_files: List[str]
    current_csv: str
    query_history: List[dict]
    context: str


def create_csv_qa_workflow(model_name: str = MODEL_NAME):
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
        """Multi-agent intent classifier using specialized subagents"""
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
        
        # Get CSV columns for analysis
        csv_columns = []
        if current_csv and os.path.exists(current_csv):
            try:
                analyzer = CSVAnalyzer(current_csv)
                csv_columns = list(analyzer.df.columns)
            except:
                csv_columns = []
        
        try:
            # Use the new multi-agent classifier
            classifier = MultiAgentIntentClassifier(model_name)
            intent_analysis = classifier.classify_intent(user_question, csv_columns)
            
            # Convert to JSON string for compatibility with existing workflow
            intent_json = json.dumps(intent_analysis, indent=2)
            
            return {
                "messages": [AIMessage(content=f"Intent Analysis: {intent_json}")],
                "context": state.get("context", "") + f" | Intent: {intent_analysis.get('intent', 'Unknown')}"
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
        
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            context=context,
            user_question=user_question,
            analysis_results=analysis_results
        )
        
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


def ask_question(question: str, csv_file: str = None, model_name: str = MODEL_NAME, raw_results: bool = False):
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
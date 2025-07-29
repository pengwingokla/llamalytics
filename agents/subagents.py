"""
Specialized subagents for intent classification
"""
import ollama
import json
from typing import Dict, Any, List
from .subagent_prompts import (
    QUESTION_TYPE_ROUTER_PROMPT,
    COLUMN_REASONER_PROMPT, 
    TOOL_SELECTOR_PROMPT
)

MODEL_NAME = "llama3.2"

class QuestionTypeRouter:
    """Classifies questions into high-level categories"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
    
    def classify(self, user_question: str) -> str:
        """Return one of: SUMMARY, SEARCH, FILTER, AGGREGATE, UNIQUE_VALUES"""
        prompt = QUESTION_TYPE_ROUTER_PROMPT.format(user_question=user_question)
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Extract just the classification word
            result = response['message']['content'].strip().upper()
            
            # Validate result
            valid_types = ["SUMMARY", "SEARCH", "FILTER", "AGGREGATE", "UNIQUE_VALUES"]
            if result in valid_types:
                return result
            else:
                # Fallback if invalid response
                return "SUMMARY"
                
        except Exception as e:
            print(f"Question Type Router Error: {e}")
            return "SUMMARY"  # Safe fallback

class ColumnReasoner:
    """Identifies relevant columns and their types"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
    
    def analyze_columns(self, user_question: str, question_type: str, columns: List[str]) -> Dict[str, Any]:
        """Return column analysis with types and reasoning"""
        columns_str = ", ".join(columns)
        prompt = COLUMN_REASONER_PROMPT.format(
            user_question=user_question,
            question_type=question_type,
            columns=columns_str
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Parse JSON response
            content = response['message']['content'].strip()
            
            # Clean JSON string
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Column Reasoner Error: {e}")
            # Fallback: use first column
            return {
                "primary_columns": [columns[0]] if columns else [],
                "column_types": {columns[0]: "text"} if columns else {},
                "reasoning": f"Fallback due to error: {e}"
            }

class ToolSelector:
    """Selects optimal tool and parameters"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
    
    def select_tool(self, user_question: str, question_type: str, column_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Return tool selection with parameters"""
        prompt = TOOL_SELECTOR_PROMPT.format(
            user_question=user_question,
            question_type=question_type,
            column_analysis=json.dumps(column_analysis, indent=2)
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Parse JSON response
            content = response['message']['content'].strip()
            
            # Clean JSON string
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Tool Selector Error: {e}")
            # Fallback based on question type
            if question_type == "SUMMARY":
                return {"tool": "get_info", "parameters": {}, "reasoning": f"Fallback due to error: {e}"}
            elif question_type == "SEARCH":
                return {"tool": "search_text", "parameters": {"column": "university_name", "term": "university"}, "reasoning": f"Fallback due to error: {e}"}
            elif question_type == "FILTER":
                return {"tool": "query_data", "parameters": {"query": "Total > 0"}, "reasoning": f"Fallback due to error: {e}"}
            elif question_type == "AGGREGATE":
                return {"tool": "group_and_aggregate", "parameters": {"group_col": "year", "agg_col": "Total", "function": "mean"}, "reasoning": f"Fallback due to error: {e}"}
            else:  # UNIQUE_VALUES
                return {"tool": "get_unique_values", "parameters": {"column": "year"}, "reasoning": f"Fallback due to error: {e}"}

class MultiAgentIntentClassifier:
    """Coordinates the three specialized subagents"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.question_router = QuestionTypeRouter(model_name)
        self.column_reasoner = ColumnReasoner(model_name)
        self.tool_selector = ToolSelector(model_name)
    
    def classify_intent(self, user_question: str, csv_columns: List[str]) -> Dict[str, Any]:
        """Run the full 3-step classification process"""
        
        # Step 1: Classify question type
        question_type = self.question_router.classify(user_question)
        
        # Step 2: Analyze relevant columns
        column_analysis = self.column_reasoner.analyze_columns(user_question, question_type, csv_columns)
        
        # Step 3: Select tool and parameters
        tool_selection = self.tool_selector.select_tool(user_question, question_type, column_analysis)
        
        # Combine results
        return {
            "intent": question_type,
            "column_analysis": column_analysis,
            "tools": [tool_selection],
            "confidence": 0.9,
            "process": {
                "step1_question_type": question_type,
                "step2_columns": column_analysis,
                "step3_tool": tool_selection
            }
        }
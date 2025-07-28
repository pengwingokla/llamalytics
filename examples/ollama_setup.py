import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import ollama

load_dotenv()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_ollama_node(model_name="llama3.2"):
    def ollama_node(state: State):
        messages = state["messages"]
        
        # Get the latest human message
        latest_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_message = msg.content
                break
        
        if not latest_message:
            return {"messages": [AIMessage(content="No message to process")]}
        
        try:
            # Call Ollama API
            response = ollama.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': latest_message
                }]
            )
            
            content = response['message']['content']
            return {"messages": [AIMessage(content=content)]}
            
        except Exception as e:
            error_msg = f"Error calling Ollama: {str(e)}"
            if "model not found" in str(e).lower():
                error_msg += f"\nTry running: ollama pull {model_name}"
            return {"messages": [AIMessage(content=error_msg)]}
    
    return ollama_node


def create_ollama_graph(model_name="llama3.2"):
    workflow = StateGraph(State)
    
    ollama_node = create_ollama_node(model_name)
    workflow.add_node("ollama", ollama_node)
    
    workflow.add_edge(START, "ollama")
    workflow.add_edge("ollama", END)
    
    return workflow.compile()


def check_ollama_status():
    """Check if Ollama is running and what models are available"""
    try:
        models = ollama.list()
        print("Available Ollama models:")
        if models.get('models'):
            for model in models['models']:
                print(f"  - {model.get('name', 'Unknown')}")
        else:
            print("  No models found. Pull a model first:")
            print("  ollama pull llama3.2")
        return True
    except Exception as e:
        print(f"Ollama not available: {e}")
        print("Make sure Ollama is installed and running:")
        print("1. Install: https://ollama.ai/download")
        print("2. Start: ollama serve")
        print("3. Pull model: ollama pull llama3.2")
        return False


if __name__ == "__main__":
    print("=== Ollama LangGraph Integration ===")
    
    if not check_ollama_status():
        exit(1)
    
    # Create graph with default model
    graph = create_ollama_graph("llama3.2")
    
    print("\nTesting Ollama integration...")
    initial_state = {
        "messages": [HumanMessage(content="Hello! Tell me a short joke.")]
    }
    
    result = graph.invoke(initial_state)
    print(f"\nOllama Response: {result['messages'][-1].content}")
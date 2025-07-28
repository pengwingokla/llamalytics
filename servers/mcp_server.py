import asyncio
import logging
import os
from typing import Any, Sequence
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import json

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("csv-data-analyst-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_csv",
            description="Analyze CSV data and answer questions using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to ask about the CSV data"
                    },
                    "csv_file": {
                        "type": "string",
                        "description": "Path to CSV file (optional, will auto-detect if not provided)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Ollama model to use",
                        "default": "llama3.2"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="get_csv_info",
            description="Get basic information about a CSV file",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_file": {
                        "type": "string",
                        "description": "Path to CSV file"
                    }
                },
                "required": ["csv_file"]
            }
        ),
        Tool(
            name="query_csv_data",
            description="Execute specific pandas queries on CSV data",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_file": {
                        "type": "string",
                        "description": "Path to CSV file"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Type of operation",
                        "enum": ["info", "summary", "filter", "group", "unique", "search"]
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters for the operation",
                        "properties": {
                            "column": {"type": "string"},
                            "value": {"type": "string"},
                            "query": {"type": "string"},
                            "group_by": {"type": "string"},
                            "agg_column": {"type": "string"},
                            "agg_func": {"type": "string"},
                            "search_term": {"type": "string"}
                        }
                    }
                },
                "required": ["csv_file", "operation"]
            }
        ),
        Tool(
            name="list_csv_files",
            description="List available CSV files in the current directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="create_sample_csv",
            description="Create a sample CSV file for testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name for the sample CSV file",
                        "default": "sample_data.csv"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    if name == "analyze_csv":
        try:
            from agents.workflow import ask_csv_question
            
            question = arguments.get("question", "")
            csv_file = arguments.get("csv_file")
            model = arguments.get("model", "llama3.2")
            
            if not question:
                return [TextContent(type="text", text="Please provide a question to analyze")]
            
            # If no CSV file specified, try to find one
            if not csv_file:
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                if csv_files:
                    csv_file = csv_files[0]
                else:
                    return [TextContent(type="text", text="No CSV file found. Please specify a file or create sample data.")]
            
            if not os.path.exists(csv_file):
                return [TextContent(type="text", text=f"CSV file '{csv_file}' not found")]
            
            answer = ask_csv_question(question, csv_file, model)
            return [TextContent(type="text", text=f"Question: {question}\n\nAnswer: {answer}")]
            
        except Exception as e:
            logger.error(f"Error analyzing CSV: {e}")
            return [TextContent(type="text", text=f"Error analyzing CSV: {str(e)}")]
    
    elif name == "get_csv_info":
        try:
            from tools.csv_tools import CSVAnalyzer
            
            csv_file = arguments.get("csv_file", "")
            
            if not os.path.exists(csv_file):
                return [TextContent(type="text", text=f"File '{csv_file}' not found")]
            
            analyzer = CSVAnalyzer(csv_file)
            info = analyzer.get_info()
            
            return [TextContent(type="text", text=f"CSV Info for {csv_file}:\n{json.dumps(info, indent=2, default=str)}")]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting CSV info: {str(e)}")]
    
    elif name == "query_csv_data":
        try:
            from tools.csv_tools import CSVAnalyzer
            
            csv_file = arguments.get("csv_file", "")
            operation = arguments.get("operation", "")
            parameters = arguments.get("parameters", {})
            
            if not os.path.exists(csv_file):
                return [TextContent(type="text", text=f"File '{csv_file}' not found")]
            
            analyzer = CSVAnalyzer(csv_file)
            result = {}
            
            if operation == "info":
                result = analyzer.get_info()
            elif operation == "summary":
                result = analyzer.get_summary_stats()
            elif operation == "filter":
                column = parameters.get("column", "")
                value = parameters.get("value", "")
                if column and value:
                    result = analyzer.filter_data(column, value)
                else:
                    result = {"error": "Filter operation requires 'column' and 'value' parameters"}
            elif operation == "group":
                group_by = parameters.get("group_by", "")
                agg_column = parameters.get("agg_column", "")
                agg_func = parameters.get("agg_func", "mean")
                if group_by and agg_column:
                    result = analyzer.group_and_aggregate(group_by, agg_column, agg_func)
                else:
                    result = {"error": "Group operation requires 'group_by' and 'agg_column' parameters"}
            elif operation == "unique":
                column = parameters.get("column", "")
                if column:
                    result = analyzer.get_unique_values(column)
                else:
                    result = {"error": "Unique operation requires 'column' parameter"}
            elif operation == "search":
                column = parameters.get("column", "")
                search_term = parameters.get("search_term", "")
                if column and search_term:
                    result = analyzer.search_text(column, search_term)
                else:
                    result = {"error": "Search operation requires 'column' and 'search_term' parameters"}
            else:
                result = {"error": f"Unknown operation: {operation}"}
            
            return [TextContent(type="text", text=f"Query Result:\n{json.dumps(result, indent=2, default=str)}")]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error querying CSV: {str(e)}")]
    
    elif name == "list_csv_files":
        try:
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                file_list = "\n".join([f"- {f}" for f in csv_files])
                return [TextContent(type="text", text=f"Available CSV files:\n{file_list}")]
            else:
                return [TextContent(type="text", text="No CSV files found in current directory")]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error listing files: {str(e)}")]
    
    elif name == "create_sample_csv":
        try:
            from tools.csv_tools import create_sample_csv
            
            filename = arguments.get("filename", "sample_data.csv")
            created_file = create_sample_csv(filename)
            
            return [TextContent(type="text", text=f"Sample CSV created: {created_file}")]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error creating sample CSV: {str(e)}")]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
# CSV Data Analysis with LangGraph & Ollama

AI-powered CSV data analysis using LangGraph workflows and local Ollama models.

## 📁 Project Structure

```
mcp-demo/
├── agents/                 # LangGraph agents
│   ├── csv_agent.py       # Simple CSV Q&A agent
│   └── csv_workflow.py    # Advanced multi-agent workflow
├── tools/                 # Core analysis tools
│   └── csv_tools.py       # Pandas-based CSV utilities
├── servers/               # MCP servers
│   └── csv_mcp_server.py  # CSV analysis MCP server
├── examples/              # Usage examples
│   ├── ollama_setup.py    # Basic Ollama integration
│   └── setup_ollama.sh    # Ollama installation script
├── tests/                 # Test suites
│   └── test_csv_agents.py # Comprehensive test suite
├── data/                  # CSV data files
│   └── sample_data.csv    # Sample dataset
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

1. **Setup environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install and setup Ollama:**
   ```bash
   # Install from https://ollama.ai/download
   ollama serve
   ollama pull llama3.2
   ```

3. **Test the system:**
   ```bash
   python tests/test_csv_agents.py
   ```

## 🛠️ Usage

### Direct Agent Usage
```python
from agents.csv_workflow import ask_csv_question

answer = ask_csv_question("What's the average salary by department?", "data/sample_data.csv")
print(answer)
```

### MCP Server
```bash
python servers/csv_mcp_server.py
```

## 📊 Features

- **AI-Powered Analysis:** Natural language questions about CSV data
- **Multi-Agent Workflow:** File management → Question classification → Data analysis → Response synthesis
- **Local & Free:** Uses Ollama models (no API costs)
- **MCP Integration:** Expose capabilities via Model Context Protocol
- **Pandas Backend:** Robust data analysis operations

## 🔧 Components

- **csv_tools.py:** Core pandas operations (filter, group, search, stats)
- **csv_agent.py:** Simple LangGraph agent for basic Q&A
- **csv_workflow.py:** Advanced workflow with specialized nodes
- **csv_mcp_server.py:** MCP server exposing all functionality

## 📝 Example Questions

- "What's the overview of this dataset?"
- "What's the average salary by department?"
- "Find employees with high salaries"
- "Show me unique cities in the data"
- "Who are the most experienced employees?"

## 🧪 Testing

Run comprehensive tests:
```bash
python tests/test_csv_agents.py
```

Test individual components:
```bash
python tools/csv_tools.py        # Test pandas tools
python agents/csv_agent.py       # Test simple agent
python agents/csv_workflow.py    # Test workflow agent
```

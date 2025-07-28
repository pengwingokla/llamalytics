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
│   └── NJ_graduation_data.csv    # New Jersey graduation dataset
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
from agents.workflow import ask_question

answer = ask_question("What is the graduation rate for Asian women at New Jersey Institute of Technology in 2020?", "data/NJ_graduation_data.csv")
print(answer)
```

### MCP Server
```bash
python servers/csv_mcp_server.py
```

## 📊 Features

- **AI-Powered Analysis:** Natural language questions about graduation and enrollment data
- **Multi-Agent Workflow:** File management → Intent classification → Data analysis → Response synthesis
- **Local & Free:** Uses Ollama/Llama 3.2 models (no API costs)
- **Advanced Filtering:** Complex demographic and institutional queries
- **Pandas Backend:** Robust statistical and aggregation operations

## 🔧 Components

- **csv_tools.py:** Core pandas operations (filter, group, search, stats)
- **csv_agent.py:** Simple LangGraph agent for basic Q&A
- **csv_workflow.py:** Advanced workflow with specialized nodes
- **csv_mcp_server.py:** MCP server exposing all functionality

## 📝 Example Questions

### Simple Operations
- "What's the overview of this graduation dataset?"
- "What years are available in the data?"
- "Show me New Jersey Institute of Technology data"
- "What universities are included in this dataset?"

### Complex Queries
- "What is the graduation rate for Asian women at New Jersey Institute of Technology in 2020?"
- "Show total enrollment for Rutgers University in 2019"
- "Find institutions with more than 1000 total students"
- "What is the average total enrollment by year?"

### Demographic Analysis
- "Compare Asian men vs Asian women enrollment across all schools"
- "Which cohort types are in the data?"
- "Show graduation rates for Hispanic students in 2020"
- "What's the enrollment breakdown by race at Princeton University?"

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

# CSV Data Analysis with LangGraph & Ollama

AI-powered CSV data analysis using LangGraph workflows and local Ollama models.

## Project Structure

```
mcp-demo/
├── agents/                
│   ├── agent.py                  # Simple CSV Q&A agent
│   └── workflow.py               # Advanced multi-agent workflow
├── tools/                 
│   └── tools.py                  # Pandas-based CSV utilities
├── servers/               
│   └── mcp_server.py             # CSV analysis MCP server
├── examples/              
│   ├── ollama_setup.py           # Basic Ollama integration
│   └── setup_ollama.sh           # Ollama installation script
├── tests/                 
│   └── test_workflow.py.         # Comprehensive test suite
├── data/                  
│   └── NJ_graduation_data.csv    # Dataset
└── requirements.txt       # Dependencies
```

## Quick Start

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
   python tests/test_agents_outputs.py
   ```

## Usage

### Direct Agent Usage
```python
from agents.workflow import ask_question

answer = ask_question("What is the graduation rate for Asian women at New Jersey Institute of Technology in 2020?", "data/NJ_graduation_data.csv")
print(answer)
```

## Features

- **AI-Powered Analysis:** Natural language questions about graduation and enrollment data
- **Multi-Agent Workflow:** File management → Intent classification → Data analysis → Response synthesis
- **Local & Free:** Uses Ollama/Llama 3.2 models (no API costs)
- **Advanced Filtering:** Complex demographic and institutional queries
- **Pandas Backend:** Robust statistical and aggregation operations

## Components

- **tools.py:** Core pandas operations (filter, group, search, stats)
- **workflow.py:** Advanced workflow with specialized nodes

## Example Questions

### Retrieval Questions
- What is the graduation rate for Asian women at New Jersey Institute of Technology in 2020?
- Which universities had cohorts larger than 5,000 students in 2019?
- Find all universities in California that tracked graduation rates for Native Hawaiian students.
### Aggregation Questions
- What is the average graduation rate across all universities for women between 2018-2022?
- Calculate the total number of Hispanic students (men and women combined) across all universities in 2021.
- What percentage of the total cohort do international students (nonresident aliens) represent on average?
- Which racial/ethnic group has the highest overall graduation rate when aggregated across all universities and years?
### Comparison Questions
- Compare graduation rates between men and women across different racial groups - which group shows the largest gender gap?
 - How do graduation rates for first-generation college students compare to continuing-generation students across different university types?
- Which universities show the most improvement in Black student graduation rates between 2015 and 2022?
Compare the demographic composition of graduating cohorts between public and private universities.
### Complex Multi-Dimensional Questions
- Identify universities where Hispanic women have higher graduation rates than Hispanic men, and rank them by the size of this gender gap.
- For universities with cohorts over 3,000 students, which institution has the most equitable graduation rates across all racial groups (smallest variation)?
- Track graduation rate trends over time for underrepresented minorities and identify which universities show consistent improvement versus decline.


"""
Specialized prompts for the multi-agent intent classification system
"""

QUESTION_TYPE_ROUTER_PROMPT = """You are a question type classifier for data analysis queries.

Your job is to classify the user's question into ONE of these high-level categories:

1. SUMMARY - Overview, basic info, dataset description
2. SEARCH - Looking for specific entities, names, or text values  
3. FILTER - Finding records that meet specific conditions
4. AGGREGATE - Calculations, totals, averages, comparisons across groups
5. UNIQUE_VALUES - Listing distinct values in columns

QUESTION: "{user_question}"

Think step by step:
1. What is the user ultimately trying to accomplish?
2. Are they looking for specific records, or summary statistics?
3. Do they mention specific entities to search for?
4. Are they asking for calculations or comparisons?

Respond with exactly one word: SUMMARY, SEARCH, FILTER, AGGREGATE, or UNIQUE_VALUES

Examples:
- "What's in this dataset?" → SUMMARY
- "Show me data for Princeton University" → SEARCH  
- "Find universities with graduation rates above 80%" → FILTER
- "What's the average enrollment by year?" → AGGREGATE
- "What years are available in the data?" → UNIQUE_VALUES
"""

COLUMN_REASONER_PROMPT = """You are a column analysis specialist for CSV data.

Your job is to identify which columns are relevant to the user's question and classify their types.

QUESTION: "{user_question}"
QUESTION_TYPE: {question_type}
AVAILABLE_COLUMNS: {columns}

Analyze the question and identify:
1. Which columns are mentioned or implied
2. What type each column likely is (categorical, numeric, text)
3. What the user wants to do with each column

Respond in JSON format:
{{
    "primary_columns": ["column1", "column2"],
    "column_types": {{
        "column1": "categorical|numeric|text",
        "column2": "categorical|numeric|text"
    }},
    "reasoning": "Brief explanation of column selection"
}}

Examples:
- Q: "Show Princeton University data" → primary_columns: ["university_name"], column_types: {{"university_name": "text"}}
- Q: "Average enrollment by year" → primary_columns: ["year", "Total"], column_types: {{"year": "categorical", "Total": "numeric"}}
- Q: "Universities with high graduation rates" → primary_columns: ["university_name", graduation_rate_columns], column_types: varying
"""

TOOL_SELECTOR_PROMPT = """You are a tool selection expert for data analysis.

Your job is to select the best tool and parameters based on the question type and column analysis.

QUESTION: "{user_question}"
QUESTION_TYPE: {question_type}
COLUMN_ANALYSIS: {column_analysis}

Available tools:
1. get_info() - Dataset overview
2. get_summary_stats() - Numeric column statistics
3. search_text(column, term) - Find text in column
4. get_unique_values(column) - List distinct values
5. query_data(query) - Filter with pandas syntax
6. group_and_aggregate(group_col, agg_col, function) - Group and calculate

Select the BEST tool for this specific question type and columns.

Respond in JSON format:
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Why this tool is optimal for the question type and columns"
}}

Tool Selection Logic:
- SUMMARY → get_info() or get_summary_stats()
- SEARCH + text column → search_text(column, search_term)
- FILTER + conditions → query_data(pandas_query)
- AGGREGATE + group/agg columns → group_and_aggregate(group_col, agg_col, function)
- UNIQUE_VALUES + specific column → get_unique_values(column)
"""
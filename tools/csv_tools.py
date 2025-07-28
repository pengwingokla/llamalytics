import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os


class CSVAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.load_csv()
    
    def load_csv(self):
        """Load CSV file into pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            raise Exception(f"Error loading CSV: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "null_counts": self.df.isnull().sum().to_dict(),
            "sample_data": self.df.head(3).to_dict('records')
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for numerical columns"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"message": "No numeric columns found"}
        
        return self.df[numeric_cols].describe().to_dict()
    
    def query_data(self, query: str) -> Dict[str, Any]:
        """Execute pandas query on the data"""
        try:
            result = self.df.query(query)
            return {
                "query": query,
                "result_count": len(result),
                "data": result.head(10).to_dict('records')
            }
        except Exception as e:
            return {"error": f"Query error: {e}"}
    
    def filter_data(self, column: str, value: Any) -> Dict[str, Any]:
        """Filter data by column value"""
        try:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found"}
            
            filtered = self.df[self.df[column] == value]
            return {
                "filter": f"{column} == {value}",
                "result_count": len(filtered),
                "data": filtered.head(10).to_dict('records')
            }
        except Exception as e:
            return {"error": f"Filter error: {e}"}
    
    def group_and_aggregate(self, group_by: str, agg_column: str, agg_func: str = "mean") -> Dict[str, Any]:
        """Group data and perform aggregation"""
        try:
            if group_by not in self.df.columns:
                return {"error": f"Group column '{group_by}' not found"}
            if agg_column not in self.df.columns:
                return {"error": f"Aggregation column '{agg_column}' not found"}
            
            result = self.df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
            return {
                "operation": f"{agg_func}({agg_column}) grouped by {group_by}",
                "data": result.to_dict('records')
            }
        except Exception as e:
            return {"error": f"Aggregation error: {e}"}
    
    def get_unique_values(self, column: str) -> Dict[str, Any]:
        """Get unique values in a column"""
        try:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found"}
            
            unique_vals = self.df[column].unique()
            return {
                "column": column,
                "unique_count": len(unique_vals),
                "unique_values": unique_vals.tolist()[:20]  # Limit to first 20
            }
        except Exception as e:
            return {"error": f"Error getting unique values: {e}"}
    
    def search_text(self, column: str, search_term: str) -> Dict[str, Any]:
        """Search for text in a column"""
        try:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found"}
            
            mask = self.df[column].astype(str).str.contains(search_term, case=False, na=False)
            result = self.df[mask]
            
            return {
                "search_term": search_term,
                "column": column,
                "matches": len(result),
                "data": result.head(10).to_dict('records')
            }
        except Exception as e:
            return {"error": f"Search error: {e}"}


def create_sample_csv(file_path: str = "sample_data.csv"):
    """Create a sample CSV file for testing"""
    data = {
        'id': range(1, 101),
        'name': [f'Person_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 100),
        'salary': np.random.randint(30000, 150000, 100),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], 100),
        'years_experience': np.random.randint(0, 30, 100)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Sample CSV created: {file_path}")
    return file_path


if __name__ == "__main__":
    # Create sample data if it doesn't exist
    sample_file = "sample_data.csv"
    if not os.path.exists(sample_file):
        create_sample_csv(sample_file)
    
    # Test the analyzer
    analyzer = CSVAnalyzer(sample_file)
    
    print("=== Dataset Info ===")
    info = analyzer.get_info()
    print(json.dumps(info, indent=2, default=str))
    
    print("\n=== Summary Stats ===")
    stats = analyzer.get_summary_stats()
    print(json.dumps(stats, indent=2, default=str))
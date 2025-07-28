#!/usr/bin/env python3
"""
Comprehensive test suite for the CSV Data Analysis Workflow
Tests the complete workflow with detailed step-by-step analysis
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.workflow import create_csv_qa_workflow, ask_question
from langchain_core.messages import HumanMessage, AIMessage
from tools.csv_tools import CSVAnalyzer


class WorkflowTester:
    """Comprehensive tester for the CSV workflow"""
    
    def __init__(self, csv_file: str = "data/NJ_graduation_data.csv"):
        self.csv_file = csv_file
        self.workflow = create_csv_qa_workflow()
        
    def test_workflow_step_by_step(self, question: str) -> Dict[str, Any]:
        """Test workflow by examining message flow step by step"""
        print(f"\n{'='*80}")
        print(f"TESTING WORKFLOW STEP-BY-STEP")
        print(f"{'='*80}")
        print(f"-- Question: {question}")
        print(f"-- Dataset: {self.csv_file}")
        
        results = {}
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "csv_files": [],
            "current_csv": self.csv_file,
            "query_history": [],
            "context": ""
        }
        
        try:
            # Run the complete workflow and analyze the state evolution
            final_result = self.workflow.invoke(initial_state)
            
            print(f"\n✅ Workflow completed successfully!")
            print(f"Final state analysis:")
            print(f"   Total messages: {len(final_result.get('messages', []))}")
            print(f"   Context: {final_result.get('context', 'None')[:100]}...")
            print(f"   Query history entries: {len(final_result.get('query_history', []))}")
            
            # Analyze each message in the conversation
            messages = final_result.get('messages', [])
            
            for i, msg in enumerate(messages):
                msg_type = "User" if isinstance(msg, HumanMessage) else "AI"
                content_preview = msg.content[:100].replace('\n', ' ')
                
                print(f"\n{'-'*60}")
                print(f"MESSAGE {i}: {msg_type}")
                print(f"{'-'*60}")
                print(f"Preview: {content_preview}...")
                
                # Identify message type by content
                if isinstance(msg, HumanMessage):
                    print("🔹 Type: User Question")
                    results[f'message_{i}_user_question'] = msg.content
                    
                elif isinstance(msg, AIMessage):
                    if "Intent Analysis:" in msg.content:
                        print("🔹 Type: Intent Classification")
                        try:
                            intent_json = json.loads(msg.content.replace("Intent Analysis: ", ""))
                            print(f"   Intent: {intent_json.get('intent', 'Unknown')}")
                            print(f"   Confidence: {intent_json.get('confidence', 'Unknown')}")
                            print(f"   Tools selected: {len(intent_json.get('tools', []))}")
                            results[f'message_{i}_intent_analysis'] = intent_json
                        except Exception as e:
                            print(f"   Parse error: {e}")
                            results[f'message_{i}_intent_analysis'] = {"error": str(e)}
                    
                    elif "Analysis Results:" in msg.content:
                        print("🔹 Type: Data Analysis Results")
                        try:
                            analysis_json = json.loads(msg.content.replace("Analysis Results: ", ""))
                            print(f"   Analysis operations: {list(analysis_json.keys())}")
                            
                            # Count results
                            total_records = 0
                            for key, value in analysis_json.items():
                                if isinstance(value, dict):
                                    if 'result_count' in value:
                                        total_records += value['result_count']
                                        print(f"     {key}: {value['result_count']} records")
                                    elif 'matches' in value:
                                        print(f"     {key}: {value['matches']} matches")
                            
                            print(f"   Total records processed: {total_records}")
                            results[f'message_{i}_analysis_results'] = {
                                "operations": list(analysis_json.keys()),
                                "total_records": total_records,
                                "raw_results": analysis_json
                            }
                        except Exception as e:
                            print(f"   Parse error: {e}")
                            results[f'message_{i}_analysis_results'] = {"error": str(e)}
                    
                    elif "Loaded CSV" in msg.content or "Using default CSV" in msg.content:
                        print("🔹 Type: File Manager Output")
                        results[f'message_{i}_file_manager'] = msg.content
                    
                    else:
                        print("🔹 Type: Final Response")
                        print(f"   Response length: {len(msg.content)} characters")
                        results[f'message_{i}_final_response'] = msg.content
            
            # Analyze query history
            query_history = final_result.get('query_history', [])
            if query_history:
                print(f"\n{'-'*60}")
                print(f"QUERY HISTORY ANALYSIS")
                print(f"{'-'*60}")
                
                for i, entry in enumerate(query_history):
                    print(f"Entry {i+1}:")
                    print(f"   Question: {entry.get('question', 'Unknown')[:50]}...")
                    results_keys = list(entry.get('results', {}).keys())
                    print(f"   Results: {results_keys}")
            
            results['workflow_success'] = True
            results['final_state'] = final_result
            
        except Exception as e:
            print(f"❌ Workflow Error: {e}")
            import traceback
            traceback.print_exc()
            results['workflow_success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_full_workflow(self, question: str) -> Dict[str, Any]:
        """Test the complete workflow end-to-end"""
        print(f"\n{'='*80}")
        print(f"TESTING FULL WORKFLOW")
        print(f"{'='*80}")
        print(f"-- Question: {question}")
        
        try:
            # Test with raw results
            raw_result = ask_question(question, self.csv_file, raw_results=True)
            
            print("   Full Workflow Results:")
            print(f"   Result type: {type(raw_result)}")
            
            if isinstance(raw_result, dict):
                print(f"   Keys: {list(raw_result.keys())}")
                
                analysis = raw_result.get('analysis', 'No analysis found')
                response = raw_result.get('response', 'No response found')
                
                print(f"\n Analysis Results Preview:")
                print(f"   Length: {len(analysis)} characters")
                print(f"   Preview: {analysis[:300]}..." if len(analysis) > 300 else analysis)
                
                print(f"\n Final Response:")
                print(f"   {response}")
                
            else:
                print(f"   Direct Response: {raw_result}")
            
            return {"success": True, "result": raw_result}
            
        except Exception as e:
            print(f"❌ Full Workflow Error: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def test_csv_tools(self) -> Dict[str, Any]:
        """Test the underlying CSV tools directly"""
        print(f"\n{'='*80}")
        print(f"TESTING CSV TOOLS DIRECTLY")
        print(f"{'='*80}")
        
        results = {}
        
        try:
            analyzer = CSVAnalyzer(self.csv_file)
            
            # Test basic info
            print(f"\n{'-'*40}")
            print("BASIC INFO TEST")
            print(f"{'-'*40}")
            info = analyzer.get_info()
            results['info'] = info
            print(f"✅ Dataset shape: {info['shape']}")
            print(f"✅ Columns: {len(info['columns'])} columns")
            print(f"✅ Column names: {', '.join(info['columns'][:5])}...")
            
            # Test search
            print(f"\n{'-'*40}")
            print("SEARCH TEST")
            print(f"{'-'*40}")
            search_result = analyzer.search_text("university_name", "New Jersey Institute of Technology")
            results['search'] = search_result
            print(f"✅ Search matches: {search_result['matches']}")
            
            # Test unique values
            print(f"\n{'-'*40}")
            print("UNIQUE VALUES TEST")
            print(f"{'-'*40}")
            years = analyzer.get_unique_values("year")
            results['unique_years'] = years
            print(f"✅ Available years: {years['unique_values']}")
            
            # Test query
            print(f"\n{'-'*40}")
            print("QUERY TEST")
            print(f"{'-'*40}")
            query_result = analyzer.query_data("year == 2020")
            results['query'] = query_result
            print(f"✅ 2020 records: {query_result['result_count']}")
            
            # Test aggregation
            print(f"\n{'-'*40}")
            print("AGGREGATION TEST")
            print(f"{'-'*40}")
            agg_result = analyzer.group_and_aggregate("year", "Total", "mean")
            results['aggregation'] = agg_result

            if 'error' in agg_result:
                print(f"❌ Aggregation error: {agg_result['error']}")
            else:
                data_records = len(agg_result.get('data', []))
                print(f"✅ Average enrollment by year: {data_records} years")
                print(f"✅ Operation: {agg_result.get('operation', 'Unknown')}")

        except Exception as e:
            print(f"❌ CSV Tools Error: {e}")
            results['error'] = str(e)
        
        return results

    def run_comprehensive_tests(self):
        """Run all tests with multiple question types"""
        print(f"{'='*100}")
        print(f"COMPREHENSIVE WORKFLOW TEST SUITE")
        print(f"{'='*100}")
        
        test_questions = [
            # Simple questions
            "What years are in the graduation data?",
            "Show me New Jersey Institute of Technology data",
            "What's the overview of this dataset?",
            
            # Complex questions
            "What is the graduation rate for Asian women at New Jersey Institute of Technology in 2020?",
            "Find institutions with more than 1000 total students",
            "What is the average total enrollment by year?"
        ]
        
        all_results = {}
        
        # Test CSV tools first
        print(f"\n🔧 Testing CSV Tools...")
        all_results['csv_tools'] = self.test_csv_tools()
        
        # Test each question type
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*100}")
            print(f"TEST {i}/{len(test_questions)}: {question}")
            print(f"{'='*100}")
            
            # Test step-by-step workflow analysis
            step_results = self.test_workflow_step_by_step(question)
            
            # Test full workflow
            workflow_results = self.test_full_workflow(question)
            
            all_results[f'test_{i}'] = {
                'question': question,
                'step_results': step_results,
                'workflow_results': workflow_results
            }
        
        # Summary
        print(f"\n{'='*100}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*100}")
        
        successful_tests = 0
        total_tests = len(test_questions)
        
        for i in range(1, total_tests + 1):
            test_key = f'test_{i}'
            if test_key in all_results:
                workflow_success = all_results[test_key]['workflow_results'].get('success', False)
                step_success = all_results[test_key]['step_results'].get('workflow_success', False)
                
                overall_success = workflow_success and step_success
                status = "✅ PASSED" if overall_success else "❌ FAILED"
                print(f"Test {i}: {status} - {all_results[test_key]['question']}")
                if overall_success:
                    successful_tests += 1
        
        print(f"\nOverall Results: {successful_tests}/{total_tests} tests passed")
        
        if successful_tests == total_tests:
            print("✅  ALL TESTS PASSED! The workflow is functioning correctly.")
        else:
            print("⚠️  Some tests failed. Check the detailed output above for issues.")
        
        return all_results


def main():
    """Main test function"""
    csv_file = "data/NJ_graduation_data.csv"
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found at {csv_file}")
        print("Please ensure the NJ graduation data file is in the correct location.")
        return
    
    # Initialize tester
    tester = WorkflowTester(csv_file)
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    # Optionally save results to file
    try:
        with open('tests/test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to test_results.json")
    except Exception as e:
        print(f"⚠️  Could not save results to file: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Interactive chat interface for CSV data analysis
"""

import os
import sys
from dotenv import load_dotenv
from agents.workflow import ask_csv_question
from tools.csv_tools import CSVAnalyzer

load_dotenv()


def show_csv_info(csv_file):
    """Show basic information about the CSV file"""
    try:
        analyzer = CSVAnalyzer(csv_file)
        info = analyzer.get_info()
        
        print(f"\n📊 Dataset: {csv_file}")
        print(f"   Rows: {info['shape'][0]:,}")
        print(f"   Columns: {info['shape'][1]}")
        print(f"   Columns: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False


def interactive_chat(csv_file):
    """Interactive chat with CSV data"""
    
    print("🤖 CSV Data Analysis Agent")
    print("=" * 50)
    
    # Show CSV info
    if not show_csv_info(csv_file):
        return
    
    print("\n💡 Example questions:")
    print("   • How many rows are in this dataset?")
    print("   • What are the unique values in [column_name]?")
    print("   • What's the average [column] by [group_column]?")
    print("   • Show me summary statistics")
    print("   • Find records where [column] > [value]")
    
    print("\n" + "=" * 50)
    print("Start asking questions! (type 'quit', 'exit', or 'q' to stop)")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            question = input("\n🙋 Your question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!")
                break
            
            # Special commands
            if question.lower() == 'help':
                print("\n💡 Try asking questions like:")
                print("   • 'What universities are in this data?'")
                print("   • 'What years does this cover?'")
                print("   • 'Show me graduation rates by gender'")
                print("   • 'Which university has the highest enrollment?'")
                continue
            
            if question.lower() == 'info':
                show_csv_info(csv_file)
                continue
            
            # Process the question
            print("🤔 Thinking...")
            
            try:
                answer = ask_csv_question(question, csv_file)
                print(f"\n🤖 Answer: {answer}")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Try rephrasing your question or type 'help' for examples")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


def main():
    """Main function"""
    
    # Check if CSV file is provided
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Look for CSV files in data directory
        data_files = []
        if os.path.exists('data'):
            data_files = [f"data/{f}" for f in os.listdir('data') if f.endswith('.csv')]
        
        if not data_files:
            print("❌ No CSV files found!")
            print("Usage: python chat_with_csv.py <csv_file>")
            print("   or: place CSV files in the 'data/' directory")
            return
        
        if len(data_files) == 1:
            csv_file = data_files[0]
        else:
            print("📁 Available CSV files:")
            for i, file in enumerate(data_files, 1):
                print(f"   {i}. {file}")
            
            try:
                choice = int(input("\nSelect file (number): ")) - 1
                csv_file = data_files[choice]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    # Check if Ollama is available
    try:
        import ollama
        ollama.list()
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("Please make sure Ollama is running and llama3.2 is pulled")
        return
    
    # Start interactive chat
    interactive_chat(csv_file)


if __name__ == "__main__":
    main()
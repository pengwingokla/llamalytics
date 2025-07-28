#!/usr/bin/env python3
"""
Test script to demonstrate agent outputs
"""

import json
from agents.multi_agent_system import ask_multi_agent_question


def print_agent_outputs(result):
    """Pretty print agent outputs"""
    print("🤖 AGENT OUTPUTS:")
    print("=" * 60)
    
    for agent_name, output in result.get('agent_outputs', {}).items():
        print(f"\n📋 {agent_name.replace('_', ' ').title()}:")
        print("-" * 40)
        
        for key, value in output.items():
            if key != 'agent':
                if isinstance(value, list):
                    if len(value) > 5:
                        print(f"  {key}: {value[:5]}... ({len(value)} total)")
                    else:
                        print(f"  {key}: {value}")
                elif isinstance(value, str) and len(value) > 150:
                    print(f"  {key}: {value[:150]}...")
                elif isinstance(value, dict):
                    print(f"  {key}: {json.dumps(value, indent=4)}")
                else:
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print(f"🤖 FINAL ANSWER: {result['answer']}")


def main():
    """Test the multi-agent system with detailed outputs"""
    
    csv_file = "data/NJ_graduation_data.csv"
    question = "How many universities are in this dataset?"
    
    print(f"❓ Question: {question}")
    print(f"📊 Dataset: {csv_file}")
    print()
    
    try:
        result = ask_multi_agent_question(question, csv_file)
        
        print(f"🔗 Agent Chain: {result['agent_chain']}")
        print()
        
        print_agent_outputs(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
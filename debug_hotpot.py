#!/usr/bin/env python3
"""
Debug script to examine HotpotQA data structure
"""

from datasets import load_dataset
import traceback

print("Loading HotpotQA to debug data structure...")

try:
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)
    print(f"Dataset loaded successfully. Length: {len(dataset)}")
    
    # Examine first few examples in detail
    for i in range(min(2, len(dataset))):
        print(f"\n{'='*50}")
        print(f"EXAMPLE {i}")
        print(f"{'='*50}")
        
        try:
            example = dataset[i]
            print(f"Example type: {type(example)}")
            print(f"Keys: {list(example.keys())}")
            
            for key in example.keys():
                try:
                    value = example[key]
                    print(f"{key}: {type(value)} = {value}")
                except Exception as e:
                    print(f"Error accessing {key}: {e}")
            
            # Test the specific logic from extract_chain.py
            print(f"\n--- Testing extraction logic ---")
            question = example['question']
            answer = example['answer']
            supporting_facts = example.get('supporting_facts', {})
            
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Supporting facts: {supporting_facts}")
            print(f"Supporting facts type: {type(supporting_facts)}")
            
            if supporting_facts and 'title' in supporting_facts:
                fact_titles = supporting_facts['title']
                print(f"Fact titles: {fact_titles}")
            else:
                fact_titles = []
                print(f"No fact titles found")
            
            print(f"Number of fact titles: {len(fact_titles)}")
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            traceback.print_exc()
        
        if i == 0:  # Just do first example for now
            break

except Exception as e:
    print(f"Error loading dataset: {e}")
    traceback.print_exc() 
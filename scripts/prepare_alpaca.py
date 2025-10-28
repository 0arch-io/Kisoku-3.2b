#!/usr/bin/env python3
"""
Convert Alpaca dataset to simple text format for MaxText
Each example becomes a single line of text in instruction-following format
"""
from datasets import load_dataset
import json

print("Loading Alpaca dataset from HuggingFace...")
dataset = load_dataset("tatsu-lab/alpaca")

print(f"Dataset loaded: {len(dataset['train'])} examples")
print("\nFirst example:")
print(dataset['train'][0])

# Create formatted text file
output_file = "/tmp/alpaca_text.jsonl"
print(f"\nConverting to JSONL format at {output_file}...")

with open(output_file, 'w') as f:
    for example in dataset['train']:
        instruction = example['instruction']
        input_text = example['input']
        output_text = example['output']

        # Format as instruction-following prompt
        if input_text:
            prompt = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n{output_text}"
        else:
            prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output_text}"

        # Write as JSONL with "text" field
        json_line = json.dumps({"text": prompt})
        f.write(json_line + '\\n')

print(f"✓ Converted {len(dataset['train'])} examples")
print(f"✓ Saved to: {output_file}")
print("\nSample formatted text:")
with open(output_file, 'r') as f:
    print(json.loads(f.readline())['text'][:500])

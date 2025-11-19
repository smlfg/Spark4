#!/usr/bin/env python3
"""
Generate dummy training data for testing
Creates a JSONL file with 50 instruction-input-output entries
"""

import json
import random

def generate_dummy_data(output_file="dummy.jsonl", num_samples=50):
    """Generate dummy training data in instruction format"""

    # Templates for diverse training examples
    templates = [
        {
            "instruction": "Translate the following English text to German:",
            "input": "Hello, how are you today?",
            "output": "Hallo, wie geht es dir heute?"
        },
        {
            "instruction": "Summarize this text in one sentence:",
            "input": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make decisions.",
            "output": "Machine learning allows computers to learn from data and make decisions using pattern recognition algorithms."
        },
        {
            "instruction": "Write a Python function to calculate factorial:",
            "input": "",
            "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "instruction": "Explain this concept in simple terms:",
            "input": "Neural networks",
            "output": "Neural networks are computer systems inspired by the human brain, consisting of connected nodes that process information and learn patterns from examples."
        },
        {
            "instruction": "Fix the bug in this code:",
            "input": "def add(a, b):\n    return a + c",
            "output": "def add(a, b):\n    return a + b  # Changed 'c' to 'b'"
        },
        {
            "instruction": "Generate a creative story opening:",
            "input": "Topic: A mysterious door",
            "output": "The door had always been there, nestled between the oak trees at the edge of the forest, yet no one could remember when it first appeared."
        },
        {
            "instruction": "Answer this math question:",
            "input": "What is 15% of 200?",
            "output": "15% of 200 is 30. (Calculation: 200 × 0.15 = 30)"
        },
        {
            "instruction": "Classify the sentiment of this review:",
            "input": "This product exceeded my expectations! Great quality and fast delivery.",
            "output": "Positive sentiment - The review expresses satisfaction with quality and delivery speed."
        },
        {
            "instruction": "Convert this to JSON format:",
            "input": "Name: John, Age: 30, City: Berlin",
            "output": '{"name": "John", "age": 30, "city": "Berlin"}'
        },
        {
            "instruction": "Explain the difference between:",
            "input": "List vs Tuple in Python",
            "output": "Lists are mutable (can be changed) and use square brackets [], while tuples are immutable (cannot be changed) and use parentheses ()."
        }
    ]

    data = []

    # Generate samples by cycling through templates with variations
    for i in range(num_samples):
        template = templates[i % len(templates)]

        # Add some variation to avoid exact duplicates
        sample = {
            "instruction": template["instruction"],
            "input": template["input"],
            "output": template["output"]
        }

        # Add sample number for uniqueness
        if i >= len(templates):
            sample["instruction"] = f"{template['instruction']} (Example {i + 1})"

        data.append(sample)

    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ Generated {num_samples} samples in '{output_file}'")
    print(f"📊 File size: {len(data)} entries")
    print(f"📝 Sample entry:")
    print(f"   Instruction: {data[0]['instruction'][:50]}...")
    print(f"   Input: {data[0]['input'][:50]}...")
    print(f"   Output: {data[0]['output'][:50]}...")

if __name__ == "__main__":
    generate_dummy_data()

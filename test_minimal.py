#!/usr/bin/env python3
"""
Minimal LoRA Adapter Test Script

1. Load base model
2. Load trained adapter
3. Test prompt: "Explain LoRA:"
4. Print: Before vs After
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate text from model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # Configuration
    BASE_MODEL = "gpt2"  # Replace with your base model path/name
    ADAPTER_PATH = "./lora_adapter"  # Replace with your adapter path
    TEST_PROMPT = "Explain LoRA:"

    print("="*80)
    print("LoRA Adapter Test - Before vs After")
    print("="*80)
    print(f"\nBase Model: {BASE_MODEL}")
    print(f"Adapter Path: {ADAPTER_PATH}")
    print(f"Test Prompt: '{TEST_PROMPT}'")
    print("="*80)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("[2/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Generate with base model (BEFORE)
    print("[3/4] Generating with base model (BEFORE)...")
    base_output = generate_response(base_model, tokenizer, TEST_PROMPT)

    # Load adapter
    print("[4/4] Loading LoRA adapter and generating (AFTER)...")
    try:
        model_with_adapter = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        adapter_output = generate_response(model_with_adapter, tokenizer, TEST_PROMPT)
        adapter_loaded = True
    except Exception as e:
        print(f"Warning: Could not load adapter from {ADAPTER_PATH}")
        print(f"Error: {e}")
        adapter_output = "N/A - Adapter not found"
        adapter_loaded = False

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\n📋 BEFORE (Base Model):")
    print("-"*80)
    print(base_output)

    print("\n📊 AFTER (With LoRA Adapter):")
    print("-"*80)
    print(adapter_output)

    print("\n" + "="*80)

    if adapter_loaded:
        print("✅ Test completed successfully!")
    else:
        print("⚠️  Test completed, but adapter could not be loaded.")
        print("   Please ensure adapter is trained and saved to:", ADAPTER_PATH)

    print("="*80)


if __name__ == "__main__":
    main()

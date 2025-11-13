#!/usr/bin/env python3
"""
Setup script to add Llama-2 chat template to base tokenizer.
This fixes the "chat template not set" error in MaxText SFT training.
"""
import sys
from transformers import AutoTokenizer

def main():
    print("Loading base tokenizer: NousResearch/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    # Add Llama-2 instruct chat template
    # Format: [INST] user message [/INST] assistant response
    llama2_chat_template = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' }}{% endif %}{% endfor %}"""

    tokenizer.chat_template = llama2_chat_template

    # Save to local directory for MaxText to use
    output_dir = "/tmp/llama2_tokenizer_with_chat_template"
    print(f"Saving tokenizer with chat template to: {output_dir}")
    tokenizer.save_pretrained(output_dir)

    # Verify the chat template is set
    print("\nVerifying chat template...")
    test_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    if test_tokenizer.chat_template:
        print("✓ Chat template successfully set!")

        # Test it with a sample conversation
        test_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        try:
            result = test_tokenizer.apply_chat_template(test_messages, tokenize=False)
            print(f"\nTest conversation formatted as:\n{result}")
            return 0
        except Exception as e:
            print(f"✗ Error testing chat template: {e}")
            return 1
    else:
        print("✗ Chat template not set!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

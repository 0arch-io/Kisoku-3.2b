# Kisoku Training Guide - Complete Documentation

## Table of Contents
1. [The Critical Fix - Chat Template Solution](#the-critical-fix)
2. [What Went Wrong - Failed Approaches](#what-went-wrong)
3. [Working Setup - Current Configuration](#working-setup)
4. [Path B Pipeline - Proper Multi-Stage Training](#path-b-pipeline)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Quick Reference Commands](#quick-reference-commands)

---

## The Critical Fix - Chat Template Solution

### Root Cause (Weeks of Failures)
**Problem**: `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`

**Why This Happened**:
- Using `NousResearch/Llama-2-7b-hf` which is a BASE model tokenizer
- Base tokenizers DO NOT have the `chat_template` attribute
- MaxText's SFT mode with `sft_train_on_completion_only=True` internally calls `tokenizer.apply_chat_template()`
- This method REQUIRES the chat_template attribute to be set

**The Solution**:
Create a Python script that programmatically adds the Llama-2 chat template to the base tokenizer.

### Setup Script: `/tmp/setup_tokenizer_with_chat_template.py`

```python
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
```

### Working Launch Script: `/tmp/launch_ultrachat_WITH_CHAT_TEMPLATE.sh`

```bash
#!/bin/bash
set -e

# Activate virtual environment properly
cd ~/maxtext
source ~/maxtext_py312/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/josephrodriguez/maxtext:/home/josephrodriguez/maxtext/src:${PYTHONPATH}"

echo "Setting up tokenizer with Llama-2 chat template..."
# Run the tokenizer setup script
python3 /tmp/setup_tokenizer_with_chat_template.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to setup tokenizer with chat template"
    exit 1
fi

echo "Tokenizer setup complete. Starting training..."

# Kill any existing training processes
pkill -f train.py || true
sleep 2

# Launch training WITH SFT mode using the tokenizer that has chat template set
nohup python3 -m MaxText.train \
  ~/maxtext/src/MaxText/configs/base.yml \
  use_sft=True \
  sft_train_on_completion_only=True \
  run_name=kisoku-ultrachat-sft-CHAT-TEMPLATE \
  base_output_directory=gs://pantheon-tpu-training/kisoku-ultrachat-sft-CHAT-TEMPLATE/ \
  dataset_type=hf \
  hf_path=HuggingFaceH4/ultrachat_200k \
  train_split=train_sft \
  hf_eval_split=test_sft \
  train_data_columns="['messages']" \
  eval_data_columns="['messages']" \
  tokenizer_path=/tmp/llama2_tokenizer_with_chat_template \
  load_parameters_path=gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-GCS/checkpoints/99999/items \
  per_device_batch_size=4.0 \
  max_target_length=2048 \
  steps=5000 \
  checkpoint_period=500 \
  ici_fsdp_parallelism=16 \
  learning_rate=3e-5 \
  adam_b1=0.9 \
  adam_b2=0.95 \
  adam_eps=1e-8 \
  adam_eps_root=0.0 \
  opt_type=adamw \
  adam_weight_decay=0.1 \
  warmup_steps_fraction=0.1 \
  cosine_learning_rate_final_fraction=0.1 \
  learning_rate_schedule_steps=5000 \
  log_period=10 \
  base_emb_dim=3072 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_mlp_dim=8192 \
  base_num_decoder_layers=32 \
  head_dim=96 \
  vocab_size=50304 \
  decoder_block=llama2 \
  dtype=bfloat16 \
  > ~/kisoku_ultrachat_sft_chat_template.log 2>&1 &

echo "Worker PID: $!"
echo "Training launched. Check ~/kisoku_ultrachat_sft_chat_template.log for progress."
```

### Required Dependency: jinja2
```bash
# Install on all TPU workers
pip install jinja2
```

### Expected Output
```
Setting up tokenizer with Llama-2 chat template...
Loading base tokenizer: NousResearch/Llama-2-7b-hf
Saving tokenizer with chat template to: /tmp/llama2_tokenizer_with_chat_template

Verifying chat template...
✓ Chat template successfully set!

Test conversation formatted as:
[INST] Hello, how are you? [/INST] I'm doing well, thank you!

Tokenizer setup complete. Starting training...
Worker PID: 268732
Training launched. Check ~/kisoku_ultrachat_sft_chat_template.log for progress.
```

---

## What Went Wrong - Failed Approaches

### Failed Approach 1: Using Base Tokenizer Directly
```bash
tokenizer_path=NousResearch/Llama-2-7b-hf  # ❌ No chat template
use_sft=True
sft_train_on_completion_only=True
```
**Error**: `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`

### Failed Approach 2: Trying Instruct Model (Wrong Architecture)
```bash
tokenizer_path=meta-llama/Llama-2-7b-chat-hf  # ❌ Different vocab size
```
**Error**: Vocabulary size mismatch (32000 vs 50304)

### Failed Approach 3: SFT Without Completion-Only
```bash
use_sft=True
# Missing: sft_train_on_completion_only=True
```
**Result**: Model learns from full conversation including user prompts, not just assistant responses

### Failed Approach 4: No SFT Mode
```bash
# No use_sft flag
dataset_type=hf
hf_path=HuggingFaceH4/ultrachat_200k
```
**Result**: Training works but doesn't properly handle conversational format

### What Finally Worked
✅ **Programmatically add chat template to base tokenizer**
✅ **Save modified tokenizer locally**
✅ **Point MaxText to modified tokenizer path**

---

## Working Setup - Current Configuration

### TPU Configuration
- **Pod**: kisoku-sft
- **Zone**: us-central2-b
- **Project**: pantheon-tpu
- **Type**: TPU v4-32 (4 workers, 16 total chips)
- **Python**: 3.12 with virtual environment at ~/maxtext_py312

### Current Running Training
- **Run Name**: kisoku-ultrachat-sft-CHAT-TEMPLATE
- **Base Model**: Kisoku-3.2b (checkpoint 99999)
- **Dataset**: HuggingFaceH4/ultrachat_200k (train_sft split)
- **Mode**: SFT with completion-only training
- **Output**: gs://pantheon-tpu-training/kisoku-ultrachat-sft-CHAT-TEMPLATE/

### Training Hyperparameters
```yaml
per_device_batch_size: 4.0
max_target_length: 2048
steps: 5000
checkpoint_period: 500
ici_fsdp_parallelism: 16

# Optimizer
opt_type: adamw
learning_rate: 3e-5
adam_b1: 0.9
adam_b2: 0.95
adam_eps: 1e-8
adam_weight_decay: 0.1

# Schedule
warmup_steps_fraction: 0.1
cosine_learning_rate_final_fraction: 0.1
learning_rate_schedule_steps: 5000

# Model Architecture (Kisoku 3.2B)
base_emb_dim: 3072
base_num_query_heads: 32
base_num_kv_heads: 8
base_mlp_dim: 8192
base_num_decoder_layers: 32
head_dim: 96
vocab_size: 50304
decoder_block: llama2
dtype: bfloat16
```

### Performance Metrics
- **Throughput**: ~106 TFLOP/s/device
- **Token Speed**: ~5000 tokens/s/device
- **Loss**: Starting ~8.1, decreasing to ~6.0 (healthy)

### How to Check Training Status
```bash
# Check training on worker 0
timeout 20 gcloud compute tpus tpu-vm ssh kisoku-sft \
  --zone=us-central2-b \
  --worker=0 \
  --project=pantheon-tpu \
  --command="tail -50 ~/kisoku_ultrachat_sft_chat_template.log | grep -E 'step|loss|TFLOP|tokens'"

# Check processes on all workers
for w in 0 1 2 3; do
  echo "=== Worker $w ==="
  timeout 15 gcloud compute tpus tpu-vm ssh kisoku-sft \
    --zone=us-central2-b \
    --worker=$w \
    --project=pantheon-tpu \
    --command="ps aux | grep train.py | grep -v grep"
done
```

---

## Path B Pipeline - Proper Multi-Stage Training

### Overview
To build a truly high-quality, fully-owned 3B model:
1. **Stage 1**: Continued Pretraining (50-100B tokens)
2. **Stage 2**: Proper SFT with diverse datasets
3. **Stage 3**: Evaluation and iteration
4. **Stage 4** (Optional): DPO alignment

### Current Situation
- **Kisoku Base**: Only pretrained on 100K samples from DCLM (~200M tokens)
- **Industry Standard**: 3B models need 2-6 trillion tokens
- **Gap**: 10,000x less pretraining than needed
- **Solution**: Continued pretraining before SFT

### Stage 1: Continued Pretraining (1-2 weeks, ~$3-5K)

#### Dataset Selection
Use high-quality, diverse pretraining data:
- **FineWeb-Edu** (1.3T tokens, filtered for educational content)
- **DCLM-Baseline** (continuation from Kisoku's original training)
- **The Stack v2** (code, ~600B tokens)
- **Target**: 50-100B tokens total

#### Configuration for Continued Pretraining
```bash
#!/bin/bash
# /tmp/launch_continued_pretraining.sh

cd ~/maxtext
source ~/maxtext_py312/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/josephrodriguez/maxtext:/home/josephrodriguez/maxtext/src:${PYTHONPATH}"

pkill -f train.py || true
sleep 2

nohup python3 -m MaxText.train \
  ~/maxtext/src/MaxText/configs/base.yml \
  run_name=kisoku-continued-pretrain-v1 \
  base_output_directory=gs://pantheon-tpu-training/kisoku-continued-pretrain/ \
  dataset_type=hf \
  hf_path=HuggingFaceFW/fineweb-edu \
  hf_data_dir=sample-10BT \
  train_split=train \
  tokenizer_path=NousResearch/Llama-2-7b-hf \
  load_parameters_path=gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-GCS/checkpoints/99999/items \
  per_device_batch_size=8.0 \
  max_target_length=4096 \
  steps=50000 \
  checkpoint_period=2500 \
  ici_fsdp_parallelism=16 \
  learning_rate=1e-4 \
  adam_b1=0.9 \
  adam_b2=0.95 \
  adam_eps=1e-8 \
  opt_type=adamw \
  adam_weight_decay=0.1 \
  warmup_steps_fraction=0.01 \
  cosine_learning_rate_final_fraction=0.1 \
  learning_rate_schedule_steps=50000 \
  log_period=10 \
  base_emb_dim=3072 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_mlp_dim=8192 \
  base_num_decoder_layers=32 \
  head_dim=96 \
  vocab_size=50304 \
  decoder_block=llama2 \
  dtype=bfloat16 \
  > ~/kisoku_continued_pretrain.log 2>&1 &

echo "Continued pretraining launched. Worker PID: $!"
```

#### Key Differences from SFT
- **NO** `use_sft=True` flag
- Higher learning rate (1e-4 vs 3e-5)
- Longer sequences (4096 vs 2048)
- More steps (50000 vs 5000)
- Different dataset (pretraining corpus vs conversational)

#### Token Count Calculation
```
per_device_batch_size: 8
max_target_length: 4096
num_devices: 16 (TPU v4-32)
steps: 50000

Total tokens = 8 × 4096 × 16 × 50000 = 26.2B tokens per run

For 100B tokens: Run ~4 times with different dataset splits
```

### Stage 2: Proper SFT (3-5 days, ~$2-3K)

#### Dataset Mix (Target: 500K samples)
Diverse, high-quality instruction data:
- **OpenHermes-2.5** (1M samples) → 150K samples
- **FLAN Collection** (chain-of-thought) → 100K samples
- **MathInstruct** (math reasoning) → 50K samples
- **Magicoder** (code instruction) → 100K samples
- **UltraChat 200k** (general conversation) → 100K samples

#### Why This Mix?
- **General knowledge**: OpenHermes, UltraChat
- **Reasoning**: FLAN, MathInstruct
- **Coding**: Magicoder
- **Diversity**: Multiple formats and domains
- **Quality**: All are curated/filtered datasets

#### Configuration for Proper SFT
```bash
#!/bin/bash
# /tmp/launch_proper_sft.sh

cd ~/maxtext
source ~/maxtext_py312/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/josephrodriguez/maxtext:/home/josephrodriguez/maxtext/src:${PYTHONPATH}"

# Setup chat template
python3 /tmp/setup_tokenizer_with_chat_template.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to setup tokenizer with chat template"
    exit 1
fi

pkill -f train.py || true
sleep 2

nohup python3 -m MaxText.train \
  ~/maxtext/src/MaxText/configs/base.yml \
  use_sft=True \
  sft_train_on_completion_only=True \
  run_name=kisoku-proper-sft-v1 \
  base_output_directory=gs://pantheon-tpu-training/kisoku-proper-sft/ \
  dataset_type=hf \
  hf_path=teknium/OpenHermes-2.5 \
  train_split=train \
  train_data_columns="['conversations']" \
  tokenizer_path=/tmp/llama2_tokenizer_with_chat_template \
  load_parameters_path=gs://pantheon-tpu-training/kisoku-continued-pretrain/checkpoints/XXXXX/items \
  per_device_batch_size=4.0 \
  max_target_length=2048 \
  steps=15000 \
  checkpoint_period=1000 \
  eval_interval=1000 \
  ici_fsdp_parallelism=16 \
  learning_rate=2e-5 \
  adam_b1=0.9 \
  adam_b2=0.95 \
  adam_eps=1e-8 \
  opt_type=adamw \
  adam_weight_decay=0.1 \
  warmup_steps_fraction=0.05 \
  cosine_learning_rate_final_fraction=0.1 \
  learning_rate_schedule_steps=15000 \
  log_period=10 \
  base_emb_dim=3072 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_mlp_dim=8192 \
  base_num_decoder_layers=32 \
  head_dim=96 \
  vocab_size=50304 \
  decoder_block=llama2 \
  dtype=bfloat16 \
  > ~/kisoku_proper_sft.log 2>&1 &

echo "Proper SFT launched. Worker PID: $!"
```

**Note**: You'll need to create a combined dataset or train on each dataset sequentially. Consider using dataset mixing utilities.

### Stage 3: Evaluation Benchmarks

#### Key Benchmarks to Track
1. **General Knowledge**: MMLU (Massive Multitask Language Understanding)
2. **Math Reasoning**: GSM8K (Grade School Math)
3. **Instruction Following**: IFEval
4. **Conversational**: MT-Bench
5. **Coding**: HumanEval

#### Evaluation Setup
```bash
# Install lm-evaluation-harness
pip install lm-eval

# Run MMLU
lm_eval --model hf \
  --model_args pretrained=gs://pantheon-tpu-training/kisoku-proper-sft/checkpoints/XXXXX \
  --tasks mmlu \
  --batch_size 8 \
  --output_path ./eval_results/

# Run GSM8K
lm_eval --model hf \
  --model_args pretrained=gs://pantheon-tpu-training/kisoku-proper-sft/checkpoints/XXXXX \
  --tasks gsm8k \
  --batch_size 8 \
  --output_path ./eval_results/
```

#### Target Scores (for 3B model)
- **MMLU**: 55-60%
- **GSM8K**: 35-45%
- **HumanEval**: 25-35%
- **MT-Bench**: 6.5-7.5 / 10

### Stage 4: Optional DPO Alignment

After SFT, optionally train with Direct Preference Optimization for better alignment:
- **Dataset**: UltraFeedback, HelpSteer
- **Steps**: 2000-5000
- **Learning rate**: 5e-7 (very low)

---

## Troubleshooting Guide

### Error: Chat Template Not Set
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
```
**Solution**: Use the tokenizer setup script before training:
```bash
python3 /tmp/setup_tokenizer_with_chat_template.py
# Then point training to: tokenizer_path=/tmp/llama2_tokenizer_with_chat_template
```

### Error: jinja2 Not Installed
```
apply_chat_template requires jinja2 to be installed
```
**Solution**: Install on all workers:
```bash
for w in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh kisoku-sft \
    --zone=us-central2-b \
    --worker=$w \
    --project=pantheon-tpu \
    --command="pip install jinja2"
done
```

### Error: Vocabulary Size Mismatch
```
ValueError: Loaded vocab size (32000) doesn't match config (50304)
```
**Solution**: Ensure tokenizer matches model architecture. Kisoku uses vocab_size=50304 from NousResearch/Llama-2-7b-hf.

### Training Hangs at "Compiling train_step"
**Symptoms**: Training stuck for 10+ minutes with no progress
**Cause**: JAX XLA compilation (normal for first run)
**Solution**: Wait 15-20 minutes. Subsequent steps will be fast.

### Loss Not Decreasing
**Symptoms**: Loss stays constant or increases
**Possible Causes**:
1. Learning rate too high → Reduce by 10x
2. Bad data format → Check with `train_data_columns` matches dataset
3. Sequence too long → Try `max_target_length=1024`

### Out of Memory (OOM)
**Symptoms**: Training crashes with "Out of memory" error
**Solutions**:
1. Reduce `per_device_batch_size` (try 2.0)
2. Reduce `max_target_length` (try 1024)
3. Increase `ici_fsdp_parallelism` if not maxed

### Cannot Connect to TPU
```
Error: Could not reach ssh target
```
**Solutions**:
1. Check TPU is running: `gcloud compute tpus tpu-vm list --zone=us-central2-b`
2. TPU might be stopped: `gcloud compute tpus tpu-vm start kisoku-sft --zone=us-central2-b`
3. Firewall issue: Check VPC firewall rules allow SSH

---

## Quick Reference Commands

### Check TPU Status
```bash
# List all TPUs
gcloud compute tpus tpu-vm list --zone=us-central2-b --project=pantheon-tpu

# Describe specific TPU
gcloud compute tpus tpu-vm describe kisoku-sft --zone=us-central2-b --project=pantheon-tpu
```

### Upload Files to All Workers
```bash
for w in 0 1 2 3; do
  gcloud compute tpus tpu-vm scp /tmp/YOUR_FILE.sh kisoku-sft:/tmp/ \
    --zone=us-central2-b \
    --worker=$w \
    --project=pantheon-tpu
done
```

### Launch Training on All Workers
```bash
for w in 0 1 2 3; do
  echo "=== Launching on worker $w ==="
  gcloud compute tpus tpu-vm ssh kisoku-sft \
    --zone=us-central2-b \
    --worker=$w \
    --project=pantheon-tpu \
    --command="chmod +x /tmp/YOUR_SCRIPT.sh && nohup /tmp/YOUR_SCRIPT.sh > /tmp/training.log 2>&1 & echo 'Worker $w started, PID: '\$!"
done
```

### Check Training Progress
```bash
# Quick check on worker 0
timeout 20 gcloud compute tpus tpu-vm ssh kisoku-sft \
  --zone=us-central2-b \
  --worker=0 \
  --project=pantheon-tpu \
  --command="tail -50 ~/kisoku_ultrachat_sft_chat_template.log | grep -E 'step|loss|TFLOP'"

# Monitor continuously (run locally)
watch -n 30 'timeout 20 gcloud compute tpus tpu-vm ssh kisoku-sft --zone=us-central2-b --worker=0 --project=pantheon-tpu --command="tail -20 ~/training.log"'
```

### Kill Training Processes
```bash
for w in 0 1 2 3; do
  echo "=== Killing on worker $w ==="
  gcloud compute tpus tpu-vm ssh kisoku-sft \
    --zone=us-central2-b \
    --worker=$w \
    --project=pantheon-tpu \
    --command="pkill -f train.py && echo 'Killed worker $w'"
done
```

### Check GCS Checkpoints
```bash
# List checkpoints
gsutil ls gs://pantheon-tpu-training/kisoku-ultrachat-sft-CHAT-TEMPLATE/checkpoints/

# Check size
gsutil du -sh gs://pantheon-tpu-training/kisoku-ultrachat-sft-CHAT-TEMPLATE/

# Download specific checkpoint
gsutil -m cp -r gs://pantheon-tpu-training/kisoku-ultrachat-sft-CHAT-TEMPLATE/checkpoints/5000/ ./local_checkpoint/
```

### Install Dependencies on All Workers
```bash
for w in 0 1 2 3; do
  echo "=== Installing on worker $w ==="
  gcloud compute tpus tpu-vm ssh kisoku-sft \
    --zone=us-central2-b \
    --worker=$w \
    --project=pantheon-tpu \
    --command="pip install jinja2 transformers[torch]"
done
```

---

## Timeline and Cost Estimates (Path B)

### Stage 1: Continued Pretraining
- **Duration**: 1-2 weeks
- **Compute**: TPU v4-32 @ ~$8/hour
- **Cost**: ~$1,344 - $2,688
- **Output**: Kisoku with 50-100B additional tokens

### Stage 2: Proper SFT
- **Duration**: 3-5 days
- **Compute**: TPU v4-32 @ ~$8/hour
- **Cost**: ~$576 - $960
- **Output**: Instruction-tuned model ready for use

### Stage 3: Evaluation
- **Duration**: 1-2 days
- **Compute**: Can use smaller instance (TPU v4-8)
- **Cost**: ~$100 - $200
- **Output**: Benchmark scores and comparison

### Stage 4: Optional DPO
- **Duration**: 1-2 days
- **Compute**: TPU v4-32 @ ~$8/hour
- **Cost**: ~$192 - $384
- **Output**: Aligned model with better instruction following

### Total
- **Duration**: 3-4 weeks
- **Total Cost**: ~$2,212 - $4,232
- **Result**: High-quality, fully-owned 3B model competitive with industry standards

---

## Key Learnings

1. **Base vs Instruct Tokenizers**: Base tokenizers lack chat_template. You must add it programmatically for SFT with completion-only training.

2. **Pretraining Matters Most**: No amount of instruction tuning can compensate for insufficient pretraining. 100K samples is 10,000x too little.

3. **Dataset Diversity**: Single-source SFT (UltraChat only) produces brittle models. Mix 4-5 diverse high-quality sources.

4. **Evaluation is Critical**: Without benchmarks, you can't tell if training is working. Set up eval from the start.

5. **Patience with JAX**: First compilation takes 15-20 minutes. Don't panic.

6. **Chat Template Format**: Llama-2 uses `[INST] user [/INST] assistant` format. Different models use different formats.

7. **Completion-Only Training**: Essential for SFT. Model should only predict assistant responses, not user prompts.

8. **Documentation Saves Time**: This guide saves weeks of debugging. Update it as you learn.

---

## Repository and Backup

Save this guide to multiple locations:
1. `/tmp/KISOKU_TRAINING_GUIDE.md` (local)
2. Commit to Kisoku GitHub repo: https://github.com/0arch-io/Kisoku-3.2b
3. Upload to GCS: `gs://pantheon-tpu-training/docs/`
4. Keep a local backup on your machine

```bash
# Commit to GitHub
cd ~/Kisoku-3.2b
git add docs/KISOKU_TRAINING_GUIDE.md
git commit -m "Add comprehensive training guide with chat template fix"
git push origin main

# Upload to GCS
gsutil cp /tmp/KISOKU_TRAINING_GUIDE.md gs://pantheon-tpu-training/docs/

# Download to local machine
gcloud compute tpus tpu-vm scp kisoku-sft:/tmp/KISOKU_TRAINING_GUIDE.md ~/Downloads/ \
  --zone=us-central2-b \
  --worker=0 \
  --project=pantheon-tpu
```

---

## Final Notes

This guide represents weeks of debugging and learning. The critical breakthrough was understanding that:
- **The problem wasn't MaxText or the training setup**
- **The problem was the tokenizer missing a required attribute**
- **The solution was to add that attribute programmatically**

This same pattern applies to many ML debugging scenarios: the error message points to a symptom, not the root cause. Always trace back to fundamentals.

Good luck with Path B. You're on track to build a truly high-quality, fully-owned model.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
**Author**: Documented from weeks of TPU training debugging
**Model**: Kisoku-3.2b (0arch-io)
**Framework**: MaxText (Google JAX)

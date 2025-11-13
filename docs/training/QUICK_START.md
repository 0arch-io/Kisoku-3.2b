# Quick Start - Kisoku Training

## The Critical Fix (What Took Weeks to Figure Out)

**Problem**: `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`

**Solution**: Base tokenizers lack chat_template. Add it programmatically:

```bash
# 1. Run this script first (saved at /tmp/setup_tokenizer_with_chat_template.py)
python3 /tmp/setup_tokenizer_with_chat_template.py

# 2. Point training to modified tokenizer
tokenizer_path=/tmp/llama2_tokenizer_with_chat_template
```

## Currently Running Training (SUCCESS!)

- **TPU**: kisoku-sft (v4-32, us-central2-b)
- **Run**: kisoku-ultrachat-sft-CHAT-TEMPLATE
- **Dataset**: HuggingFaceH4/ultrachat_200k
- **Mode**: SFT with completion-only training
- **Status**: Training successfully, loss decreasing (~8.1 → ~6.0)
- **Performance**: ~106 TFLOP/s/device, ~5000 tokens/s/device

## Check Training Status

```bash
timeout 20 gcloud compute tpus tpu-vm ssh kisoku-sft \
  --zone=us-central2-b \
  --worker=0 \
  --project=pantheon-tpu \
  --command="tail -50 ~/kisoku_ultrachat_sft_chat_template.log | grep -E 'step|loss|TFLOP'"
```

## Path Forward (Your Choice)

You chose **Option B**: Stop and do proper multi-stage training

### Stage 1: Continued Pretraining (1-2 weeks, $3-5K)
- Add 50-100B tokens to Kisoku's 200M base
- Use FineWeb-Edu, DCLM, The Stack v2
- Script ready at: `/tmp/launch_continued_pretraining.sh` (in full guide)

### Stage 2: Proper SFT (3-5 days, $2-3K)
- Mix 500K samples from diverse sources:
  - OpenHermes-2.5 (150K)
  - FLAN (100K)
  - MathInstruct (50K)
  - Magicoder (100K)
  - UltraChat (100K)
- Script ready in full guide

### Stage 3: Evaluation
- MMLU, GSM8K, HumanEval, MT-Bench
- Target: 55-60% MMLU, 35-45% GSM8K for 3B model

## Full Documentation

**Complete guide**: `/tmp/KISOKU_TRAINING_GUIDE.md` (23KB, 20+ pages)

**Backup locations**:
- GCS: `gs://pantheon-tpu-training/docs/KISOKU_TRAINING_GUIDE.md`
- All TPU workers: `~/KISOKU_TRAINING_GUIDE.md`

## Key Files

1. `/tmp/setup_tokenizer_with_chat_template.py` - THE FIX
2. `/tmp/launch_ultrachat_WITH_CHAT_TEMPLATE.sh` - Working launch script
3. `/tmp/KISOKU_TRAINING_GUIDE.md` - Complete documentation

## What We Learned

The error wasn't MaxText or the training config. It was the tokenizer missing a required attribute. The solution was to add it programmatically before training.

This same pattern applies everywhere in ML: error messages show symptoms, not root causes. Always trace back to fundamentals.

---

**Next Steps**: Read full guide, decide when to start Stage 1 (continued pretraining)

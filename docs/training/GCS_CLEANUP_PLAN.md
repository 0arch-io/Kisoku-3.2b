# GCS Storage Cleanup Plan
## Bucket: gs://pantheon-tpu-training

### Current Storage Summary

**Total actual storage identified:**
- `kisoku-datasets/`: 515 MiB
- `datasets/`: 751 MiB
- `docs/`: 23 KiB
- **TOTAL**: ~1.27 GB

**Empty/failed experiment directories (0B each):**
All other training output directories show 0B, indicating failed runs that never wrote checkpoints.

### What to KEEP

1. **gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-GCS/**
   - Contains the base Kisoku 3.2B model checkpoint
   - Used as `load_parameters_path` in all training runs
   - **STATUS**: CRITICAL - DO NOT DELETE

2. **gs://pantheon-tpu-training/kisoku-ultrachat-sft-CHAT-TEMPLATE/**
   - Currently running successful SFT training
   - **STATUS**: IN USE - DO NOT DELETE

3. **gs://pantheon-tpu-training/docs/**
   - Contains training guides and documentation
   - Size: 23 KiB (negligible)
   - **STATUS**: KEEP

4. **gs://pantheon-tpu-training/kisoku-datasets/**
   - Pretokenized/preprocessed datasets
   - Size: 515 MiB
   - **STATUS**: KEEP (may be needed for future training)

5. **gs://pantheon-tpu-training/datasets/**
   - Dataset storage
   - Size: 751 MiB
   - **STATUS**: KEEP (may be needed for future training)

### What to DELETE (Failed Experiment Runs)

All directories showing 0B can be safely deleted:

```bash
# Failed SFT experiment runs (all 0B):
gs://pantheon-tpu-training/kisoku-fine-tuned-openhermes-FINAL/
gs://pantheon-tpu-training/kisoku-fine-tuned-openhermes-grain/
gs://pantheon-tpu-training/kisoku-fine-tuned-openhermes/
gs://pantheon-tpu-training/kisoku-fine-tuned-smoltalk/
gs://pantheon-tpu-training/kisoku-fine-tuned-synthetic/
gs://pantheon-tpu-training/kisoku-fine-tuned-tulu/
gs://pantheon-tpu-training/kisoku-openhermes-final/
gs://pantheon-tpu-training/kisoku-openhermes-hf-tokenizer/
gs://pantheon-tpu-training/kisoku-openhermes-sft/
gs://pantheon-tpu-training/kisoku-synthetic/
gs://pantheon-tpu-training/kisoku-ultrachat-nosft/
gs://pantheon-tpu-training/kisoku-ultrachat-sft-full/
gs://pantheon-tpu-training/kisoku-ultrachat-sft/
gs://pantheon-tpu-training/kisoku-ultrachat/
gs://pantheon-tpu-training/kisoku-3.2b-params/

# Old checkpoint conversion attempts (potentially larger):
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-huggingface-llama-CORRECT/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-huggingface-llama-FIXED/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-huggingface-llama-debug/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-huggingface-llama/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-huggingface/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-instruct-ALPACA-ULTIMATE/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-instruct-BEST/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-instruct-ULTIMATE/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-params-MULTIHOST/
gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-params-only/

# Test file:
gs://pantheon-tpu-training/test_write_1761581670.txt
```

### Cleanup Commands

**Option 1: Delete individually** (safer, allows verification):
```bash
# Delete failed SFT runs
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-fine-tuned-*
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-openhermes-*
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-synthetic
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-ultrachat-nosft
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-ultrachat-sft-full
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-ultrachat-sft
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-ultrachat
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-3.2b-params

# Delete old checkpoint conversion attempts
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-huggingface-*
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-instruct-*
gsutil -m rm -r gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-params-*

# Delete test file
gsutil rm gs://pantheon-tpu-training/test_write_1761581670.txt
```

**Option 2: Automated cleanup script** (faster):
```bash
#!/bin/bash
# Save to /tmp/cleanup_gcs.sh

# Delete all failed SFT runs
for dir in kisoku-fine-tuned-openhermes-FINAL kisoku-fine-tuned-openhermes-grain \
           kisoku-fine-tuned-openhermes kisoku-fine-tuned-smoltalk \
           kisoku-fine-tuned-synthetic kisoku-fine-tuned-tulu \
           kisoku-openhermes-final kisoku-openhermes-hf-tokenizer \
           kisoku-openhermes-sft kisoku-synthetic \
           kisoku-ultrachat-nosft kisoku-ultrachat-sft-full \
           kisoku-ultrachat-sft kisoku-ultrachat kisoku-3.2b-params; do
  echo "Deleting gs://pantheon-tpu-training/$dir..."
  gsutil -m rm -r gs://pantheon-tpu-training/$dir/ || echo "Failed or already deleted: $dir"
done

# Delete old checkpoint conversion attempts
for dir in kisoku-3.2b-huggingface-llama-CORRECT kisoku-3.2b-huggingface-llama-FIXED \
           kisoku-3.2b-huggingface-llama-debug kisoku-3.2b-huggingface-llama \
           kisoku-3.2b-huggingface kisoku-3.2b-instruct-ALPACA-ULTIMATE \
           kisoku-3.2b-instruct-BEST kisoku-3.2b-instruct-ULTIMATE \
           kisoku-3.2b-params-MULTIHOST kisoku-3.2b-params-only; do
  echo "Deleting gs://pantheon-tpu-training/kisoku-checkpoints/$dir..."
  gsutil -m rm -r gs://pantheon-tpu-training/kisoku-checkpoints/$dir/ || echo "Failed or already deleted: $dir"
done

# Delete test file
gsutil rm gs://pantheon-tpu-training/test_write_1761581670.txt || echo "Test file already deleted"

echo "Cleanup complete!"
```

### Expected Outcome

After cleanup, the bucket will contain ONLY:
```
gs://pantheon-tpu-training/
├── docs/                                    (23 KiB - documentation)
├── datasets/                                (751 MiB - datasets)
├── kisoku-datasets/                         (515 MiB - preprocessed data)
├── kisoku-checkpoints/
│   └── kisoku-3.2b-GCS/                    (base model checkpoint)
└── kisoku-ultrachat-sft-CHAT-TEMPLATE/     (currently running training)
```

### Cost Impact

Current actual storage: ~1.27 GB
- This is negligible cost (~$0.026/month at $0.020/GB/month)
- The 0B directories cost nothing in storage
- Most cost savings will come from deleting old checkpoint conversion attempts if they contain data

### Verification After Cleanup

Run these commands to verify:
```bash
# List remaining directories
gsutil ls gs://pantheon-tpu-training/

# Check total bucket size
gsutil du -sh gs://pantheon-tpu-training/

# Verify critical paths still exist
gsutil ls gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-GCS/checkpoints/99999/items
gsutil ls gs://pantheon-tpu-training/docs/
```

---

**IMPORTANT NOTES:**
1. Double-check that kisoku-sft TPU is still running the successful training before deleting anything
2. The base checkpoint at `kisoku-checkpoints/kisoku-3.2b-GCS/` is CRITICAL - never delete
3. Most directories showing 0B are already empty, so deletion is just cleanup
4. The real question is whether old checkpoint conversion attempts contain data worth keeping

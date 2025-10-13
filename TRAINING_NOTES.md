# Training Notes

## Current Training (Oct 12-29, 2025)

**IMPORTANT**: The actual production training is using **MaxText**, not the custom JAX scripts in this repo.

### What We're Actually Running

**Framework**: MaxText (Google's official JAX/XLA framework)
**Command**:
```bash
python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=openmind-3.2b-dclm \
  base_emb_dim=3072 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_mlp_dim=8192 \
  base_num_decoder_layers=32 \
  head_dim=96 \
  per_device_batch_size=4 \
  max_target_length=2048 \
  steps=10000000 \
  dataset_type=hf \
  hf_path=mlfoundations/dclm-baseline-1.0-parquet \
  tokenizer_path=gpt2 \
  vocab_size=50304 \
  enable_checkpointing=False
```

### Why MaxText Instead of Custom Code?

The custom JAX training scripts in `training/` were initial prototypes, but we switched to MaxText for production because:

1. **Multi-host FSDP sharding** - MaxText handles TPU v4-32 multi-worker coordination automatically
2. **Production stability** - Google's official framework, battle-tested
3. **Better performance** - Highly optimized XLA compilation
4. **Less debugging** - Works out of the box, no custom FSDP code needed

### Training Scripts in This Repo

The scripts in `training/` are:
- **train_modern.py**: 2.7B model with custom JAX FSDP (prototype)
- **train_baseline.py**: GPT-2 style baseline for comparison
- **train_3.2b.py**: 3.2B custom implementation (not used for final training)

These are kept for:
- Educational reference
- Architecture comparison
- Ablation studies
- Future fine-tuning experiments

### Actual Training Location

Training is running on TPU v4-32 (openmind-x) in us-central2-b:
- Log file: `/tmp/train_dclm_v2.log` on worker 0
- Monitored via: `gcloud compute tpus tpu-vm ssh`
- No local checkpoints (memory constraints)
- Will export final weights on Oct 29

### Dataset

**NOT using FineWeb-Edu** (as scripts suggest)
**USING: DCLM-baseline** from Apple/ML Foundations
- Source: `mlfoundations/dclm-baseline-1.0-parquet`
- Composition: ~70% web, ~20% code, ~10% math
- Streaming via HuggingFace datasets
- No local preprocessing needed

### Performance

Current metrics (live training):
```
Throughput: 80,704 tokens/sec
Per-device: 5,044 tokens/sec
MFU: 40-50%
Step time: ~1.624 seconds
Current step: 600+
Current loss: 10.5
```

### Repository Organization

```
openmind-3.2b/
├── training/              # Prototype custom JAX scripts (NOT production)
│   ├── train_modern.py    # 2.7B GQA + RoPE + SwiGLU
│   ├── train_baseline.py  # GPT-2 baseline
│   └── train_3.2b.py      # 3.2B prototype
├── README.md              # Main documentation (ACCURATE)
├── TRAINING_NOTES.md      # This file (explains actual setup)
└── [future]
    ├── checkpoints/       # Final weights (after Oct 29)
    ├── logs/             # Training logs
    └── evaluation/       # Benchmark results
```

### Next Steps

After training completes (Oct 29):
1. Export final checkpoint from MaxText
2. Convert to HuggingFace format
3. Run benchmarks (MMLU, HumanEval, GSM8K)
4. Upload weights to HuggingFace Hub
5. Update this repo with evaluation results

### Questions?

For actual training details, see:
- README.md (main documentation)
- Training command above
- Real-time logs on TPU VM

The custom scripts in `training/` are reference implementations, not production code.

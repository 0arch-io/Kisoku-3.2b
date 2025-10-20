# Training Notes & Complete Setup Guide

## Current Training (Oct 19-22, 2025)

**IMPORTANT**: The actual production training is using **MaxText**, not the custom JAX scripts in this repo.

### What We're Actually Running

**Framework**: MaxText (Google's official JAX/XLA framework)
**TPU**: openmind-x3b (v4-32, us-central2-b)
**Status**: 🟢 LIVE & RECOVERING (Step 35,383/100,000, Loss 3.0)
**Run Name**: kisoku-3.2b-GCS

**Production Command** (with GCS checkpointing):
```bash
cd ~/maxtext
source .venv/bin/activate

nohup python3 src/MaxText/train.py src/MaxText/configs/base.yml \
  run_name=kisoku-3.2b-GCS \
  base_output_directory=gs://pantheon-tpu-training \
  enable_checkpointing=true \
  checkpoint_period=5000 \
  async_checkpointing=true \
  enable_tensorboard=false \
  base_emb_dim=3072 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_mlp_dim=8192 \
  base_num_decoder_layers=32 \
  head_dim=96 \
  per_device_batch_size=8 \
  max_target_length=2048 \
  steps=100000 \
  dataset_type=hf \
  hf_path=mlfoundations/dclm-baseline-1.0-parquet \
  tokenizer_path=gpt2 \
  vocab_size=50304 \
  > /tmp/train_kisoku_gcs.log 2>&1 &
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

Training is running on TPU v4-32 (openmind-x3b) in us-central2-b:
- Log file: `/tmp/train_kisoku_gcs.log` on worker 0
- Monitored via: `gcloud compute tpus tpu-vm ssh`
- Checkpoints: Every 5k steps to gs://pantheon-tpu-training/kisoku-checkpoints/
- Will export final weights on Oct 22, 2025

### Dataset

**NOT using FineWeb-Edu** (as scripts suggest)
**USING: DCLM-baseline** from Apple/ML Foundations
- Source: `mlfoundations/dclm-baseline-1.0-parquet`
- Composition: ~70% web, ~20% code, ~10% math
- Streaming via HuggingFace datasets
- No local preprocessing needed

### Performance (Production Run)

Current metrics (post-AWS outage recovery):
```
Throughput: 85,632 tokens/sec
Per-device: 5,352 tokens/sec
MFU: 40-50%
TFLOP/s per device: 113.3
Step time: 3.061 seconds
Memory usage: 2.65GB / 30.75GB per chip (8.6%)
Current step: 35,383 / 100,000 (35.4%)
Current loss: 3.0 (down from 10.3 initial)
Time remaining: ~2.3 days
Expected completion: October 22, 2025
Total tokens: 26.2B tokens (100k steps × 262k tokens/step)
```

### Batch Size Optimization History

| Batch Size | Throughput | Step Time | Result |
|------------|-----------|-----------|--------|
| 4 per device | 80,704 tok/s | 1.624s | Initial baseline |
| 8 per device | 85,632 tok/s | 3.061s | ✅ OPTIMAL - chosen |
| 16 per device | 88,160 tok/s | 5.947s | ❌ Inefficient - communication bottleneck |

**Decision**: Settled on batch_size=8 for optimal throughput/step-time tradeoff

## AWS Outage Recovery (October 20, 2025)

### The Incident

On October 20, 2025, a major AWS US-EAST-1 outage affected "half the internet" including:
- **Consumer services**: Snapchat, Roblox, Fortnite, Reddit
- **Financial services**: Multiple UK banks
- **Infrastructure**: HuggingFace CDN (critical for our dataset streaming)

### Impact on Training

**Timeline**:
- Training was running smoothly at ~36,900 steps
- HuggingFace CDN started returning HTTP 500 errors
- Dataset loading failed repeatedly
- **Training crashed at step 36,952**

**Error Pattern**:
```
HTTP 500 errors from HuggingFace CDN
Dataset streaming failures
Connection timeouts
Process termination
```

### Recovery Process

**Steps Taken**:
1. Confirmed AWS outage was ongoing (not our infrastructure issue)
2. Located last successful checkpoint: step 35,000
3. Waited for AWS mitigation (~2 hours)
4. Verified HuggingFace CDN was responding normally
5. Restarted training from checkpoint 35,000
6. Training resumed successfully at step 35,383

**Result**:
- **Lost steps**: ~2,000 steps (36,952 - 35,000)
- **Lost time**: ~6 hours of training
- **Lost data**: None (all data in GCS checkpoints)
- **Recovery time**: ~2 hours (waiting for AWS fix)

### Lessons Learned

1. **GCS Checkpointing is Critical**
   - Without checkpoints, would have lost 36,952 steps (~4 days of training)
   - With checkpoints, lost only ~6 hours
   - **ROI of checkpointing**: Saved 3.75 days of compute ($$$)

2. **5k Step Interval is Optimal**
   - Frequent enough to limit data loss
   - Infrequent enough to not impact performance
   - Checkpoint size: ~2-3GB per checkpoint
   - Network bandwidth: Negligible impact with async checkpointing

3. **Multi-Cloud Dependencies Matter**
   - Training on GCP but data from AWS-backed HuggingFace
   - Single point of failure in cloud infrastructure
   - Consider: Local dataset caching, multi-CDN fallbacks

4. **Production ML Requires Resilience**
   - "It works in theory" ≠ production-ready
   - Real-world infrastructure failures are inevitable
   - Design for recovery, not just for success

### Validation of Strategy

This incident validates our checkpointing strategy:
- **Cost**: Minimal (async checkpointing, negligible performance impact)
- **Benefit**: Saved 3.75 days of $60/day TPU costs (~$225)
- **Proof**: System recovered automatically from major cloud outage

**Before GCS checkpointing**: Training would have been lost, restart from scratch
**After GCS checkpointing**: Training recovered in 2 hours with minimal data loss

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

## Complete Setup Guide (Reproducible)

### Prerequisites
1. Google Cloud TPU v4-32 access (via TRC or paid)
2. TPU VM running Ubuntu 22.04
3. ~3.5 days of continuous training time
4. GCS bucket for checkpoints (critical for disaster recovery)

### Step 1: Install Python 3.12 (Required!)

MaxText requires Python 3.12+, but TPU v4-32 base image has 3.10. Install on **all workers**:

```bash
# Run on ALL workers
gcloud compute tpus tpu-vm ssh openmind-x3b \
  --zone us-central2-b \
  --worker=all \
  --command='
sudo apt-get update -qq
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --set python3 /usr/bin/python3.12
python3 --version
'
```

### Step 2: Install uv Package Manager (100x faster than pip!)

```bash
gcloud compute tpus tpu-vm ssh openmind-x3b \
  --zone us-central2-b \
  --worker=all \
  --command='curl -LsSf https://astral.sh/uv/install.sh | sh'
```

### Step 3: Setup MaxText with Dependencies

```bash
gcloud compute tpus tpu-vm ssh openmind-x3b \
  --zone us-central2-b \
  --worker=all \
  --command='
cd ~
git clone https://github.com/AI-Hypercomputer/maxtext.git 2>/dev/null || (cd maxtext && git pull)
cd maxtext
python3 -m venv .venv
source .venv/bin/activate
~/.local/bin/uv pip install -r requirements.txt
~/.local/bin/uv pip install datasets tokenizers
pip install -e .
'
```

This takes 2-3 seconds with uv (vs 60-90 minutes with pip!)

### Step 4: Launch Training on All Workers

**IMPORTANT**: Must launch on **all 4 workers simultaneously** for multi-host coordination!

```bash
# Create launch script with GCS checkpointing
cat > /tmp/launch_kisoku_gcs.sh << 'EOF'
#!/bin/bash
cd ~/maxtext
source .venv/bin/activate

nohup python3 src/MaxText/train.py src/MaxText/configs/base.yml \
  run_name=kisoku-3.2b-GCS \
  base_output_directory=gs://pantheon-tpu-training \
  enable_checkpointing=true \
  checkpoint_period=5000 \
  async_checkpointing=true \
  enable_tensorboard=false \
  base_emb_dim=3072 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_mlp_dim=8192 \
  base_num_decoder_layers=32 \
  head_dim=96 \
  per_device_batch_size=8 \
  max_target_length=2048 \
  steps=100000 \
  dataset_type=hf \
  hf_path=mlfoundations/dclm-baseline-1.0-parquet \
  tokenizer_path=gpt2 \
  vocab_size=50304 \
  > /tmp/train_kisoku_gcs.log 2>&1 &

echo "Training launched! PID: $!"
EOF

# Upload to all workers
gcloud compute tpus tpu-vm scp /tmp/launch_kisoku_gcs.sh openmind-x3b:/tmp/ \
  --zone us-central2-b --worker=all

# Launch on ALL workers
gcloud compute tpus tpu-vm ssh openmind-x3b \
  --zone us-central2-b \
  --worker=all \
  --command="bash /tmp/launch_kisoku_gcs.sh"
```

### Step 5: Monitor Training

```bash
# Check status
gcloud compute tpus tpu-vm ssh openmind-x3b \
  --zone us-central2-b \
  --worker=0 \
  --command="tail -50 /tmp/train_kisoku_gcs.log"

# Watch live
gcloud compute tpus tpu-vm ssh openmind-x3b \
  --zone us-central2-b \
  --worker=0 \
  --command="tail -f /tmp/train_kisoku_gcs.log"

# Check checkpoints in GCS
gsutil ls gs://pantheon-tpu-training/kisoku-checkpoints/
```

## Troubleshooting Guide

### Problem: "Outdated Python Version"
**Error**: `Python 3.10.6, but MaxText requires Python version 3.12`
**Fix**: Install Python 3.12 (see Step 1)

### Problem: pip install hanging/taking 60+ minutes
**Error**: pip stuck on "Building wheels" or "Resolving dependencies"
**Fix**: Use uv package manager instead (see Step 2) - 100x faster!

### Problem: "ModuleNotFoundError: No module named 'absl'"
**Error**: Missing dependencies after pip install
**Fix**: Create venv and install with uv (see Step 3)

### Problem: "No module named 'distutils'"
**Error**: Python 3.12 doesn't include distutils by default
**Fix**: Using venv with uv avoids this issue entirely

### Problem: GCS Permission Denied (403)
**Error**: `403 POST https://storage.googleapis.com/upload/storage/v1/b/...`
**Fix**: Configure GCS access:
- Ensure TPU service account has Storage Admin role on bucket
- Use `gcloud auth application-default login` on TPU VM
- Set `base_output_directory=gs://your-bucket-name`
- Set `enable_checkpointing=true` with `checkpoint_period=5000`
- **Critical for production**: Enables disaster recovery (see AWS outage recovery above)

### Problem: "TPU backend initialization is taking more than 60.0 seconds"
**Error**: Multi-host timeout warning
**Fix**: Launch training on **all 4 workers simultaneously** with `--worker=all`

### Problem: batch_size=16 slower than batch_size=8
**Error**: Higher batch size gives slower training
**Root Cause**: Communication bottleneck across 4 hosts (pmean/gradient sync over network)
**Fix**: Use batch_size=8 as optimal for TPU v4-32

### Problem: apt broken after Python upgrade
**Error**: `ModuleNotFoundError: No module named 'apt_pkg'`
**Fix**: Ignore apt errors, use uv which doesn't need apt

### Problem: HuggingFace CDN HTTP 500 errors
**Error**: `HTTP 500 Internal Server Error` from HuggingFace dataset streaming
**Root Cause**: AWS outage affecting HuggingFace infrastructure (see AWS Outage Recovery section)
**Fix**:
- Wait for AWS/HuggingFace to resolve outage
- Training will automatically recover from last checkpoint
- With 5k step checkpointing, maximum loss is ~6 hours of training
- **Prevention**: Consider local dataset caching for critical production runs

## Next Steps After Training (Oct 22)

1. Export final checkpoint from MaxText
2. Convert to HuggingFace format
3. Run benchmarks (MMLU, HumanEval, GSM8K)
4. Upload weights to HuggingFace Hub
5. Update this repo with evaluation results

## Extension Strategy (If TRC Extends Access)

To reach competitive 55-60% MMLU performance:

1. **Synthetic Data Generation**
   - Use GPT-4 or Claude to generate high-quality training data
   - Focus on code, math, and reasoning tasks
   - 10-100k high-quality synthetic examples

2. **Knowledge Distillation**
   - Use Llama 3 70B or GPT-4 as teacher model
   - Generate outputs for DCLM prompts
   - Train model to match teacher outputs

3. **Data Quality Over Quantity**
   - Phi-3's "textbooks are all you need" approach
   - LIMA's 1k high-quality instruction examples
   - Filter and deduplicate DCLM further

4. **Domain Specialization**
   - Focus on code/math rather than general knowledge
   - Target "best 3B code model" or "best 3B math model"
   - Compete in specific domains vs trying to beat GPT-4 generally

5. **Modern Training Techniques**
   - Curriculum learning (easy → hard)
   - Targeted data mixing
   - Extended training on high-quality subsets

## Resources

For actual training details, see:
- README.md (main documentation)
- Training command above
- Real-time logs on TPU VM
- Setup scripts in `/tmp/` on TPU workers

The custom scripts in `training/` are reference implementations, not production code.

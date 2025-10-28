# Kisoku 3.2B (規則)

> **Kisoku** (規則): "Principles" in Japanese. Building optimal AI through rigorous methodology, systematic optimization, and complete transparency.

A 3.2 billion parameter language model trained from scratch on Google Cloud TPU v4-32, demonstrating efficient training on high-quality data (DCLM-Baseline 1.0). Open-source model, training code, and comprehensive documentation for the research community.

[![Training](https://img.shields.io/badge/status-complete-green)]() [![Architecture](https://img.shields.io/badge/architecture-GQA%20%2B%20RoPE%20%2B%20SwiGLU-blue)]() [![Dataset](https://img.shields.io/badge/dataset-DCLM--Baseline-green)]() [![Hardware](https://img.shields.io/badge/hardware-TPU%20v4--32-orange)]() [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Kisoku%203.2B-yellow)](https://huggingface.co/0arch-io/kisoku-3.2b-base)

## Overview

**Kisoku 3.2B** is a 3.2 billion parameter language model trained from scratch using Google Cloud TPU v4-32 infrastructure. The model implements Grouped-Query Attention (GQA) for efficient inference and was trained on the high-quality DCLM-Baseline 1.0 dataset.

**Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)**

## Status

- **Training**: ✅ COMPLETE (100,000 steps)
- **Final Loss**: 2.733
- **Model Release**: ✅ Published to [HuggingFace Hub](https://huggingface.co/0arch-io/kisoku-3.2b-base)
- **Hardware Used**: TPU v4-32 (32 chips, 4 hosts)
- **Total Training Time**: ~2 weeks continuous training
- **TPU-Hours**: ~850 hours on v4-32
- **Checkpoints**: gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-GCS/

## Model Architecture

```
Total Parameters: 3,244,343,296 (3.2B)
Architecture: Transformer Decoder
Layers: 32
Hidden Dimension: 3072
FFN Dimension: 8192
Attention Heads: 32 (query)
KV Heads: 8 (Grouped Query Attention, 4:1 ratio)
Head Dimension: 96
Vocabulary Size: 50,304 (GPT-2 tokenizer)
Max Sequence Length: 2048
Context Window: 2048 tokens

Key Features:
- Grouped Query Attention (GQA) - 4x reduction in KV cache memory
- RoPE (Rotary Position Embeddings)
- SwiGLU activation functions
- RMSNorm for layer normalization
```

## Training Data

**Dataset**: DCLM-Baseline 1.0 (mlfoundations/dclm-baseline-1.0-parquet)
**Source**: High-quality curated web text from DataComp-LM
**Dataset Size**: 1.6T tokens available
**Tokens Trained**: 26.2B tokens (100,000 steps × 262,144 tokens/step)
**Data Quality**: Perplexity-filtered, deduplicated, safety filtering applied

## Training Configuration

### Hardware
- **TPU**: Google Cloud TPU v4-32
- **Topology**: 2×2×4 (4 hosts, 8 chips per host)
- **Location**: us-central2-b
- **Framework**: MaxText (JAX/Flax)

### Hyperparameters

```
Framework: MaxText (Google's JAX/XLA framework)
Batch Size: 16 per device (512 global batch)
Effective Batch Size: 262,144 tokens per step
Learning Rate: Peak 3e-4, Min 3e-5
Warmup Steps: 5,000
LR Schedule: Cosine decay with warmup
Optimizer: AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)
Gradient Clipping: 1.0 (global norm)
Precision: BFloat16
Total Steps: 100,000
Checkpointing: Every 5,000 steps to GCS
```

### Performance

- **Throughput**: 85,632 tokens/sec (global)
- **Per-Device**: 5,352 tokens/sec/device
- **MFU**: 41% (Model FLOPs Utilization)
- **TFLOP/s**: 113.3 per device
- **Step Time**: 3.061 seconds
- **Training Duration**: ~2 weeks continuous

## Training Results

**Loss Convergence:**
```
Step 0:       10.523  (Initial random weights)
Step 25000:   2.967
Step 50000:   2.799
Step 75000:   2.715
Step 99999:   2.733   (Final converged state)
```

- Smooth convergence throughout training
- No loss spikes or instabilities
- 73% loss reduction from initial state
- 20 checkpoints saved successfully

## Technical Documentation

For complete technical details, see:
- **[TRC_FULL_REPORT.md](TRC_FULL_REPORT.md)** - Comprehensive 13-page report
- **[TRC_SUMMARY.md](TRC_SUMMARY.md)** - 3-page executive summary
- **[scripts/](scripts/)** - Training and setup scripts

## Usage

Model available on HuggingFace Hub: [0arch-io/kisoku-3.2b-base](https://huggingface.co/0arch-io/kisoku-3.2b-base)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("0arch-io/kisoku-3.2b-base")
tokenizer = AutoTokenizer.from_pretrained("0arch-io/kisoku-3.2b-base")
```

## Key Features

### Grouped-Query Attention (GQA)
- 4:1 query-to-KV head ratio (32 query heads, 8 KV heads)
- 4× reduction in KV cache memory during inference
- Maintains near-MHA quality with MQA efficiency
- Enables longer context windows on limited memory

### High-Quality Training Data
- DCLM-Baseline 1.0: curated web text with quality filtering
- Same dataset approach as Apple's DCLM-7B (64% MMLU)
- Demonstrates that data quality matters at smaller scales

### Efficient Training
- 41% MFU on TPU v4-32
- Robust GCS checkpointing strategy
- Fully reproducible training methodology

## Research Contributions

1. **Open Model Weights**: Public release on HuggingFace Hub
2. **Reproducible Training**: Full code and configuration available
3. **Efficient Architecture**: GQA implementation reference
4. **High-Quality Data**: Demonstrates DCLM effectiveness at 3B scale
5. **Complete Documentation**: Training methodology fully documented

## License

- **Model Weights**: Apache 2.0
- **Training Code**: MIT License
- **Dataset**: DCLM-baseline (Apple/DataComp-LM license)

## Acknowledgments

**Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)**

- **Google TRC**: For providing TPU v4-32 access
- **MaxText Team**: For excellent JAX training framework
- **ML Foundations**: For hosting DCLM dataset
- **Apple**: For DCLM-baseline dataset and demonstrating data quality importance

## Citation

```bibtex
@software{kisoku2025,
  title={Kisoku 3.2B: Principled Approach to Open Language Model Training},
  author={0ARCH},
  year={2025},
  url={https://github.com/0arch-io/Kisoku-3.2b},
  note={Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)}
}
```

## Contact

- **GitHub**: [0arch-io/Kisoku-3.2b](https://github.com/0arch-io/Kisoku-3.2b)
- **HuggingFace**: [0arch-io/kisoku-3.2b-base](https://huggingface.co/0arch-io/kisoku-3.2b-base)
- **Project**: pantheon-tpu

---

**Training Status**: ✅ COMPLETE
**Final Step**: 100,000 / 100,000 (100%)
**Final Loss**: 2.733
**Total Tokens**: 26.2B tokens
**Model Release**: [HuggingFace Hub](https://huggingface.co/0arch-io/kisoku-3.2b-base)

---

*This project demonstrates successful large-scale language model training using Google Cloud TPU Research Cloud resources, contributing Kisoku 3.2B to the open-source research community.*

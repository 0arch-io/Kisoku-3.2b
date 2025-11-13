#!/bin/bash

echo "Checking GCS bucket sizes..."
echo ""

dirs="kisoku-checkpoints
kisoku-ultrachat-sft-CHAT-TEMPLATE
kisoku-fine-tuned-openhermes-FINAL
kisoku-fine-tuned-openhermes-grain
kisoku-fine-tuned-openhermes
kisoku-fine-tuned-smoltalk
kisoku-fine-tuned-synthetic
kisoku-fine-tuned-tulu
kisoku-openhermes-final
kisoku-openhermes-hf-tokenizer
kisoku-openhermes-sft
kisoku-synthetic
kisoku-ultrachat-nosft
kisoku-ultrachat-sft-full
kisoku-ultrachat-sft
kisoku-ultrachat
kisoku-3.2b-params
kisoku-datasets
datasets
docs"

for dir in $dirs; do
  echo "=== $dir ==="
  gsutil du -sh "gs://pantheon-tpu-training/$dir/" 2>&1 | head -1
done

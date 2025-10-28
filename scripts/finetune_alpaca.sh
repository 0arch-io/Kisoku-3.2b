#!/bin/bash
# Fine-tune Kisoku 3.2B on Alpaca dataset
# TPU v4-32 (4 hosts, 32 chips total)

cd ~/maxtext
source .venv/bin/activate

# Add MaxText to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

echo "==========================================="
echo "  Kisoku 3.2B - Alpaca Fine-tuning"
echo "  TPU: kisoku3-2b-finetune (v4-32)"
echo "  Dataset: tatsu-lab/alpaca (52K samples)"
echo "  Steps: 20,000 | LR: 2e-5"
echo "==========================================="
echo ""

python3 src/MaxText/train.py \
  src/MaxText/configs/base.yml \
  run_name=kisoku-3.2b-alpaca \
  base_output_directory=gs://pantheon-tpu-training/kisoku-checkpoints \
  enable_checkpointing=true \
  checkpoint_period=2000 \
  async_checkpointing=true \
  save_checkpoint_on_completion=true \
  enable_tensorboard=false \
  \
  load_parameters_path=gs://pantheon-tpu-training/kisoku-checkpoints/kisoku-3.2b-GCS/checkpoints/99999/items \
  \
  base_emb_dim=3072 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  base_mlp_dim=8192 \
  base_num_decoder_layers=32 \
  head_dim=96 \
  vocab_size=50304 \
  \
  per_device_batch_size=12 \
  max_target_length=2048 \
  steps=20000 \
  learning_rate=2e-5 \
  \
  dataset_type=hf \
  hf_path=tatsu-lab/alpaca \
  tokenizer_path=gpt2

echo ""
echo "==========================================="
echo "  Fine-tuning completed!"
echo "==========================================="

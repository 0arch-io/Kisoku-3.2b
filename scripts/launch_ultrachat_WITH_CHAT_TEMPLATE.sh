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

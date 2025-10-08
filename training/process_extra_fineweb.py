#!/usr/bin/env python3
"""
Extra FineWeb-Edu Processing - 10M sequences to replace Wikipedia
Worker 2 is down, so this will run on Worker 0 after main task completes
"""

import os
import sys
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration
SEQ_LEN = 1024
SHARD_SIZE = 50000  # 50K sequences per shard
GCS_BUCKET = "gs://openmind-2b-training-data"

# Get HF token from environment (set with: export HF_TOKEN="your_token_here")
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN

print("=" * 60)
print("EXTRA FINEWEB-EDU PROCESSING (10M sequences)")
print("=" * 60)
print(f"Replacing Wikipedia data with higher-quality FineWeb-Edu")
print(f"Target: 10,000,000 sequences")
print("=" * 60)

# Initialize tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def create_tf_example(tokens):
    """Create a TF Example proto"""
    feature = {
        'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_shard(sequences, shard_path):
    """Write sequences to a TFRecord shard and upload to GCS"""
    local_path = f"/tmp/extra_web_shard.tfrecord"

    with tf.io.TFRecordWriter(local_path) as writer:
        for tokens in sequences:
            example = create_tf_example(tokens)
            writer.write(example.SerializeToString())

    # Upload to GCS
    os.system(f"gsutil -m cp {local_path} {shard_path}")
    os.remove(local_path)

# Load FineWeb-Edu
print("\nLoading FineWeb-Edu dataset...")
web_dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    split="train",
    streaming=True
)

print("\nProcessing 10M sequences...")
WEB_SEQS = 10_000_000
shard_idx = 0
global_shard = 200  # Start from shard 200 to avoid conflicts
sequences = []

for i, item in enumerate(tqdm(web_dataset, total=WEB_SEQS, desc="Extra FineWeb-Edu")):
    # Tokenize
    text = item["text"]
    tokens = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=SEQ_LEN,
        return_tensors='np'
    )['input_ids'][0]

    sequences.append(tokens)

    # Write shard when full
    if len(sequences) >= SHARD_SIZE:
        shard_path = f"{GCS_BUCKET}/web/shard_{global_shard:05d}.tfrecord"
        print(f"\nWriting shard {global_shard} ({len(sequences):,} seqs) -> {shard_path}")
        write_shard(sequences, shard_path)
        sequences = []
        global_shard += 1
        shard_idx += 1

    if i >= WEB_SEQS - 1:
        break

# Write remaining sequences
if sequences:
    shard_path = f"{GCS_BUCKET}/web/shard_{global_shard:05d}.tfrecord"
    print(f"\nWriting final shard {global_shard} ({len(sequences):,} seqs)")
    write_shard(sequences, shard_path)
    shard_idx += 1

print(f"\n{'='*60}")
print(f"✅ EXTRA WEB DATA COMPLETE!")
print(f"Processed {shard_idx} shards (10M sequences)")
print(f"{'='*60}")
print(f"\nFinal dataset composition:")
print(f"  Web (FineWeb-Edu): 65% (32.5M sequences)")
print(f"  Code (StarCoder): 25% (12.5M sequences)")
print(f"  Math (OpenMathInstruct): 10% (5M sequences)")
print(f"  TOTAL: 50M sequences = 51.2B tokens")
print("=" * 60)

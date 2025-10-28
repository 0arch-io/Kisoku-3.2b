#!/usr/bin/env python3
"""
TFRecord Shard Creation for TPU Training - Google Cloud Best Practice
Creates many small TFRecord files that upload to GCS and read in parallel during training

Strategy:
- Each worker handles different data portions
- Write TFRecord shards directly to GCS (no local disk space issues)
- Training can read from gs:// paths in parallel
- Fast and scalable

Dataset Mix (Llama 3 inspired):
- 45% Web: FineWeb-Edu (23B tokens)
- 25% Code: StarCoder (13B tokens) - Python, TypeScript, Rust, Go
- 20% Books: PG-19 (10B tokens)
- 10% Math: OpenMathInstruct-1 (5B tokens)
Total: 51B tokens
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
SHARD_SIZE = 50000  # 50K sequences per shard = ~200MB per file
GCS_BUCKET = "gs://openmind-2b-training-data"

# Get HF token from environment (set with: export HF_TOKEN="your_token_here")
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN

# Get worker ID from command line argument
WORKER_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NUM_WORKERS = 4

print("=" * 60)
print(f"TFRECORD SHARD CREATION - WORKER {WORKER_ID}/{NUM_WORKERS}")
print("=" * 60)
print(f"GCS Bucket: {GCS_BUCKET}")
print(f"Shard size: {SHARD_SIZE:,} sequences (~200MB per file)")
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
    local_path = f"/tmp/shard_{WORKER_ID}.tfrecord"

    with tf.io.TFRecordWriter(local_path) as writer:
        for tokens in sequences:
            example = create_tf_example(tokens)
            writer.write(example.SerializeToString())

    # Upload to GCS
    os.system(f"gsutil -m cp {local_path} {shard_path}")
    os.remove(local_path)

def process_dataset(dataset_name, dataset_stream, text_key, total_seqs, prefix):
    """Process a dataset and create TFRecord shards"""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"Total sequences: {total_seqs:,}")
    print(f"Worker {WORKER_ID} handles every {NUM_WORKERS}th shard")
    print(f"{'='*60}")

    shard_idx = 0
    global_shard = 0
    sequences = []

    for i, item in enumerate(tqdm(dataset_stream, total=total_seqs, desc=dataset_name)):
        # Tokenize
        text = item[text_key]
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
            # Only this worker writes its shards
            if global_shard % NUM_WORKERS == WORKER_ID:
                shard_path = f"{GCS_BUCKET}/{prefix}/shard_{global_shard:05d}.tfrecord"
                print(f"\nWorker {WORKER_ID} writing shard {global_shard} ({len(sequences):,} seqs) -> {shard_path}")
                write_shard(sequences, shard_path)
                shard_idx += 1

            sequences = []
            global_shard += 1

        if i >= total_seqs - 1:
            break

    # Write remaining sequences
    if sequences and global_shard % NUM_WORKERS == WORKER_ID:
        shard_path = f"{GCS_BUCKET}/{prefix}/shard_{global_shard:05d}.tfrecord"
        print(f"\nWorker {WORKER_ID} writing final shard {global_shard} ({len(sequences):,} seqs)")
        write_shard(sequences, shard_path)

    print(f"\n✅ Worker {WORKER_ID} completed {shard_idx} shards for {dataset_name}")

# Distribute work across workers
# Worker 0: Web data
# Worker 1: Code data
# Worker 2: Books data
# Worker 3: Math data

if WORKER_ID == 0:
    print("\n🌐 WORKER 0: Processing WEB DATA (45%)")
    web_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True
    )
    WEB_SEQS = 22_500_000
    process_dataset("FineWeb-Edu", web_dataset, "text", WEB_SEQS, "web")

elif WORKER_ID == 1:
    print("\n💻 WORKER 1: Processing CODE DATA (25%)")
    CODE_SEQS = 12_500_000
    seqs_per_lang = CODE_SEQS // 4

    for lang in ['python', 'typescript', 'rust', 'go']:
        print(f"\nLoading {lang}...")
        code_dataset = load_dataset(
            "bigcode/starcoderdata",
            data_dir=lang,
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        process_dataset(f"StarCoder-{lang}", code_dataset, "content", seqs_per_lang, f"code/{lang}")

elif WORKER_ID == 2:
    print("\n🌐 WORKER 2: Processing MORE WEB DATA (FineWeb-Edu - 20%)")
    print("Higher quality than Wikipedia, faster processing")
    web_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True
    )
    WEB_SEQS_ADDITIONAL = 10_000_000  # Additional 10M = total 32.5M (65%)
    process_dataset("FineWeb-Edu-Additional", web_dataset, "text", WEB_SEQS_ADDITIONAL, "web")

elif WORKER_ID == 3:
    print("\n🔢 WORKER 3: Processing MATH DATA (10%)")
    math_dataset = load_dataset(
        "nvidia/OpenMathInstruct-1",
        split="train",
        streaming=True
    )
    MATH_SEQS = 5_000_000

    # After math completes, process additional FineWeb-Edu
    print("\n🌐 WORKER 3: Also processing MORE WEB DATA (FineWeb-Edu - 20%)")
    print("Worker 2 is unresponsive, so Worker 3 will handle additional web data")
    web_dataset_extra = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True
    )
    WEB_SEQS_EXTRA = 10_000_000  # Additional 10M = total 32.5M (65%)
    # Process math first, then web

    # Combine problem + solution
    def math_stream():
        for item in math_dataset:
            # Check what fields are available
            if 'question' in item:
                text = f"Question: {item['question']}\nAnswer: {item.get('answer', item.get('solution', item.get('generated_solution', '')))}"
            elif 'problem' in item:
                text = f"Problem: {item['problem']}\nSolution: {item.get('generated_solution', item.get('solution', ''))}"
            else:
                # Just use the text field if it exists
                text = item.get('text', str(item))
            yield {"text": text}

    process_dataset("OpenMathInstruct", math_stream(), "text", MATH_SEQS, "math")

    # Now process additional web data since Worker 2 is down
    process_dataset("FineWeb-Edu-Extra", web_dataset_extra, "text", WEB_SEQS_EXTRA, "web")

print(f"\n{'='*60}")
print(f"✅ WORKER {WORKER_ID} COMPLETE!")
print(f"{'='*60}")
print("\nAll workers done = Full 50B token dataset in TFRecord shards on GCS")
print("Training can read in parallel from gs://openmind-2b-training-data/")

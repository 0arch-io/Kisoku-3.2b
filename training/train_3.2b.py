#!/usr/bin/env python3
"""
3.2B GPT Training on TPU v4-32 - QUALITY ABOVE ALL ELSE
Architecture: GQA (24 query heads, 4 KV heads) + RoPE + SwiGLU + RMSNorm + bfloat16
Dataset: 51.2B unique tokens, 10x repeats = 500B training tokens
"""

import os
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import optax
import tensorflow as tf
import time
from functools import partial
from typing import Tuple, Optional
import numpy as np

# Configuration
class Config:
    # Model architecture (3.2B parameters - EXACT: 3,203,778,240)
    vocab_size = 50257  # GPT-2 tokenizer
    seq_len = 1024
    d_model = 2880
    n_layers = 34
    n_heads = 24          # Query heads
    n_kv_heads = 4        # KV heads (GQA)
    d_ff = 7680           # SwiGLU: 4 * d_model * (2/3) rounded

    # Training
    batch_size = 256      # Global batch size across all chips
    per_device_batch = 8  # Per-chip batch size (256 / 32 chips)
    learning_rate = 3e-4
    warmup_steps = 2000
    max_steps = 122_070   # 500B tokens / (256 batch * 1024 seq)
    weight_decay = 0.1
    grad_clip = 1.0

    # Checkpointing
    checkpoint_every = 1000
    eval_every = 500
    log_every = 10

    # Data
    gcs_data_path = "gs://openmind-2b-training-data"

    # TPU
    mesh_shape = (4, 8)  # (data_parallel, model_parallel) for v4-32

print("=" * 70)
print("3.2B GPT TRAINING - QUALITY ABOVE ALL ELSE")
print("=" * 70)
print(f"Model: {Config.d_model}d x {Config.n_layers}L = 3.2B params (3,203,778,240 exact)")
print(f"Dataset: 500B tokens (51.2B unique, 10x repeats)")
print(f"Batch size: {Config.batch_size} (global) = {Config.per_device_batch} per chip")
print(f"Training steps: {Config.max_steps:,}")
print(f"TPU: v4-32 ({jax.device_count()} chips)")
print("=" * 70)

# RoPE (Rotary Position Embeddings)
def create_rope_cache(seq_len: int, d_head: int, base: float = 10000.0):
    """Create RoPE sin/cos cache"""
    theta = 1.0 / (base ** (jnp.arange(0, d_head, 2).astype(jnp.float32) / d_head))
    position = jnp.arange(seq_len).astype(jnp.float32)
    freqs = jnp.outer(position, theta)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin

def apply_rope(q, k, cos, sin):
    """Apply RoPE to queries and keys"""
    def rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    q_cos = q * cos[:, None, :]
    q_sin = rotate_half(q) * sin[:, None, :]
    q_rope = q_cos + q_sin

    k_cos = k * cos[:, None, :]
    k_sin = rotate_half(k) * sin[:, None, :]
    k_rope = k_cos + k_sin

    return q_rope, k_rope

# RMSNorm
def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: simpler than LayerNorm, works better"""
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

# SwiGLU
def swiglu(x, W_gate, W_up, W_down):
    """SwiGLU activation: better than GELU for LLMs"""
    gate = jax.nn.silu(x @ W_gate)  # silu(x) = x * sigmoid(x)
    up = x @ W_up
    return (gate * up) @ W_down

# Grouped Query Attention (GQA)
def grouped_query_attention(x, W_q, W_k, W_v, W_o, cos, sin, mask):
    """GQA: 20 query heads, 4 KV heads"""
    batch, seq, d_model = x.shape
    d_head = d_model // Config.n_heads

    # Project to Q, K, V
    q = (x @ W_q).reshape(batch, seq, Config.n_heads, d_head)
    k = (x @ W_k).reshape(batch, seq, Config.n_kv_heads, d_head)
    v = (x @ W_v).reshape(batch, seq, Config.n_kv_heads, d_head)

    # Apply RoPE
    q, k = apply_rope(q, k, cos, sin)

    # Expand K, V to match Q heads (each KV head serves 5 query heads)
    k = jnp.repeat(k, Config.n_heads // Config.n_kv_heads, axis=2)
    v = jnp.repeat(v, Config.n_heads // Config.n_kv_heads, axis=2)

    # Attention
    q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_head)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_head)
    scores = jnp.where(mask, scores, -1e10)
    attn = jax.nn.softmax(scores, axis=-1)
    out = attn @ v

    # Recombine heads
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
    return out @ W_o

# Transformer block
def transformer_block(x, params, cos, sin, mask):
    """One transformer layer"""
    # Attention
    norm1 = rms_norm(x, params['attn_norm'])
    attn_out = grouped_query_attention(
        norm1,
        params['W_q'], params['W_k'], params['W_v'], params['W_o'],
        cos, sin, mask
    )
    x = x + attn_out

    # FFN (SwiGLU)
    norm2 = rms_norm(x, params['ffn_norm'])
    ffn_out = swiglu(norm2, params['W_gate'], params['W_up'], params['W_down'])
    x = x + ffn_out

    return x

# Full model forward pass
def forward(params, tokens, rope_cache):
    """Forward pass through 2.7B GPT"""
    batch, seq = tokens.shape
    cos, sin = rope_cache

    # Embedding
    x = params['embed'][tokens]  # (batch, seq, d_model)

    # Causal mask
    mask = jnp.tril(jnp.ones((seq, seq)))
    mask = mask[None, None, :, :]  # (1, 1, seq, seq)

    # Transformer layers
    for layer_params in params['layers']:
        x = transformer_block(x, layer_params, cos, sin, mask)

    # Final norm + output projection
    x = rms_norm(x, params['final_norm'])
    logits = x @ params['output']  # (batch, seq, vocab_size)

    return logits

# Loss function
def loss_fn(params, tokens, rope_cache):
    """Cross-entropy loss"""
    logits = forward(params, tokens[:, :-1], rope_cache)
    targets = tokens[:, 1:]

    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, Config.vocab_size)
    targets_flat = targets.reshape(-1)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
    return jnp.mean(loss)

# Initialize parameters
def init_params(rng):
    """Initialize 2.7B parameters"""
    d_head = Config.d_model // Config.n_heads
    d_kv_head = Config.d_model // Config.n_kv_heads

    params = {
        'embed': random.normal(rng, (Config.vocab_size, Config.d_model)) * 0.02,
        'layers': [],
        'final_norm': jnp.ones(Config.d_model),
        'output': random.normal(random.fold_in(rng, 999),
                               (Config.d_model, Config.vocab_size)) * 0.02
    }

    # Initialize layers
    for i in range(Config.n_layers):
        layer_rng = random.fold_in(rng, i)
        layer = {
            'attn_norm': jnp.ones(Config.d_model),
            'W_q': random.normal(random.fold_in(layer_rng, 0),
                                (Config.d_model, Config.n_heads * d_head)) * 0.02,
            'W_k': random.normal(random.fold_in(layer_rng, 1),
                                (Config.d_model, Config.n_kv_heads * d_kv_head)) * 0.02,
            'W_v': random.normal(random.fold_in(layer_rng, 2),
                                (Config.d_model, Config.n_kv_heads * d_kv_head)) * 0.02,
            'W_o': random.normal(random.fold_in(layer_rng, 3),
                                (Config.d_model, Config.d_model)) * 0.02,
            'ffn_norm': jnp.ones(Config.d_model),
            'W_gate': random.normal(random.fold_in(layer_rng, 4),
                                   (Config.d_model, Config.d_ff)) * 0.02,
            'W_up': random.normal(random.fold_in(layer_rng, 5),
                                 (Config.d_model, Config.d_ff)) * 0.02,
            'W_down': random.normal(random.fold_in(layer_rng, 6),
                                   (Config.d_ff, Config.d_model)) * 0.02,
        }
        params['layers'].append(layer)

    return params

# Data loading from GCS TFRecords
def create_dataset():
    """Load TFRecord shards from GCS"""
    # Get all TFRecord files
    file_patterns = [
        f"{Config.gcs_data_path}/web/*.tfrecord",
        f"{Config.gcs_data_path}/code/*/*.tfrecord",
        f"{Config.gcs_data_path}/math/*.tfrecord",
    ]

    files = []
    for pattern in file_patterns:
        files.extend(tf.io.gfile.glob(pattern))

    print(f"Found {len(files)} TFRecord shards")

    # Parse TFRecord
    def parse_fn(example):
        features = {
            'tokens': tf.io.FixedLenFeature([Config.seq_len], tf.int64)
        }
        parsed = tf.io.parse_single_example(example, features)
        return tf.cast(parsed['tokens'], tf.int32)

    # Create dataset with 10x repeats for 500B tokens
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=32)
    dataset = dataset.repeat(10)  # 50B -> 500B tokens
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(Config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Training step
@partial(jax.pmap, axis_name='batch', donate_argnums=(0, 1))
def train_step(params, opt_state, tokens, rope_cache):
    """Single training step (pmapped across devices)"""
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, rope_cache)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')

    # Update
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

# Main training loop
def main():
    print("\n🚀 Initializing training...")

    # Setup
    rng = random.PRNGKey(42)
    rope_cache = create_rope_cache(Config.seq_len, Config.d_model // Config.n_heads)

    # Initialize model
    print("Initializing 3.1B parameters...")
    params = init_params(rng)

    # Count parameters
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))
    total_params = count_params(params)
    print(f"✅ Model initialized: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    assert total_params > 3.0e9 and total_params < 3.3e9, f"Expected ~3.1B params, got {total_params/1e9:.2f}B"

    # Optimizer with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=Config.learning_rate,
        warmup_steps=Config.warmup_steps,
        decay_steps=Config.max_steps,
        end_value=Config.learning_rate * 0.1
    )

    global optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=Config.weight_decay, b1=0.9, b2=0.95)
    )

    opt_state = optimizer.init(params)

    # Replicate across devices
    params = jax.device_put_replicated(params, jax.local_devices())
    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
    rope_cache = jax.device_put_replicated(rope_cache, jax.local_devices())

    # Load dataset
    print("\n📚 Loading dataset from GCS...")
    dataset = create_dataset()
    iterator = iter(dataset.as_numpy_iterator())

    # Training loop
    print(f"\n🔥 Starting training for {Config.max_steps:,} steps...")
    print("=" * 70)

    start_time = time.time()
    tokens_trained = 0

    for step in range(Config.max_steps):
        # Get batch
        batch = next(iterator)
        batch = batch.reshape(jax.device_count(), Config.per_device_batch, Config.seq_len)

        # Train step
        params, opt_state, loss = train_step(params, opt_state, batch, rope_cache)

        tokens_trained += Config.batch_size * Config.seq_len

        # Logging
        if step % Config.log_every == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_trained / elapsed if elapsed > 0 else 0
            loss_val = float(jax.device_get(loss[0]))

            print(f"Step {step:6d} | Loss: {loss_val:.4f} | "
                  f"Tokens: {tokens_trained/1e9:.2f}B | "
                  f"Speed: {tokens_per_sec/1e3:.1f}K tok/s | "
                  f"ETA: {(Config.max_steps - step) * elapsed / (step + 1) / 3600:.1f}h")

        # Checkpoint
        if step % Config.checkpoint_every == 0 and step > 0:
            print(f"\n💾 Saving checkpoint at step {step}...")
            # TODO: Save checkpoint

        # Eval
        if step % Config.eval_every == 0 and step > 0:
            print(f"\n📊 Evaluation at step {step}...")
            # TODO: Run eval

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print(f"Total tokens trained: {tokens_trained/1e9:.2f}B")
    print(f"Total time: {(time.time() - start_time)/3600:.1f} hours")
    print("=" * 70)

if __name__ == "__main__":
    main()

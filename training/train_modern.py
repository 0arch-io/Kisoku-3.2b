#!/usr/bin/env python3
"""2.7B Modern GPT - GQA + RoPE + SwiGLU (Llama-3 style architecture) WITH CHECKPOINTING"""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import optax
import numpy as np
import time, sys, pickle, os
from functools import partial

# Modern architecture config - matches Llama/Phi-3/Gemma style
config = {
    'vocab_size': 50257,
    'n_layers': 32,
    'd_model': 2560,
    'n_heads': 20,        # Query heads
    'n_kv_heads': 4,      # GQA: 4 KV heads (5 queries per KV head)
    'd_ff': 10240,
    'seq_len': 1024,
    'rope_theta': 10000.0,  # RoPE base frequency
}

print("="*60)
print("2.7B MODERN GPT - GQA + RoPE + SwiGLU + CHECKPOINTING")
print("="*60)
print(f"Architecture:")
print(f"  - Grouped Query Attention (GQA): {config['n_heads']} query / {config['n_kv_heads']} KV heads")
print(f"  - RoPE positional embeddings (theta={config['rope_theta']})")
print(f"  - SwiGLU activation (Swish-Gated Linear Unit)")
print(f"  - RMSNorm (Root Mean Square Normalization)")
print("="*60)

# Create checkpoint directory
CKPT_DIR = "/tmp/checkpoints_modern"
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"\nCheckpoint directory: {CKPT_DIR}")

jax.distributed.initialize()
total_devices = jax.device_count()
devices = np.array(jax.devices()).reshape(8, 2)
mesh = Mesh(devices, axis_names=('data', 'fsdp'))

BATCH = 64
print(f"\nGlobal batch: {BATCH * 8} | Devices: {total_devices}")

def param_sharding(shape):
    if len(shape) >= 2 and shape[0] % 2 == 0:
        return NamedSharding(mesh, P(None, 'fsdp'))
    elif len(shape) >= 2 and shape[1] % 2 == 0:
        return NamedSharding(mesh, P(None, 'fsdp'))
    else:
        return NamedSharding(mesh, P())

data_sharding = NamedSharding(mesh, P('data', None))
activation_sharding = NamedSharding(mesh, P('data', None, None))

# ============== RoPE Implementation ==============
def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    """Precompute RoPE frequencies"""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    freqs_cis = np.exp(1j * freqs)
    return jnp.array(freqs_cis, dtype=jnp.complex64)

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to queries and keys"""
    xq_ = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_complex = jax.lax.complex(xq_[..., 0], xq_[..., 1])
    xk_complex = jax.lax.complex(xk_[..., 0], xk_[..., 1])

    freqs_cis = freqs_cis[:xq.shape[1], :]
    freqs_cis = freqs_cis[None, :, None, :]

    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis

    xq_out = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(*xq.shape)
    xk_out = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(*xk.shape)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

head_dim = config['d_model'] // config['n_heads']
freqs_cis = precompute_freqs_cis(head_dim, config['seq_len'], config['rope_theta'])

def init_params(key):
    params = {}
    k1, key = jax.random.split(key)

    params['embed'] = jax.device_put(
        (jax.random.normal(k1, (config['vocab_size'], config['d_model'])) * 0.02).astype(jnp.bfloat16),
        param_sharding((config['vocab_size'], config['d_model']))
    )

    for i in range(config['n_layers']):
        k1, k2, k3, k4, k5, key = jax.random.split(key, 6)

        params[f'layer_{i}'] = {
            'attn_q': jax.device_put(
                (jax.random.normal(k1, (config['d_model'], config['n_heads'] * head_dim)) * 0.02).astype(jnp.bfloat16),
                param_sharding((config['d_model'], config['n_heads'] * head_dim))
            ),
            'attn_k': jax.device_put(
                (jax.random.normal(k2, (config['d_model'], config['n_kv_heads'] * head_dim)) * 0.02).astype(jnp.bfloat16),
                param_sharding((config['d_model'], config['n_kv_heads'] * head_dim))
            ),
            'attn_v': jax.device_put(
                (jax.random.normal(k3, (config['d_model'], config['n_kv_heads'] * head_dim)) * 0.02).astype(jnp.bfloat16),
                param_sharding((config['d_model'], config['n_kv_heads'] * head_dim))
            ),
            'attn_out': jax.device_put(
                (jax.random.normal(k4, (config['n_heads'] * head_dim, config['d_model'])) * 0.02).astype(jnp.bfloat16),
                param_sharding((config['n_heads'] * head_dim, config['d_model']))
            ),
            'ffn_gate': jax.device_put(
                (jax.random.normal(k5, (config['d_model'], config['d_ff'])) * 0.02).astype(jnp.bfloat16),
                param_sharding((config['d_model'], config['d_ff']))
            ),
            'ffn_up': jax.device_put(
                (jax.random.normal(key, (config['d_model'], config['d_ff'])) * 0.02).astype(jnp.bfloat16),
                param_sharding((config['d_model'], config['d_ff']))
            ),
            'ffn_down': jax.device_put(
                (jax.random.normal(key, (config['d_ff'], config['d_model'])) * 0.02).astype(jnp.bfloat16),
                param_sharding((config['d_ff'], config['d_model']))
            ),
            'ln1_g': jax.device_put(jnp.ones(config['d_model'], dtype=jnp.bfloat16), param_sharding((config['d_model'],))),
            'ln2_g': jax.device_put(jnp.ones(config['d_model'], dtype=jnp.bfloat16), param_sharding((config['d_model'],))),
        }

    params['ln_f_g'] = jax.device_put(jnp.ones(config['d_model'], dtype=jnp.bfloat16), param_sharding((config['d_model'],)))
    return params

def rmsnorm(x, g):
    rms = jnp.sqrt(jnp.mean(x**2, -1, keepdims=True) + 1e-5)
    return g * x / rms

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, T, n_kv_heads, head_dim = x.shape
    return jnp.repeat(x, n_rep, axis=2)

def attention_gqa(x, q_w, k_w, v_w, out_w, freqs_cis):
    B, T, C = x.shape
    q = jnp.dot(x, q_w).reshape(B, T, config['n_heads'], head_dim)
    k = jnp.dot(x, k_w).reshape(B, T, config['n_kv_heads'], head_dim)
    v = jnp.dot(x, v_w).reshape(B, T, config['n_kv_heads'], head_dim)

    q, k = apply_rotary_emb(q, k, freqs_cis)

    n_rep = config['n_heads'] // config['n_kv_heads']
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

    return jnp.dot(out, out_w)

def swiglu_ffn(x, gate_w, up_w, down_w):
    gate = jnp.dot(x, gate_w)
    up = jnp.dot(x, up_w)
    swish_gate = gate * jax.nn.sigmoid(gate)
    hidden = swish_gate * up
    return jnp.dot(hidden, down_w)

def transformer_layer(x, layer, freqs_cis):
    x = jax.lax.with_sharding_constraint(x, activation_sharding)
    x_norm = rmsnorm(x, layer['ln1_g'])
    x = x + attention_gqa(x_norm, layer['attn_q'], layer['attn_k'], layer['attn_v'],
                          layer['attn_out'], freqs_cis)
    x_norm = rmsnorm(x, layer['ln2_g'])
    x = x + swiglu_ffn(x_norm, layer['ffn_gate'], layer['ffn_up'], layer['ffn_down'])
    return x

transformer_layer_remat = jax.checkpoint(transformer_layer)

def gpt_forward(tokens, params):
    x = params['embed'][tokens]
    x = jax.lax.with_sharding_constraint(x, activation_sharding)

    for i in range(config['n_layers']):
        if i % 8 == 0:
            x = transformer_layer_remat(x, params[f'layer_{i}'], freqs_cis)
        else:
            x = transformer_layer(x, params[f'layer_{i}'], freqs_cis)

    x = rmsnorm(x, params['ln_f_g'])
    logits = jnp.dot(x, params['embed'].T).astype(jnp.float32)
    return logits

def loss_fn(params, tokens):
    logits = gpt_forward(tokens, params)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)

def save_checkpoint(step, params, opt_state, loss, total_tokens):
    """Save checkpoint - only on worker 0 to avoid duplicates"""
    if jax.process_index() == 0:
        ckpt_path = f"{CKPT_DIR}/checkpoint_step_{step}.pkl"
        print(f"\n💾 SAVING CHECKPOINT: {ckpt_path}")

        # Convert to CPU for saving
        params_cpu = jax.tree_map(lambda x: np.array(x), params)
        opt_state_cpu = jax.tree_map(lambda x: np.array(x) if hasattr(x, 'shape') else x, opt_state)

        checkpoint = {
            'step': step,
            'params': params_cpu,
            'opt_state': opt_state_cpu,
            'config': config,
            'loss': float(loss),
            'total_tokens': total_tokens,
            'timestamp': time.time(),
        }

        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"✅ Checkpoint saved! Step {step}, Loss {float(loss):.4f}, {total_tokens/1e9:.1f}B tokens\n")
        sys.stdout.flush()

print("\nInitializing modern architecture...")
params = init_params(jax.random.PRNGKey(42))
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Model: {total_params/1e9:.2f}B params")
print(f"  Q proj: {config['n_heads']} heads × {head_dim} dim")
print(f"  K,V proj: {config['n_kv_heads']} heads × {head_dim} dim (GQA saves {((config['n_heads'] - config['n_kv_heads']) * head_dim * config['d_model'] * 2 * config['n_layers'] / 1e6):.0f}M params!)")

print("\nLoading FineWeb-Edu...")
data_mmap = np.load('/tmp/fineweb_tokenized/fineweb_shared.npy', mmap_mode='r')
data_buffer = np.array(data_mmap[:75000], dtype=np.int32)
print(f"Buffered 75000 sequences")

lr_schedule = optax.warmup_cosine_decay_schedule(0.0, 6e-4, 2000, 500_000, 6e-5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule, b1=0.9, b2=0.95, weight_decay=0.1))
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

print("\nCompiling modern architecture...")
dummy = jnp.zeros((BATCH, config['seq_len']), dtype=jnp.int32)
dummy = jax.device_put(dummy, data_sharding)
t0 = time.time()
params, opt_state, _ = train_step(params, opt_state, dummy)
print(f"Compiled in {time.time()-t0:.1f}s!\n")

# Pre-generate indices
rng = np.random.RandomState(42)
all_indices = rng.randint(0, len(data_buffer), (10000, BATCH), dtype=np.int32)

# Checkpoint schedule
CHECKPOINT_STEPS = [10000, 20000, 50000, 100000, 250000, 500000]
print(f"TRAINING MODERN ARCHITECTURE! (GQA + RoPE + SwiGLU)")
print(f"Checkpoints at steps: {CHECKPOINT_STEPS}\n")

total_tokens = 0
start = time.time()

for step in range(500_000):
    idx_offset = step % 10000
    idx = all_indices[idx_offset]
    batch_np = data_buffer[idx]
    batch = jax.device_put(jnp.asarray(batch_np), data_sharding)

    params, opt_state, loss = train_step(params, opt_state, batch)

    if idx_offset == 0 and step > 0:
        rng_reseed = np.random.RandomState(42 + step // 10000)
        all_indices = rng_reseed.randint(0, len(data_buffer), (10000, BATCH), dtype=np.int32)

    if step % 10 == 0:
        loss.block_until_ready()
        total_tokens += BATCH * 8 * config['seq_len'] * 10

        elapsed = time.time() - start
        tok_s = total_tokens / elapsed
        mfu = (tok_s / 1e6) / (275.0 * 16 / 1000) * 100
        print(f"[STEP {step:5d}] Loss: {float(loss):.4f} | {tok_s/1e6:.2f}M tok/s | MFU: {mfu:.1f}%")
        sys.stdout.flush()

        if step % 50 == 0 and step > 0:
            tok_day = tok_s * 86400
            days_500b = 500e9 / tok_day
            status = "✓ MODERN ARCH!" if days_500b <= 7 else f"{days_500b:.1f} days"
            print(f"  {tok_day/1e9:.1f}B tok/day | 500B in {status}")
            sys.stdout.flush()
    else:
        total_tokens += BATCH * 8 * config['seq_len']

    # Save checkpoints at key milestones
    if (step + 1) in CHECKPOINT_STEPS:
        loss.block_until_ready()
        save_checkpoint(step + 1, params, opt_state, loss, total_tokens)

print("\n🎉 TRAINING COMPLETE!")
print(f"Final checkpoint saved at step 500000")
print("Done!")

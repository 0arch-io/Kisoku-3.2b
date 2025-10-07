#!/usr/bin/env python3
"""2.7B GPT - FINAL OPTIMIZED: Best we can do on v4-32 without MaxText"""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import optax
import numpy as np
import time, sys

config = {'vocab_size': 50257, 'n_layers': 32, 'd_model': 2560, 'n_heads': 20, 'd_ff': 10240, 'seq_len': 1024}

print("="*60); print("2.7B GPT - FINAL OPTIMIZED"); print("="*60)

jax.distributed.initialize()
total_devices = jax.device_count()
devices = np.array(jax.devices()).reshape(8, 2)
mesh = Mesh(devices, axis_names=('data', 'fsdp'))

# Optimized batch: 512 global (64 per device, 32 per TensorCore)
# This is the max that fits with seq=1024
BATCH = 64  # Per device
print(f"Global batch: {BATCH * 8} | Devices: {total_devices}")

def param_sharding(shape):
    if len(shape) >= 2 and shape[0] % 2 == 0:
        return NamedSharding(mesh, P(None, 'fsdp'))
    elif len(shape) >= 2 and shape[1] % 2 == 0:
        return NamedSharding(mesh, P(None, 'fsdp'))
    else:
        return NamedSharding(mesh, P())

data_sharding = NamedSharding(mesh, P('data', None))
activation_sharding = NamedSharding(mesh, P('data', None, None))

def init_params(key):
    params = {}
    k1, k2, key = jax.random.split(key, 3)
    params['embed'] = jax.device_put((jax.random.normal(k1, (config['vocab_size'], config['d_model'])) * 0.02).astype(jnp.bfloat16), param_sharding((config['vocab_size'], config['d_model'])))
    params['pos_embed'] = jax.device_put((jax.random.normal(k2, (config['seq_len'], config['d_model'])) * 0.02).astype(jnp.bfloat16), param_sharding((config['seq_len'], config['d_model'])))

    for i in range(config['n_layers']):
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        params[f'layer_{i}'] = {
            'attn_qkv': jax.device_put((jax.random.normal(k1, (config['d_model'], 3*config['d_model'])) * 0.02).astype(jnp.bfloat16), param_sharding((config['d_model'], 3*config['d_model']))),
            'attn_out': jax.device_put((jax.random.normal(k2, (config['d_model'], config['d_model'])) * 0.02).astype(jnp.bfloat16), param_sharding((config['d_model'], config['d_model']))),
            'ffn_w1': jax.device_put((jax.random.normal(k3, (config['d_model'], config['d_ff'])) * 0.02).astype(jnp.bfloat16), param_sharding((config['d_model'], config['d_ff']))),
            'ffn_w2': jax.device_put((jax.random.normal(k4, (config['d_ff'], config['d_model'])) * 0.02).astype(jnp.bfloat16), param_sharding((config['d_ff'], config['d_model']))),
            'ln1_g': jax.device_put(jnp.ones(config['d_model'], dtype=jnp.bfloat16), param_sharding((config['d_model'],))),
            'ln2_g': jax.device_put(jnp.ones(config['d_model'], dtype=jnp.bfloat16), param_sharding((config['d_model'],))),
        }

    params['ln_f_g'] = jax.device_put(jnp.ones(config['d_model'], dtype=jnp.bfloat16), param_sharding((config['d_model'],)))
    return params

def rmsnorm(x, g):
    rms = jnp.sqrt(jnp.mean(x**2, -1, keepdims=True) + 1e-5)
    return g * x / rms

def attn_fused(x, qkv_w, out_w):
    B, T, C = x.shape
    head_dim = C // config['n_heads']
    qkv = jnp.dot(x, qkv_w).reshape(B, T, 3, config['n_heads'], head_dim)
    q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    return jnp.dot(out, out_w)

# Selective remat: checkpoint every 8th layer (4 total checkpoints)
def transformer_layer(x, layer):
    x = jax.lax.with_sharding_constraint(x, activation_sharding)
    x_norm = rmsnorm(x, layer['ln1_g'])
    x = x + attn_fused(x_norm, layer['attn_qkv'], layer['attn_out'])
    x_norm = rmsnorm(x, layer['ln2_g'])
    h = jax.nn.gelu(jnp.dot(x_norm, layer['ffn_w1']))
    x = x + jnp.dot(h, layer['ffn_w2'])
    return x

transformer_layer_remat = jax.checkpoint(transformer_layer)

def gpt_forward(tokens, params):
    x = params['embed'][tokens]
    pos = params['pos_embed'][:tokens.shape[1]]
    x = x + pos
    x = jax.lax.with_sharding_constraint(x, activation_sharding)

    # Selective remat: checkpoint every 8th layer
    for i in range(config['n_layers']):
        if i % 8 == 0:
            x = transformer_layer_remat(x, params[f'layer_{i}'])
        else:
            x = transformer_layer(x, params[f'layer_{i}'])

    x = rmsnorm(x, params['ln_f_g'])
    logits = jnp.dot(x, params['embed'].T).astype(jnp.float32)
    return logits

def loss_fn(params, tokens):
    logits = gpt_forward(tokens, params)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)

print("\nInitializing...")
params = init_params(jax.random.PRNGKey(42))
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Model: {total_params/1e9:.2f}B params")

print("Loading FineWeb...")
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

print("\nCompiling...")
dummy = jnp.zeros((BATCH, config['seq_len']), dtype=jnp.int32)
dummy = jax.device_put(dummy, data_sharding)
t0 = time.time()
params, opt_state, _ = train_step(params, opt_state, dummy)
print(f"Compiled in {time.time()-t0:.1f}s!\n")

# Pre-generate indices
rng = np.random.RandomState(42)
all_indices = rng.randint(0, len(data_buffer), (10000, BATCH), dtype=np.int32)

print(f"TRAINING! Global batch={BATCH*8}, seq=1024, selective remat\n")
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
            status = "✓ BEATING APPLE!" if days_500b <= 20 else f"Need {0.30/tok_s:.2f}M tok/s for 20 days"
            print(f"  {tok_day/1e9:.1f}B tok/day | 500B in {days_500b:.1f} days | {status}")
            sys.stdout.flush()
    else:
        total_tokens += BATCH * 8 * config['seq_len']

print("Done!")

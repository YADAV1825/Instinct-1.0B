---
license: apache-2.0
---

# Instinct-1-1B

*Instinct-1-1B is a fully reproducible, from-scratch trained 1B parameter language model trained on 20B tokens of PILE using TPU v4 infrastructure.*

**Instinct-1-1B** is a 1 Billion parameter Large Language Model built from scratch under the **AutonomousX** organization.

Compute for this project was supported by **[Google's TRC Program (TPU Research Cloud)](https://sites.research.google/trc/about/)**.

This model was developed by **Rohit Yadav**, a **B.Tech 3rd year student from NIT Jalandhar, India** E-mail: yrohit1825@gmail.com.

---

## Model Overview

| Attribute | Value |
|-----------|------|
| Model Name | Instinct-1-1B |
| Organization | AutonomousX |
| Parameters | ~1 Billion |
| Vocabulary Size | 50,304 |
| Training Dataset | Pythia / The PILE |
| Tokens Seen | 85 Billion |
| Training Hardware | TPU v4-8 |

Validation was performed using **rolling validation shards of the dataset**.

![image](https://cdn-uploads.huggingface.co/production/uploads/68bf07a31d80a360f1405b72/ICT8s2ycXLVz9MLgc9iBD.png)

---

## Architecture Details

Unlike previous versions, this model utilizes standard Self-Attention without Rotary Position Embeddings (RoPE). 

| Hyperparameter | Value |
|-----------|------|
| Layers | 24 |
| Model Dimension | 1840 |
| Attention Heads | 16 |
| Feed Forward Dimension | 4968 |
| Sequence Length | 1024 |

---

## Training Details

Instinct-1-1B was trained completely **from scratch** using **JAX/Flax on TPU v4-8 hardware**.

Training pipeline includes:

* Dataset streaming from **The PILE / Pythia Data**
* Custom tokenizer with **50,304 vocabulary size**
* TPU optimized **JAX / Flax training loop with pmap**
* Checkpointing and validation during training
* Rolling validation shard evaluation

The model was trained on **20B tokens** and it's is a checkpoint of final version trained on **85B tokens** in total

---

## Reproducibility

The entire pipeline used to train the model is fully reproducible.

This includes:

* Dataset pipeline
* Tokenizer creation
* Model architecture
* TPU training loop
* Checkpointing system

You can reproduce the complete training pipeline from scratch.

---

## Run Inference (Model is available for Inference on Both GPUs and TPUs)

A ready-to-run **Google Colab TPU/GPU inference script** is provided below. 

Simply open a notebook and run it with a TPU or GPU runtime. Please be patient, it may take some time to download and compile.

---
<div style="max-height:450px; overflow:auto;">

```python
# Install huggingface_hub if not installed
!pip install -q huggingface_hub

from huggingface_hub import snapshot_download

repo_id = "autonomousX/Instinct-1-1B"

# Download entire repository
local_path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir="TPU_1b",
    local_dir_use_symlinks=False
)

print("Download complete!")
print("Saved to:", local_path)

# =========================
# FAST 1B INFERENCE CELL
# =========================

import os
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from transformers import AutoTokenizer
import jax.random as random

# ---------------- CONFIG ----------------
SEQ_LEN = 1024
VOCAB_SIZE = 50304

N_LAYERS = 24
D_MODEL  = 1840   
N_HEADS  = 16
D_FF     = 4968

CKPT_PATH = os.path.abspath("TPU_1b/checkpoint_0") # Ensure your checkpoint directory matches

# ---------------- MODEL ----------------
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x * (scale / norm)

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask):

        # ---- Attention ----
        h = RMSNorm(D_MODEL)(x)
        h = nn.SelfAttention(
            num_heads=N_HEADS,
            dtype=jnp.bfloat16,
            use_bias=False,
            deterministic=True,
        )(h, mask=mask)
        x = x + h

        # ---- MLP ----
        h = RMSNorm(D_MODEL)(x)
        h = nn.Dense(D_FF, dtype=jnp.bfloat16)(h)
        h = nn.gelu(h)
        h = nn.Dense(D_MODEL, dtype=jnp.bfloat16)(h)

        return x + h

class GPT(nn.Module):
    @nn.compact
    def __call__(self, input_ids):

        batch, seq_len = input_ids.shape
        mask = nn.attention.make_causal_mask(
            jnp.ones((batch, seq_len), dtype=jnp.bool_)
        )

        x = nn.Embed(
            VOCAB_SIZE,
            D_MODEL,
            embedding_init=nn.initializers.normal(0.02),
            dtype=jnp.bfloat16,
        )(input_ids)

        RematBlock = nn.remat(Block)

        for _ in range(N_LAYERS):
            x = RematBlock()(x, mask)

        x = RMSNorm(D_MODEL)(x)

        logits = nn.Dense(
            VOCAB_SIZE,
            use_bias=False,
            dtype=jnp.bfloat16
        )(x)

        return logits

# ---------------- LOAD CHECKPOINT ----------------
def create_state():
    model = GPT()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, SEQ_LEN), dtype=jnp.int32))
    tx = optax.adamw(1e-4) # Placeholder optimizer for loading
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

state = create_state()
state = checkpoints.restore_checkpoint(CKPT_PATH, state)
params = state.params
model = GPT()

print("Checkpoint loaded.")

# ---------------- GENERATION ----------------
def generate(params, input_ids, max_new_tokens=30, temperature=0.9, top_k=40):
    rng = random.PRNGKey(0)

    for _ in range(max_new_tokens):
        logits = model.apply(params, input_ids)
        logits = logits[:, -1, :]
        logits = logits.astype(jnp.float32)

        logits = logits / temperature

        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        probs = jax.nn.softmax(top_k_logits, axis=-1)

        rng, subkey = random.split(rng)
        next_token_idx = random.categorical(subkey, jnp.log(probs))

        next_token = jnp.take_along_axis(
            top_k_indices,
            next_token_idx[:, None],
            axis=-1
        )

        input_ids = jnp.concatenate([input_ids, next_token], axis=1)

    return input_ids

# ---------------- RUN ----------------
tokenizer = AutoTokenizer.from_pretrained("autonomousX/Instinct-1-1B")

prompt = "I am John,"
tokens = tokenizer(prompt, return_tensors="np")
input_ids = jnp.array(tokens["input_ids"], dtype=jnp.int32)

output_ids = generate(params, input_ids, 200)

print("\n=== GENERATED TEXT ===\n")
print(tokenizer.decode(output_ids[0].tolist()))
```
</div>


<div id="autonomousx-profile-sections" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; width: 100%; display: flex; flex-direction: column; gap: 24px; margin: 30px 0;">

<style>
#autonomousx-profile-sections * { box-sizing: border-box; }

.ax-card {
position: relative;
overflow: hidden;
width: 100%;
padding: 25px;
border-radius: 12px;
border: 1px solid #e5e7eb;
background: #ffffff;
box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
text-align: left;
}

@keyframes diagonalShimmer {
0% { transform: translateX(-150%) skewX(-15deg); }
50% { transform: translateX(150%) skewX(-15deg); }
100% { transform: translateX(150%) skewX(-15deg); }
}

.ax-card::before {
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background: linear-gradient(90deg, rgba(255, 255, 255, 0) 0%, rgba(139, 92, 246, 0.05) 40%, rgba(236, 72, 153, 0.1) 50%, rgba(139, 92, 246, 0.05) 60%, rgba(255, 255, 255, 0) 100%);
animation: diagonalShimmer 5s infinite ease-in-out;
pointer-events: none;
z-index: 1;
}

.ax-card-content { position: relative; z-index: 2; }

.ax-card h1 { margin: 0 0 4px 0; font-size: 14px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; }
.ax-card h2 { margin: 0 0 16px 0; font-size: 32px; color: #8b5cf6; font-weight: 800; letter-spacing: -0.5px; }

/* The new dark interior box */
.ax-dark-box {
background: #0f172a;
color: #e2e8f0;
padding: 20px;
border-radius: 8px;
margin-top: 15px;
border: 1px solid #1e293b;
}

.ax-dark-box p { margin: 0 0 12px 0; font-size: 15px; line-height: 1.6; }
.ax-dark-box a { color: #a78bfa; text-decoration: none; font-weight: 600; }
.ax-dark-box a:hover { color: #d946ef; text-decoration: underline; }

.ax-icon { margin-right: 8px; font-style: normal; }
.ax-highlight-text { color: #f472b6; font-weight: 700; }

.ax-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.ax-badge { background: #1e293b; color: #cbd5e1; padding: 6px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; border: 1px solid #334155; }
</style>

<div class="ax-card">
<div class="ax-card-content">
<h1>Author</h1>
<h2>Rohit Yadav</h2>

<div class="ax-dark-box">
<p>
<strong>B.Tech 3rd Year</strong><br>
Dr. B.R. Ambedkar National Institute of Technology (NIT) Jalandhar, India
</p>

<p>
<span class="ax-icon">📧</span> E-mail: <a href="mailto:yrohit1825@gmail.com">yrohit1825@gmail.com</a><br>
<span class="ax-icon">🔗</span> LinkedIn: <a href="https://www.linkedin.com/in/rohit-yadav-25535b256/" target="_blank">Rohit Yadav</a><br>
<span class="ax-icon">💻</span> Github: <a href="https://github.com/YADAV1825" target="_blank">YADAV1825</a>
</p>

<p class="ax-highlight-text" style="margin-top: 16px;">
🚀 I am actively seeking Internships and Collaborations!
</p>

<div style="margin-top: 20px;">
<h3 style="font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px 0;">Research Interests</h3>
<div class="ax-badges">

<span class="ax-badge">Bio_Informatics</span>
<span class="ax-badge">Large Language Models</span>
<span class="ax-badge">MultiModal Pipelines</span>
<span class="ax-badge">Systems Programming</span>
<span class="ax-badge">AI Infrastructure</span>
<span class="ax-badge">Distributed Training</span>
</div>
</div>
</div>
</div>
</div>

<div class="ax-card">
<div class="ax-card-content">
<h1>Organization</h1>
<h2 style="color: #3b82f6;">AutonomousX</h2>

<div class="ax-dark-box">
<p>
<strong>AutonomousX</strong> focuses on open-source contributions aimed at building Large Language Models from scratch using custom training pipelines.
</p>

<p>
Our work explores different training configurations including optimizers, datasets, and scalable TPU training using <strong>JAX and pmap</strong>. The goal is to provide transparent and reproducible implementations so that researchers, students, and developers can understand how modern LLMs are trained end-to-end.
</p>

<p style="margin-bottom: 0;">
Due to the current scarcity of complete beginner-friendly guides for training LLMs on TPUs, especially using JAX, AutonomousX aims to bridge this gap by publishing full training pipelines, scripts, and documentation for the open-source community.
</p>
</div>
</div>
</div>

</div>

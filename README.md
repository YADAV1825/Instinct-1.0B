---
license: apache-2.0
---
# Instinct-1-1B

*Instinct-1-1B is a fully reproducible, from-scratch trained 1B parameter language model trained on 85B tokens using TPU v4 infrastructure.*

**Instinct-1-1B** is a 1 Billion parameter Large Language Model built entirely from scratch under the **AutonomousX** organization. 

Compute for this project was supported by **[Google's TRC Program (TPU Research Cloud)](https://sites.research.google/trc/about/)**.

---


### 👤 Author

**Rohit Yadav**
> **B.Tech 3rd Year**<br>
> Dr. B.R. Ambedkar National Institute of Technology (NIT) Jalandhar, India
>
> 📧 **E-mail:** [yrohit1825@gmail.com](mailto:yrohit1825@gmail.com)<br>
> 🔗 **LinkedIn:** [Rohit Yadav](https://www.linkedin.com/in/rohit-yadav-25535b256/)<br>
> 💻 **Github:** [YADAV1825](https://github.com/YADAV1825)
>
> 🚀 **I am actively seeking Internships and Collaborations!**

**Research Interests:** `Large Language Models` • `MultiModal Pipelines` • `Systems Programming` • `AI Infrastructure` • `Distributed Training`

---

### 🏢 Organization

<h4 style="color: #3b82f6; margin-bottom: 0;">AutonomousX</h4>

> **AutonomousX** focuses on open-source contributions aimed at building Large Language Models from scratch using custom training pipelines.
>
> Our work explores different training configurations including optimizers, datasets, and scalable TPU training using **JAX and pmap**. The goal is to provide

---

### ⚠️ Disclaimer
**This is a base model, not an SFT (Supervised Fine-Tuned) or RLHF (Reinforcement Learning from Human Feedback) model.** As a raw completion model, it may output undesired, biased, or nonsensical text. It is intended primarily for research and educational purposes.

---

### 📊 Model Overview & Architecture

Unlike previous versions, this model utilizes standard Self-Attention **without** Rotary Position Embeddings (RoPE).

| Attribute | Value |
| :--- | :--- |
| **Model Name** | Instinct-1-1B |
| **Organization** | AutonomousX |
| **Parameters** | ~1 Billion |
| **Vocabulary Size** | 50,304 |
| **Dataset** | Pythia / [The Pile by EleutherAI](https://pile.eleuther.ai/) |
| **Tokens Seen** | 85 Billion |
| **Training Hardware** | TPU v4-8 |
| **Layers** | 24 |
| **Model Dimension (D_MODEL)** | 1840 |
| **Attention Heads** | 16 |
| **Feed Forward Dimension**| 4968 |
| **Sequence Length** | 1024 |

*Validation was performed using rolling validation shards of the dataset.*

![image](https://cdn-uploads.huggingface.co/production/uploads/68bf07a31d80a360f1405b72/ICT8s2ycXLVz9MLgc9iBD.png)

---

### 🧠 Training Details

Instinct-1-1B was trained completely **from scratch** using **JAX/Flax on TPU v4-8 hardware**. 

The training pipeline includes:
* Dataset streaming from **The PILE / Pythia Data**.
* Custom tokenizer with a **50,304 vocabulary size**.
* TPU optimized **JAX / Flax** training loop using **pmap**.
* Checkpointing and validation during training.
* Rolling validation shard evaluation.

The model was successfully trained on a total of **85B tokens**.

---

### 🔄 Reproducibility

The entire pipeline used to train the model is fully reproducible. This includes the dataset pipeline, tokenizer creation, model architecture, TPU training loop, and checkpointing system.

You can reproduce the complete training pipeline from scratch using the AutonomousX training scripts.

---

### 🚀 Run Inference (Colab TPU/GPU)

> The trained LLM inference script and model weights are available at: [autonomousX/Instinct-1-1B on Hugging Face](https://huggingface.co/autonomousX/Instinct-1-1B).
> The training script can be found here [train.py](https://github.com/YADAV1825/Instinct-1.0B/blob/main/train.py)

A ready-to-run Google Colab TPU/GPU inference script is provided below. Simply open a notebook, set your runtime to TPU or GPU, and run it. *(Please be patient, it may take some time to download and compile).*

<details>
<summary>Click here to view the full 1B Inference Code</summary>

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
---
transparent and reproducible implementations so that researchers, students, and developers can understand how modern LLMs are trained end-to-end.
>
> Due to the current scarcity of complete beginner-friendly guides for training LLMs on TPUs, especially using JAX, AutonomousX aims to bridge this gap by publishing full training pipelines, scripts, and documentation for the open-source community.

---



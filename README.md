---
license: apache-2.0
---

# Instinct-1-1B

*Instinct-1-1B is a fully reproducible, from-scratch trained 1B parameter language model trained on 85B tokens of PILE using TPU v4 infrastructure.*

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

The model was trained on **85B tokens**
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
> Our work explores different training configurations including optimizers, datasets, and scalable TPU training using **JAX and pmap**. The goal is to provide transparent and reproducible implementations so that researchers, students, and developers can understand how modern LLMs are trained end-to-end.
>
> Due to the current scarcity of complete beginner-friendly guides for training LLMs on TPUs, especially using JAX, AutonomousX aims to bridge this gap by publishing full training pipelines, scripts, and documentation for the open-source community.

---





/div>

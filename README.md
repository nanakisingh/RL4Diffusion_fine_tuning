## RL4Diffusion: Enhancing Reasoning in Diffusion-Based Language

# Abstract
We study reinforcement learning (RL) post-training for LLaDA, a Large Language Diffusion Model, to improve alignment with human intent. Unlike traditional autoregressive models, LLaDA leverages bidirectional context, conditioning token unmasking on the entire generated sequence. While LLaDA already outperforms Gemma2-9B and LLaMA-8B following supervised fine-tuning (SFT), it has not previously undergone RL fine-tuning. We introduce two RL fine-tuning pipelines: one based on Group Relative Policy Optimization (GRPO) and another on Score Entropy Policy Optimization (SEPO), designed for diffusion models with non-differentiable reward functions. After fine-tuning LLaDA for one epoch on 363 examples, we observe a marked 24.67% improvement.


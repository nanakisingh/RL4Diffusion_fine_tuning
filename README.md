## RL4Diffusion: Enhancing Reasoning in Diffusion-Based Language

# Abstract
We study reinforcement learning (RL) post-training for LLaDA, a Large Language Diffusion Model, to improve alignment with human intent. Unlike traditional autoregressive models, LLaDA leverages bidirectional context, conditioning token unmasking on the entire generated sequence. While LLaDA already outperforms Gemma2-9B and LLaMA-8B following supervised fine-tuning (SFT), it has not previously undergone RL fine-tuning. We introduce two RL fine-tuning pipelines: one based on Group Relative Policy Optimization (GRPO) and another on Score Entropy Policy Optimization (SEPO), designed for diffusion models with non-differentiable reward functions. After fine-tuning LLaDA for one epoch on 363 examples, we observe a marked 24.67% improvement.

# Code base

This repository, RL4Diffusion_fine_tuning, provides a pipeline for fine-tuning large language diffusion models using reinforcement learning. It adapts the GRPO (Group Relative Policy Optimization) and SEPO (Score Entropy Policy Optimization - https://arxiv.org/abs/2502.01384) fine tuning frameworks for the LLaDA-8B Instruct Large Language Diffusion Model (https://arxiv.org/abs/2502.09992). 

Code Overview:
1. main_sepo_finetune.py:
   - Main training scipt initializes models, loads data, computes rewards, and updates policies using SEPO
2. SEPO_LLaDA_wrapper.py:
    - Wraps LLaDA model to interface with SEPO fine tuning pipeline
3. reward_functions.py:
   - Implements reward models for exact match, instruction-following, factual consistency

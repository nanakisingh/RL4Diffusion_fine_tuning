## RL4Diffusion: Enhancing Reasoning in Diffusion-Based Language

# Abstract
We study reinforcement learning (RL) post-training for LLaDA, a Large Language Diffusion Model, to improve alignment with human intent. Unlike traditional autoregressive models, LLaDA leverages bidirectional context, conditioning token unmasking on the entire generated sequence. While LLaDA already outperforms Gemma2-9B and LLaMA-8B following supervised fine-tuning (SFT), it has not previously undergone RL fine-tuning. We introduce two RL fine-tuning pipelines: one based on Group Relative Policy Optimization (GRPO) and another on Score Entropy Policy Optimization (SEPO), designed for diffusion models with non-differentiable reward functions. After fine-tuning LLaDA for one epoch on 363 examples, we observe a marked 24.67% improvement.

<img width="1313" alt="Screen Shot 2025-04-29 at 4 45 24 PM" src="https://github.com/user-attachments/assets/03d3790e-fb1b-40d0-85c8-1d6426922e40" />

# Code base

This repository, RL4Diffusion_fine_tuning, provides a pipeline for fine-tuning large language diffusion models using reinforcement learning. It adapts the GRPO (Group Relative Policy Optimization) and SEPO (Score Entropy Policy Optimization - https://arxiv.org/abs/2502.01384) fine tuning frameworks for the LLaDA-8B Instruct Large Language Diffusion Model (https://arxiv.org/abs/2502.09992). 

Code Overview:
1. main_sepo_finetune.py:
   - Main training scipt initializes models, loads data, computes rewards, and updates policies using SEPO
2. SEPO_LLaDA_wrapper.py:
    - Wraps LLaDA model to interface with SEPO fine tuning pipeline
3. reward_functions.py:
   - Implements reward models for exact match, instruction-following, factual consistency


# LLaDA-GRPO, LLaDA-SEPO Performance

<img width="686" alt="Screen Shot 2025-04-30 at 9 30 32 PM" src="https://github.com/user-attachments/assets/3a7d0b10-55b1-4d9f-b39d-4ceeb9afbe55" />

<img width="611" alt="Screen Shot 2025-04-30 at 9 29 18 PM" src="https://github.com/user-attachments/assets/95a64075-12c0-4db8-9199-0220d9c50fcd" />


# Hyperparameter testing for LLaDA-8B Instruct


<img width="440" alt="Screen Shot 2025-04-30 at 9 30 40 PM" src="https://github.com/user-attachments/assets/15c88c8a-f355-45b9-bb82-e512cd096d5d" />

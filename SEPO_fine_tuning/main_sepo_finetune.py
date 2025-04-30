'''

    Main SEPO Finetune code 
    
    - Takes user arguments and finetunes LLaDA-8B Instruct with SEPO
    - Structure heavily inspired by original SEPO code
    - However, most helper functions were written by RL4Diffusion team (for LLaDA   model interface and to align code with math from SEPO paper)


'''

# Import libraries 
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import wandb
import os
import datetime
import json
import random
from utils import str2bool, set_seed
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import defined functions 
import SEPO_LLaDA_wrapper
import reward_functions

# Generate final outputs, intermediate logits and final sequence rewards 
# for batch of input prompts
def generate_samples(model, prompts, args, reward_model):
    final_outputs = []
    intermediate_logits = []
    all_rewards = []
    
    for prompt in prompts:
        with torch.no_grad():
            x, logits = model._sample_no_gradient(
                prompt=prompt,
                number_steps=args.steps_llada,
                sequence_length=args.sequence_length,
                eval_sp_size=args.batch_size,
                copy_flag_temp=args.copy_flag_temp
            )
        
        # Decode generated sequences
        responses = [model.tokenizer.decode(torch.argmax(xi, dim=-1)) for xi in x]
        # Calculate rewards
        rewards = reward_model.get_rewards([prompt]*len(responses), responses)
        
        final_outputs.append(x)
        intermediate_logits.append(logits)
        all_rewards.append(rewards)
    
    return final_outputs, intermediate_logits, all_rewards

# Normalize rewards across number of steps 
def normalize_rewards(rewards_list):
    eps_error=1e-8
    if len(rewards_list) > 1:
        rewards_tensor = torch.stack(rewards_list)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + eps_error)
        return [rewards_tensor[i] for i in range(rewards_tensor.shape[0])]
    return rewards_list

# Initialise reference, new and old models
def initialize_models(args):
    new_model = SEPO_LLaDA_wrapper.GRPO_LLaDA_Wrapper(
        model_name="GSAI-ML/LLaDA-8B-Instruct", 
        device=f"cuda:{args.gpu_number}"
    )
    
    old_model = SEPO_LLaDA_wrapper.GRPO_LLaDA_Wrapper(
        model_name="GSAI-ML/LLaDA-8B-Instruct",
        device=f"cuda:{args.gpu_number}"
    )
    
    ref_model = SEPO_LLaDA_wrapper.GRPO_LLaDA_Wrapper(
        model_name="GSAI-ML/LLaDA-8B-Instruct",
        device=f"cuda:{args.gpu_number}"
    )
    
    # Set requires_grad for reference and old parameters to False 
    for param in ref_model.parameters():
        param.requires_grad = False
        
    for param in old_model.model.parameters():
        param.requires_grad = False
        
    return new_model, old_model, ref_model

# Inspired by original SEPO code and modified to fit math from original paper
# Compure reward, entropy and KL Losses 
def compute_losses(new_logits, old_logits, ref_logits, rewards, new_model, old_model, args, timesteps, dt):

    # Initialize losses
    reward_loss = 0.0
    entropy_loss = 0.0
    kl_loss = 0.0
    
    # Move to device 
    timesteps = timesteps.to(new_model.device)
    
    # Loop through all sequences/generated logits for sequences
    for i in range(len(new_logits)):
        # Move to device 
        t = timesteps[i].to(new_model.device)
        move_chance_t = (1 - torch.exp(-t)).unsqueeze(-1).unsqueeze(-1).to(new_model.device)
        move_chance_s = (1 - torch.exp(-(t - dt))).unsqueeze(-1).unsqueeze(-1).to(new_model.device)
        
        # Calculate: probabilities for the old model 
        p_old = torch.softmax(old_logits[i].to(new_model.device), dim=-1).detach()
        old_dist = old_model._build_distrib(p_old, move_chance_t, move_chance_s).detach()
        old_dist[:, :, 4] = 1e-8  
        
        # Calculate: probabilities for the new model 
        p_new = torch.softmax(new_logits[i].to(new_model.device), dim=-1)
        new_dist = new_model._build_distrib(p_new, move_chance_t, move_chance_s)
        new_dist[:, :, 4] = 1e-8
        
        # Reward loss 
        curr_reward = old_model._compute_loss_sepo(rewards[i].to(new_model.device), new_dist, old_dist)
        reward_loss += curr_reward
        
        # Entropy bonus
        if args.entropy:
            entropy_loss += new_model._compute_entropy_bonus(p_new)
        
        # Taken from original SEPO code 
        # KL divergence (reference on same device)
        if args.kl_coeff > 0:
            kl_loss += F.kl_div(F.log_softmax(new_logits[i]).to(new_model.device), F.softmax(ref_logits[i].to(new_model.device), dim=-1).detach(),reduction='batchmean')
    
    return reward_loss, entropy_loss, kl_loss

# Main finetuning loop 
def fine_tune(new_model, reward_model, old_model, ref_model, args):

    # Load and batch prompts
    batched_prompts, _ = SEPO_LLaDA_wrapper.read_and_batch_aqua_data("aqua_saved.json", args.batch_size)
    
    # Intialize parameters
    eps = 1e-5
    dt = (1 - eps) / args.steps_llada
    optimizer = torch.optim.Adam(new_model.model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"----- Epoch {epoch} / {args.num_epochs} -----")
        
        # Reference model parameters = new model parameters
        ref_model.model.load_state_dict(new_model.model.state_dict())
        
        # Shuffle batch
        random.shuffle(batched_prompts)
        
        # Loop through batch
        for batch_idx, prompts in enumerate(batched_prompts):
            current_step = args.num_steps_sepo * epoch + batch_idx
        
            # Loop through each diffusion step 
            for step in range(args.num_steps_sepo):
                current_step = args.num_steps_sepo * epoch + step
                print(f"Step {current_step}, Epoch {epoch}")
                
                # Update old model to current new model state
                old_model.model.load_state_dict(new_model.model.state_dict())
                                
                # Old mode: generate samples + normalize rewards
                _, old_logits, old_rewards = generate_samples(old_model, prompts, args, combined_reward)
                old_rewards = normalize_rewards(old_rewards)
                            
                # New model: generate samples
                _, new_logits, new_rewards = generate_samples(new_model, prompts, args, combined_reward)
                            
                # Reference model: generate samples
                with torch.no_grad():
                    _, ref_logits, _ = generate_samples(ref_model, prompts, args)
                
                # Simulate timesteps - not actually used in LLaDA code/generation process but needed for SEPO
                timesteps = torch.linspace(1, eps, args.steps_llada + 1, device=new_model.device)
                            
                # GRPO optimization steps
                for mu_step in range(args.mu_sepo):
                    optimizer.zero_grad()
                    
                    # Compute losses
                    reward_loss, entropy_loss, kl_loss = compute_losses(new_logits, old_logits, ref_logits, old_rewards,new_model, old_model, args, timesteps, dt)
                    
                    # Normalize losses
                    batch_size = len(new_logits)
                    
                    # Determine all losses
                    # maximising reward - so put negative sign (taken from SEPO paper)
                    reward_loss = -reward_loss / batch_size  
                    entropy_loss = entropy_loss / batch_size
                    if args.kl_coeff > 0: 
                        kl_loss = kl_loss / batch_size
                    else:
                        kl_loss = 0.0
                    
                    # Update coefficients - inspired by SEPO paper
                    current_kl_coeff = update_coefficient(current_step,  args.kl_coeff_schedule_warmup, args.kl_coeff)
                    current_entropy_coeff = update_coefficient(current_step,args.entropy_coeff_schedule_warmup, args.entropy_coeff) if args.entropy else 0.0
                    
                    # Total loss - calculation from paper 
                    total_loss = (reward_loss + current_kl_coeff * kl_loss - current_entropy_coeff * entropy_loss)
                    
                    # Backpropagate and update
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
                    optimizer.step()
                    
                    # Logging
                    print(f"Step {current_step} Loss: {total_loss.item():.4f} "
                        f"(Reward: {-reward_loss.item():.4f}, "
                        f"KL: {kl_loss.item():.4f}, "
                        f"Entropy: {entropy_loss.item():.4f})")
                
                # Save checkpoint periodically
                if (current_step + 1) % args.save_every_n_steps == 0:
                    save_path = os.path.join(args.base_path, f'model_step{current_step+1}.ckpt')
                    torch.save(new_model.state_dict(), save_path)

# Inspired by SEPO paper 
def update_coefficient(step, warmup_steps, max_value):
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps * max_value
    return max_value

# Taken from original SEPO code
# Allows user to pass in training inputs 
def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument_group("General Run Arguments")
    parser.add_argument('--base_path', type=str, default='/data/user_data/jkalra/genai',
                      help="Base directory for model checkpoints and logs")
    parser.add_argument('--name', type=str, default='date_and_time',
                      help="Experiment name (default: current timestamp)")
    parser.add_argument('--save_every_n_steps', type=int, default=10,
                      help="Steps between checkpoint saves")
    parser.add_argument("--gpu_number", type=int, default=0,
                      help="GPU device ID (default: 0)")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for reproducibility")
    parser.add_argument('--batch_size', type=int, default=1,
                      help="Samples per training batch")
    parser.add_argument('--total_num_steps', type=int, default=128,
                      help="Total training steps")
    parser.add_argument('--copy_flag_temp', type=float, default=None,
                      help="Temperature for copy flag mechanism")
    parser.add_argument('--sequence_length', type=int, default=128,
                      help="Max tokens per generated sequence")

    # Optimizer settings
    optimizer_group = parser.add_argument_group("Optimizer Arguments")
    optimizer_group.add_argument('--learning_rate', type=float, default=1e-4,
                               help="Initial learning rate")
    optimizer_group.add_argument('--wd', type=float, default=0.0,
                               help="Weight decay (L2 regularization)")
    optimizer_group.add_argument('--gradnorm_clip', type=float, default=1.0,
                               help="Max gradient norm for clipping")

    # KL divergence settings
    kl_group = parser.add_argument_group("KL-Divergence Arguments")
    kl_group.add_argument("--put_kl", type=int, default=0,
                        help="Enable KL regularization (1=True, 0=False)")
    kl_group.add_argument("--truncate_kl", type=str2bool, default=True,
                        help="Truncate KL calculations")
    kl_group.add_argument('--truncate_steps', type=int, default=1,
                        help="Steps before KL truncation")
    kl_group.add_argument('--kl_coeff', type=float, default=1e-4,
                        help="KL loss weight")
    kl_group.add_argument('--kl_coeff_schedule_warmup', type=int, default=0,
                        help="KL coefficient warmup steps")

    # Entropy settings
    entropy_group = parser.add_argument_group("Entropy Bonus Arguments")
    entropy_group.add_argument('--entropy', type=str2bool, default=False,
                            help="Enable entropy bonus")
    entropy_group.add_argument('--entropy_coeff', type=float, default=1e-5,
                            help="Entropy loss weight")
    entropy_group.add_argument('--entropy_coeff_schedule_warmup', type=int, default=0,
                            help="Entropy coefficient warmup steps")

    # GRPO-specific settings
    grpo_group = parser.add_argument_group("SEPO Arguments")
    grpo_group.add_argument('--num_epochs', type=int, default=1,
                          help="Training epochs")
    grpo_group.add_argument('--num_steps_sepo', type=int, default=1,
                          help="Steps per SEPO update")
    grpo_group.add_argument("--sepo", type=int, default=1,
                          help="Policy update size control")
    grpo_group.add_argument("--mu_sepo", type=int, default=5,
                          help="Smoothing parameter")
    grpo_group.add_argument('--eps_sepo', type=float, default=0.2,
                          help="Epsilon parameter")
    grpo_group.add_argument('--steps_llada', type=int, default=128,
                          help="Steps per model generation")

    return parser.parse_args()

# Main code 
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, use_cuda=True)
    
    # Initialize models
    new_model, old_model, ref_model = initialize_models(args)
    reward_model = reward_functions.CombinedReward(device=f"cuda:{args.gpu_number}")
    
    # Run training
    fine_tune(new_model, reward_model, old_model, ref_model, args)
    
    print("Fine Tuning Complete")
'''

    SEPO LLaDA Wrapper
    
    - Inspired by original SEPO code
    - Holds all model methods, inference and gradient update functions 

'''

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from llada_model.generate import generate_ft
import torch.nn.functional as F
import numpy as np

# Import defined functions
import grpo_wrapper_helper

class SEPO_LLaDA_Wrapper(nn.Module):
    def __init__(self, model_name: str, device: str):
        super().__init__()
        
        self.device = device
        # Load in LLaDA model - used for reference, old and new policy models  
        self.model = AutoModel.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.bfloat16).to(device).eval()
        
        # Tokenizer for all inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()
    
    def _sample_prompt_prior(self, prompt, batch_size=1, gen_length=128, mask_id=126336):
        """
        Create initial input: [prompt tokens] + [MASK] * (gen_length)
        """
        if prompt: 
            prompts = prompt * batch_size
        else: 
            # Dummy prompt for testing
            prompts = ["What is the capital of France?"] * batch_size

        # Note: for LLaDA - batch size of 1 
        # Apply chat template
        batch_inputs = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}],add_generation_prompt=True,tokenize=False) for p in prompts]
        input_ids = [self.tokenizer(p)['input_ids'] for p in batch_inputs]
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids],batch_first=True).to(self.device)

        # Prompt length 
        prompt_length = input_ids.shape[1]

        # Now create final tensor with MASK tokens - thing that gets unmasked 
        full_input = torch.full((batch_size, prompt_length + gen_length),mask_id,dtype=torch.long,device=self.device
        )

        # Insert prompt tokens at the beginning
        full_input[:, :prompt_length] = input_ids

        return full_input
        
    
    # Taken from LLaDA code   
    def _llada_update_step(self, model, x, prompt_index, steps, block_start, block_end, temperature, cfg_scale, remasking, mask_id, dt, t, return_process=False):
        
        mask_index = (x == mask_id)

        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        confidence = torch.gather(F.softmax(logits.float(), dim=-1), dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j, block_start:block_end], k=1) 
            
            # selecting 1 token for simplicity
            select_index += block_start
            transfer_index[j, select_index] = True

        new_x = torch.where(transfer_index, x0, x)

        move_chance_t = 1 - torch.exp(-t)
        move_chance_t = move_chance_t[:, None, None]
        condt = t.squeeze(-1)
        copy_flag = (x != mask_id).to(x.dtype)

        if return_process:
            return new_x, x, condt, move_chance_t, copy_flag
        else:
            return new_x, None, None, None, None
    
    # Inspired by SEPO code - designed using paper 
    def _build_distrib(self, probs, move_chance_t, move_chance_s, mask_id=126336):
        q_xs = probs * (move_chance_t - move_chance_s)
        mask_value = move_chance_s.squeeze(-1)  
        q_xs[:, :, mask_id] = mask_value
        return q_xs
    
    # Inspired by SEPO code - designed using paper 
    def _compute_loss_sepo(self, rew, new_prob_dist, old_prob_dist, epsilon=0.2):
        bsz = rew.shape[0]

        ratio = new_prob_dist / old_prob_dist
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        inner_sum = (ratio * old_prob_dist).sum(dim=(1, 2)) 
        clipped_inner_sum = (clipped_ratio * old_prob_dist).sum(dim=(1, 2))

        loss = (1 / bsz) * (rew * inner_sum).sum()
        clipped_loss = (1 / bsz) * (rew * clipped_inner_sum).sum()

        return torch.min(loss, clipped_loss)

    # Inspired by SEPO code - designed using paper 
    def _compute_entropy_bonus(self, pi_y, eps_error=1e-8):
  
        # Remove last token
        pi_y_truncated = pi_y[:, :, :-1]

        # Normalize 
        norm = pi_y_truncated.sum(dim=-1, keepdim=True)
        pi_y_truncated = torch.where(norm > 0, pi_y_truncated / norm, torch.zeros_like(pi_y_truncated))

        # Compute entropy
        log_probs = torch.log(torch.clamp(pi_y_truncated, min=eps_error))
        entropy = -(pi_y_truncated * log_probs).sum(dim=-1).mean()

        return entropy
    
    # Used in main code - one prompt at a time
    def _sample_no_gradient(self, prompt, number_steps=16, sequence_length=128, eval_sp_size=None, copy_flag_temp=None):
                
        m = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
        
        return generate_ft(self.model, input_ids, steps=number_steps, gen_length=sequence_length, block_length=32)
   
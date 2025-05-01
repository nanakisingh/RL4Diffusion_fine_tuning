# train_llada_grpo.py
from datasets import load_dataset
from functools import partial
from types import MethodType
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from lladacode.generate import generate as llada_sample      # function
from lladacode import get_log_likelihood
from peft import LoraConfig, get_peft_model, TaskType
from transformers import logging
logging.set_verbosity_debug()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


# ── 1. model ----------------------------------------------------------------
model_name = "GSAI-ML/LLaDA-8B-Instruct"
device = 'cuda'
tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
policy     = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                trust_remote_code=True).to(device)

def llada_generate(self, input_ids, **kw):
    out = llada_sample(
        self, input_ids,
        steps=kw.get("steps", 128),
        gen_length=kw.get("max_new_tokens", 128),
        block_length=kw.get("block_length", 32),
        temperature=kw.get("temperature", 0.9),
        cfg_scale=kw.get("cfg_scale", 0.0),
    )
    return out

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,    # fine-tuning, not inference
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  
)

policy = get_peft_model(policy, lora_cfg)
print(policy.print_trainable_parameters())

policy.generate = MethodType(llada_generate, policy)
policy.generation_config = GenerationConfig.from_model_config(policy.config)

orig_forward = policy.forward

def forward_with_prune(self, input_ids, attention_mask=None, logits_to_keep=None, **kwargs):
    outputs = orig_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    if logits_to_keep is not None:
        # keep only the last `logits_to_keep` timesteps
        outputs.logits = outputs.logits[:, -logits_to_keep:, :]
    return outputs

policy.forward = MethodType(forward_with_prune, policy)

# ── 2. data ------------------------------------------------------------------
ds = load_dataset("gsm8k", "main", split="train[:10%]", cache_dir="./cache")

def fmt(example):
    msgs = [{"role": "user", "content": example["question"]}]
    example["prompt"] = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False)
    return example

ds = ds.map(fmt, remove_columns=["question"])
answers = ds["answer"]                       # capture gold answers list

# ── 3. reward ----------------------------------------------------------------
def reward_fn(completions, prompts, answer):
    scores = []
    for c, a in zip(completions, answer):
        try:
            pred = int(c.strip().split('\n')[-1].split()[-1].strip('.'))
            gold = int(a.split('####')[-1])
            em = 1.0 if pred == gold else 0.0
        except Exception:
            em = 0.0
        scores.append(em)
    return scores

# ── 4. config ----------------------------------------------------------------
cfg = GRPOConfig(
    output_dir                 = "/data/user_data/jkalra/genai/llada-8b-gsm8k-grpo",
    per_device_train_batch_size= 8,     # one prompt per forward pass
    # gradient_accumulation_steps= 2,     # accumulate to global=2
    num_generations            = 8,     # two completions per prompt
    # learning_rate              = 5e-6,
    num_train_epochs           = 1,
    max_prompt_length          = 128,
    # beta                       = 0.03,
    # temperature                = 0.7,
    logging_steps              = 1,
    report_to=["tensorboard"],
    push_to_hub=True,
    save_strategy = "steps",               # save every `save_steps`
    save_steps    = 100,                  # checkpoint every 1 000 updates :contentReference[oaicite:2]{index=2}
    save_total_limit = 3,
    disable_tqdm = False,
)

trainer = GRPOTrainer(
    model         = policy,
    processing_class     = tokenizer,             
    args          = cfg,
    train_dataset = ds,
    reward_funcs  = [reward_fn],
)

trainer.train()
trainer.save_model(cfg.output_dir)
trainer.push_to_hub(dataset_name="gsm8k")

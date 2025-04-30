from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "GSAI-ML/LLaDA-8B-Instruct"
save_path = " "

# Load from Hugging Face hub
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Save locally for future use
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

'''

    LLM Reward Models for SEPO: 
    
    1. Factual consistency model
    2. Instruction reward model
    
    - Tokenize both input sequences before passing through HuggingFace models 
    - Weight both scalar values with predefined weights

'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

class CombinedReward:
    
    def __init__(self, device="cuda"):
        self.device = device
        
        # Instruction reward model: OpenAssistant/reward-model-deberta-v3-large
        self.instruction_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large").to(device)
        self.instruction_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large")
        
        # Factual consistency model: microsoft/deberta-v2-xxlarge-mnli
        self.factcheck_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xxlarge-mnli").to(device)
        self.factcheck_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge-mnli")

    def get_rewards(self, prompts, responses):
        
        # Prompt - singular prompt at a time because of LLaDA restriction 
        
        # Instruction scores
        instr_inputs = self.instruction_tokenizer(prompts, responses, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            instr_scores = self.instruction_model(**instr_inputs).logits.squeeze(-1)
        
        # Factual scores
        fact_inputs = self.factcheck_tokenizer(prompts, responses, padding=True, truncation=True,max_length=512, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            fact_logits = self.factcheck_model(**fact_inputs).logits
            # Convert to probabilities with softmax
            fact_scores = torch.softmax(fact_logits, dim=1)[:, 2]  
        
        # Weight both scores - optimal values: 0.7, 0.3 
        weighted_reward = (0.7 * instr_scores) + (0.3 * fact_scores)
        
        return weighted_reward
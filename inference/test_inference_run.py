'''

    Baseline Hyperparameter Experimentation Code
    
    - Perform parameter testing with 5 parameters with LLaDA-8B Instruct: 
        steps/gen_length, block_length, temperature, cfg_scale, remasking
    - Read and format data from AQuA dataset 
    - Run inference and store results in .json files for further evaluation


'''




import torch
from generate import generate
from transformers import AutoTokenizer, AutoModel
from read_aqua_dataset import read_aqua_data
import json
import argparse 

def inference(args):
    
    device = 'cuda'
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    prompts, responses = read_aqua_data('test.json')
    
    num_iters = args.num_iters
    
    if args.exp_type == "bl":
        # Parameter: Block Length
        output_files = ["bl_32_output.json", "bl_64_output.json", "bl_1_output.json", "bl_128_output.json"]
        block_len = [32,64,1,128]
    
        for i in range(len(output_files)):
            curr_output_file = output_files[i]
            curr_block_len = block_len[i]
            
            print("Iteration type: ", curr_output_file)
        
            with open(curr_output_file, "w") as f: 
                    
                for i in range(num_iters):
                    print(f"Iteration: {i}")
                    prompt = prompts[i]
                    curr_answer = responses[i]
                    
                    # print("\nPrompt: ", prompt)
                    n_p = "\nPrompt: " + prompt
                    
                    # number of steps/gen_length
                    rounded_steps = 128 # fixed length
                    # ((len(responses[i]) +31)//32)*32
                    
                    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
                    m = [{"role": "user", "content": prompt}, ]
                    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = tokenizer(prompt)['input_ids']
                    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
                
                    # TODO: Change paramaters: steps/gen_length, block_length, temperature, cfg_scale, remasking
                    out = generate(model, input_ids, steps=rounded_steps, gen_length=rounded_steps, block_length=curr_block_len, temperature=0., cfg_scale=0., remasking='low_confidence')
                    
                    g_o = "\nGenerated output:\n" + tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    t_o = "\nTrue output:" + curr_answer
                    
                    f.write(n_p)
                    f.write("\n")
                    f.write(g_o)
                    f.write("\n")
                    f.write(t_o)
                    f.write("\n")
                    f.write("\n")
                    f.write("----------------------------------------------------------------------------------------------")
    elif args.exp_type == "cfg":
        # Parameter: CFG 
        output_files_2 = ["cfg_01_output.json", "cfg_05_output.json", "cfg_08_output.json", "cfg_1_output.json"]
        cfg_val = [0.1, 0.5, 0.8, 1] 
        
        for i in range(len(output_files_2)):
            curr_output_file = output_files_2[i]
            curr_cfg_val = cfg_val[i]
            
            print("Iteration type: ", curr_output_file)
        
            with open(curr_output_file, "w") as f: 
                
                for i in range(num_iters):
                    print(f"Iteration: {i}")
                    prompt = prompts[i]
                    curr_answer = responses[i]
                    
                    # print("\nPrompt: ", prompt)
                    n_p = "\nPrompt: " + prompt
                    
                    # number of steps/gen_length
                    rounded_steps = 128 # fixed length
                    # ((len(responses[i]) +31)//32)*32
                    
                    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
                    m = [{"role": "user", "content": prompt}, ]
                    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = tokenizer(prompt)['input_ids']
                    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
                
                    # TODO: Change paramaters: steps/gen_length, block_length, temperature, cfg_scale, remasking
                    out = generate(model, input_ids, steps=rounded_steps, gen_length=rounded_steps, block_length=64, temperature=0., cfg_scale=curr_cfg_val, remasking='low_confidence')
                
                    g_o = "\nGenerated output:\n" + tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    t_o = "\nTrue output:" + curr_answer
                    
                    f.write(n_p)
                    f.write("\n")
                    f.write(g_o)
                    f.write("\n")
                    f.write(t_o)
                    f.write("\n")
                    f.write("\n")
                    f.write("----------------------------------------------------------------------------------------------")
                
    elif args.exp_type == "temp":
        # Parameter: Temperature 
        output_files_3 = ["temp_01_output.json", "temp_05_output.json",  "temp_09_output.json"]
        temp_val = [0.1, 0.5, 0.9] 
        
        for i in range(len(output_files_3)):
            curr_output_file = output_files_3[i]
            curr_temp_val = temp_val[i]
            
            print("Iteration type: ", curr_output_file)
        
            with open(curr_output_file, "w") as f: 
                
                for i in range(num_iters):
                    print(f"Iteration: {i}")
                    prompt = prompts[i]
                    curr_answer = responses[i]
                    
                    # print("\nPrompt: ", prompt)
                    n_p = "\nPrompt: " + prompt
                    
                    # number of steps/gen_length
                    rounded_steps = 128 # fixed length
                    # ((len(responses[i]) +31)//32)*32
                    
                    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
                    m = [{"role": "user", "content": prompt}, ]
                    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = tokenizer(prompt)['input_ids']
                    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
                
                    # TODO: Change paramaters: steps/gen_length, block_length, temperature, cfg_scale, remasking
                    out = generate(model, input_ids, steps=rounded_steps, gen_length=rounded_steps, block_length=64, temperature=curr_temp_val, cfg_scale=0.0, remasking='low_confidence')
                    
                    g_o = "\nGenerated output:\n" + tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    t_o = "\nTrue output:" + curr_answer
                    
                    f.write(n_p)
                    f.write("\n")
                    f.write(g_o)
                    f.write("\n")
                    f.write(t_o)
                    f.write("\n")
                    f.write("\n")
                    f.write("----------------------------------------------------------------------------------------------")
    
    elif args.exp_type == "remasking":
        # Parameter: Remasking 
        output_files_4 = ["remasking_random_output.json", "remasking_lc_output.json"]
        remasking_val = ['random', 'low_confidence'] 
        
        for i in range(len(output_files_4)):
            curr_output_file = output_files_4[i]
            curr_remasking_val = remasking_val[i]
            
            print("Iteration type: ", curr_output_file)
        
            with open(curr_output_file, "w") as f: 
                
                for i in range(num_iters):
                    print(f"Iteration: {i}")
                    prompt = prompts[i]
                    curr_answer = responses[i]
                    
                    # print("\nPrompt: ", prompt)
                    n_p = "\nPrompt: " + prompt
                    
                    # number of steps/gen_length
                    rounded_steps = 128 # fixed length
                    # ((len(responses[i]) +31)//32)*32
                    
                    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
                    m = [{"role": "user", "content": prompt}, ]
                    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = tokenizer(prompt)['input_ids']
                    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
                
                    # TODO: Change paramaters: steps/gen_length, block_length, temperature, cfg_scale, remasking
                    out = generate(model, input_ids, steps=rounded_steps, gen_length=rounded_steps, block_length=64, temperature=0.0, cfg_scale=0.0, remasking=curr_remasking_val)
                
        
                    g_o = "\nGenerated output:\n" + tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    t_o = "\nTrue output:" + curr_answer
                    
                    f.write(n_p)
                    f.write("\n")
                    f.write(g_o)
                    f.write("\n")
                    f.write(t_o)
                    f.write("\n")
                    f.write("\n")
                    f.write("----------------------------------------------------------------------------------------------")
    
    elif args.exp_type == "num_steps":
        # Parameter: Number of steps 
        output_files_5 = ["num_steps_256_output.json", "num_steps_192_output.json", "num_steps_128_output.json", "num_steps_64_output.json"]
        num_steps_val = [256, 192, 128, 64]
            
        for i in range(len(output_files_5)):
            curr_output_file = output_files_5[i]
            curr_num_steps_val = num_steps_val[i]
            
            print("Iteration type: ", curr_output_file)
        
            with open(curr_output_file, "w") as f: 
                
                # for i in range(len(prompts)):
                for i in range(num_iters):
                    print(f"Iteration: {i}")
                    prompt = prompts[i]
                    curr_answer = responses[i]
                    
                    # print("\nPrompt: ", prompt)
                    n_p = "\nPrompt: " + prompt
                    
                    # number of steps/gen_length
                    rounded_steps = curr_num_steps_val # CHANGED length
                    # ((len(responses[i]) +31)//32)*32
                    
                    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
                    m = [{"role": "user", "content": prompt}, ]
                    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = tokenizer(prompt)['input_ids']
                    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
                
                    # TODO: Change paramaters: steps/gen_length, block_length, temperature, cfg_scale, remasking
                    out = generate(model, input_ids, steps=rounded_steps, gen_length=rounded_steps, block_length=64, temperature=0.0, cfg_scale=0.0, remasking='low_confidence')
                
        
                    
                    g_o = "\nGenerated output:\n" + tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    t_o = "\nTrue output:" + curr_answer
                    
                    f.write(n_p)
                    f.write("\n")
                    f.write(g_o)
                    f.write("\n")
                    f.write(t_o)
                    f.write("\n")
                    f.write("\n")
                    f.write("----------------------------------------------------------------------------------------------")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--exp_type", choices=["bl", "cfg", "temp", "remasking", "num_steps"], required=True)
    args = parser.parse_args()
    inference(args)
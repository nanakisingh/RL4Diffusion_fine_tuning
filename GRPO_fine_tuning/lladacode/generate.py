import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompts,                   # (B, L)
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.9,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    print("temperature: ", temperature)
    B, L = prompts.size()
    T = L + gen_length
    device = prompts.device

    # initialize full batch to [MASK]
    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :L] = prompts                         # copy all prompts in one go
    prompt_mask = x != mask_id                 # (B, T) mask of prompt tokens

    # compute block / step counts
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    # for each block, we will unmask the tokens in the block
    for bidx in range(num_blocks):
        # which tokens in this block are still masked?
        block_slice = slice(L + bidx*block_length, L + (bidx+1)*block_length)
        block_mask = (x[:, block_slice] == mask_id)               # (B, block_length)
        # how many to unmask each step
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)  # (B, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)                          # (B, T)

            # model forward with optional CFG
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_mask] = mask_id
                logits = model(torch.cat([x, un_x], dim=0)).logits
                B2 = logits.size(0) // 2
                real, uncond = torch.split(logits, B2, dim=0)
                logits = uncond + (cfg_scale + 1) * (real - uncond)
            else:
                logits = model(x).logits                        # (B, T, V)

            # sample and compute confidence
            noisy = add_gumbel_noise(logits, temperature)       # (B, T, V)
            x0 = noisy.argmax(-1)                               # (B, T)

            if remasking == "low_confidence":
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                conf = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                conf = torch.rand((B, T), device=device)

            # prevent future blocks from being selected
            conf[:, L + (bidx+1)*block_length :] = -float("inf")
            # only masked positions are allowed
            conf = torch.where(mask_index, conf, -float("inf"))

            # pick top-k per batch element
            # we need the max k to topk once
            max_k = int(num_transfer[:, step].max().item())
            topv, topi = conf.topk(max_k, dim=-1)               # (B, max_k)
            # now zero-mask any positions beyond each element’s k
            range_k = torch.arange(max_k, device=device).unsqueeze(0)
            mask_k = range_k < num_transfer[:, step].unsqueeze(1)  # (B, max_k)
            sel_idx = topi[mask_k]                                # (sum_k,)
            sel_batch = torch.nonzero(mask_k, as_tuple=True)[0]   # which batch each belongs to

            # scatter update
            transfer = torch.zeros_like(mask_index)
            transfer[sel_batch, sel_idx] = True
            x = torch.where(transfer, x0, x)

    return x


def main():
    device = 'cuda'
    model_name = '/data/user_data/jkalra/genai/llada-8b-gsm8k-grpo'
    print(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    initial_prompt = """
    Maya and Leo loaded crates into the warehouse at 180 crates per hour each, so together they did
    180 × 2 = <<1802=360>>360 crates per hour.
    After 5 hours, they’d moved
    360 × 5 = <<3605=1800>>1800 crates.
    The warehouse holds 5000 crates, so remaining crates were
5000 − 1800 = <<5000-1800=3200>>3200 crates.
Then 3 more workers joined Maya and Leo, making 2 + 3 = <<2+3=5>>5 loaders.
At 180 crates per person, they moved
180 × 5 = <<180*5=900>>900 crates per hour.
To finish 3200 crates took
3200 / 900 = <<3200/900=3.555…>>3.56≈ <<3200/900=3.56>>3.56 hours.
Total time = 5 + 3.56 ≈ <<5+3.56=8.56>>8.56 hours.
    """
    prompt = initial_prompt + " Stella and Twinkle are filling up a truck with a capacity of 6000 stone blocks at the rate of 250 blocks per hour per person. They work for four hours and are then joined by 6 other people who also work at the same rate. How many hours did filling the truck take?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=128, temperature=0, cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
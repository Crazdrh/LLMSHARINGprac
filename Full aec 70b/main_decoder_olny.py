# main_decode_only.py

import torch
from gpt_tokenizer import GPTDeepSeekTokenizer
from gpt_config import GPT70BConfig
from gpt_model import GPT70B

if __name__ == "__main__":
    # 1) Instantiate decode-only tokenizer
    tokenizer = GPTDeepSeekTokenizer()

    # 2) Suppose we already got token IDs from somewhere else
    #    For demonstration, let's just make some up
    input_ids_list = [101, 102, 103]  # "pretend" these are real tokens
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)  # shape (1, 3)

    # 3) Build a GPT config
    #    We'll still set vocab_size = tokenizer.get_vocab_size(),
    #    even though we don't have an encode method in this wrapper.
    vocab_size = tokenizer.get_vocab_size()
    config = GPT70BConfig(vocab_size=vocab_size, embed_dim=512, num_heads=8, num_layers=4)

    # 4) Instantiate the GPT model
    model = GPT70B(config)

    # 5) Forward pass with dummy input_ids
    logits, _ = model(input_ids)

    # Now let's say the model generated a next token:
    # (In reality, you'd do sampling or beam search from logits.)
    next_token_id = logits[0, -1, :].argmax().item()

    # 6) Decode the output token ID to text
    output_text = tokenizer.decode([next_token_id])
    print("Decoded text:", output_text)

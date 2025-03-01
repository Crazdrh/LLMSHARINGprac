############################################################
# COA-GPT: A HIGH-LEVEL TRANSFORMER FOR COURSE-OF-ACTION   #
# PLANNING IN REAL-WORLD MILITARY OPERATIONS               #
# -------------------------------------------------------- #
# This single Python file integrates:                      #
#  1) Tokenizer (BPE-based)                                #
#  2) Transformer Model (Encoder-Decoder or Decoder-Only)  #
#  3) COA Dataset + Dataloader                             #
#  4) Training Script + Example Usage                      #
# -------------------------------------------------------- #
############################################################

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# For real-world usage, install or import advanced tokenizers
# pip install tokenizers
from tokenizers import Tokenizer, trainers, models, normalizers, pre_tokenizers, processors


#########################################
# 1. TOKENIZER: Build / Load BPE Model #
#########################################
def build_coa_gpt_tokenizer(
    domain_corpus_paths,
    vocab_size=32000,
    special_tokens=["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"],
):
    """
    Creates a Byte-Pair Encoding (BPE) tokenizer for the COA-GPT system,
    trained on domain-specific (military) texts. This helps encode acronyms,
    equipment codes, coordinate info, etc.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    custom_tokens = ["[COA_START]", "[COA_END]", "OBJ_Lion", "x:", "y:", "M1A2_SEPv3"]
    for tok in custom_tokens:
        tokenizer.add_special_tokens([tok])
    # Normalization pipeline
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),        # Normalizes Unicode
        normalizers.Lowercase()   # Converts all to lowercase
    ])

    # Pre-tokenization splits on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation()
    ])

    # Trainer - determines how the BPE merges are learned
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # Train on all domain corpora
    tokenizer.train(files=domain_corpus_paths, trainer=trainer)

    # Post-processing: define how to handle [CLS] and [SEP] when encoding
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]"))
        ]
    )

    return tokenizer


##########################################
# 2. DATASET + DATALOADER FOR TRAINING  #
##########################################
class COADataset(Dataset):
    """
    A dataset that pairs an input text (mission or partial plan) with a
    target COA text. Each item is encoded using the COA-GPT tokenizer.
    """

    def __init__(self, data, tokenizer, max_length=128):
        """
        :param data: list of {"input_text": str, "target_text": str}
        :param tokenizer: a trained COA-GPT tokenizer
        :param max_length: maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_text = sample["input_text"]
        target_text = sample["target_text"]

        # Encode input text
        input_enc = self.tokenizer.encode(input_text)
        # Encode target text
        target_enc = self.tokenizer.encode(target_text)

        # Convert to torch tensors
        input_ids = torch.tensor(
            input_enc.ids[: self.max_length], dtype=torch.long
        )
        target_ids = torch.tensor(
            target_enc.ids[: self.max_length], dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }


##########################################
# 3. TRANSFORMER ARCHITECTURE COMPONENTS #
##########################################

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism with:
      - Query, Key, Value projections
      - Scaled dot-product attention
      - Residual connection will be handled outside
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        B, T, _ = x.size()
        # Project to Q, K, V
        Q = self.W_q(x).view(B, T, self.num_heads, self.depth).transpose(1,2)
        K = self.W_k(x).view(B, T, self.num_heads, self.depth).transpose(1,2)
        V = self.W_v(x).view(B, T, self.num_heads, self.depth).transpose(1,2)

        # Scaled dot product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.depth)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, num_heads, seq_len, depth)

        # Reshape back
        out = out.transpose(1,2).contiguous().view(B, T, self.d_model)
        return self.fc_out(out)


class FeedForwardNetwork(nn.Module):
    """
    Standard two-layer FFN with ReLU activation, used after attention.
    """
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    A single block that combines:
      - Multi-head self-attention
      - Residual connections
      - Layer normalization
      - FeedForward network
    """
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention + residual
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding, makes the model aware of
    sequence order (time steps).
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


########################################
# 4. COA-GPT MODEL (DECODER-ONLY STYLE) #
########################################
class COAGPTModel(nn.Module):
    """
    A high-level GPT-style model for Course-of-Action generation:
      - Embedding layer
      - Positional encoding
      - A stack of Transformer blocks
      - Final linear -> vocab logits
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, hidden_dim=2048, num_layers=6):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """
        :param input_ids: (batch_size, seq_len)
        :return: logits of shape (batch_size, seq_len, vocab_size)
        """
        # 1) Token embedding
        x = self.token_embed(input_ids)  # (B, T, d_model)

        # 2) Add positional encoding
        x = self.pos_encoding(x)         # (B, T, d_model)

        # 3) Pass through stacked Transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, d_model)

        # 4) Final linear layer -> (B, T, vocab_size)
        logits = self.fc_out(x)
        return logits


####################################
# 5. TRAINING LOOP & ENTRY POINT  #
####################################
def train_coa_gpt(model, dataloader, epochs=3, lr=1e-4, device="cuda"):
    """
    Trains the COA-GPT model using cross-entropy on the
    next-token prediction task.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # Prepare data
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            # Forward pass -> predictions
            logits = model(input_ids)  # (B, T, vocab_size)

            # Flatten logits and targets to compute cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0  # if 0 is our PAD or out-of-range
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx+1) % 50 == 0:
                avg_loss = total_loss / (batch_idx+1)
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")

        print(f"Epoch {epoch+1} complete. Avg Loss = {total_loss/len(dataloader):.4f}")


def main():
    # 1) Suppose you have domain text files for training the tokenizer
    corpus_files = [
        "path/to/military_data_1.txt",
        "path/to/military_data_2.txt"
    ]
    # Build or load a COA tokenizer
    if not os.path.exists("coa_gpt_bpe_tokenizer.json"):
        print("Training a new tokenizer...")
        tokenizer_obj = build_coa_gpt_tokenizer(corpus_files, vocab_size=32000)
        tokenizer_obj.save("coa_gpt_bpe_tokenizer.json")
    else:
        print("Loading existing tokenizer...")
        tokenizer_obj = Tokenizer.from_file("coa_gpt_bpe_tokenizer.json")

    # 2) Prepare your training data
    # Hypothetical examples of input->target pairs
    train_data = [
        {
            "input_text": "Mission: Observe Bridge Bobcat. Enemy possible at x:75 y:30.",
            "target_text": "COA: Dispatch Recon to x:75 y:30 with overwatch from M1A2 at x:72 y:26."
        },
        {
            "input_text": "Mission: Seize OBJ_Lion. Air threat reported at x:123 y:76.",
            "target_text": "COA: Deploy UH-60 to scout x:123 y:76. Armor crosses Bridge Bear."
        },
        # Add thousands or millions of lines for real training
    ]

    dataset = COADataset(train_data, tokenizer_obj, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3) Build the COA-GPT model
    vocab_size = tokenizer_obj.get_vocab_size()
    model = COAGPTModel(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        hidden_dim=2048,
        num_layers=6
    )

    # 4) Train
    train_coa_gpt(
        model=model,
        dataloader=dataloader,
        epochs=3,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 5) Save the trained model weights
    torch.save(model.state_dict(), "coa_gpt_model.pt")
    print("Model training complete and weights saved to 'coa_gpt_model.pt'.")


if __name__ == "__main__":
    main()

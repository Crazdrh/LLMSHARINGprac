
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, trainers, models, normalizers, pre_tokenizers, processors

def build_coa_gpt_tokenizer(domain_corpus_paths, vocab_size=50000, special_tokens=["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()])
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files=domain_corpus_paths, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))]
    )
    return tokenizer

class COADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        input_enc = self.tokenizer.encode(sample["input_text"])
        target_enc = self.tokenizer.encode(sample["target_text"])
        input_ids = torch.tensor(input_enc.ids[:self.max_length], dtype=torch.long)
        target_ids = torch.tensor(target_enc.ids[:self.max_length], dtype=torch.long)
        return {"input_ids": input_ids, "target_ids": target_ids}

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, _ = x.size()
        Q = self.W_q(x).view(B, T, self.num_heads, self.depth).transpose(1,2)
        K = self.W_k(x).view(B, T, self.num_heads, self.depth).transpose(1,2)
        V = self.W_v(x).view(B, T, self.num_heads, self.depth).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.depth)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B, T, self.d_model)
        return self.fc_out(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class COAGPT70B(nn.Module):
    def __init__(self, vocab_size, d_model=12288, num_heads=96, hidden_dim=49152, num_layers=40):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, input_ids):
        x = self.token_embed(input_ids)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        logits = self.fc_out(x)
        return logits
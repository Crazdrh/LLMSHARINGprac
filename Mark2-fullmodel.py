import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW


##############################################################################
# 1) EXTREMELY ADVANCED TOKENIZER: COATextTokenizer
##############################################################################
class COATextTokenizer:
    """
    In a real system, you'd load a subword / BPE vocab from disk.
    This is a placeholder for demonstration, called COATextTokenizer.
    """

    def __init__(self, vocab=None, unk_token="[UNK]", pad_token="[PAD]", max_vocab_size=32000):
        if vocab is None:
            # Minimal demonstration vocab
            vocab = {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "hello": 4,
                "world": 5,
                "this": 6,
                "is": 7,
                "a": 8,
                "test": 9,
                "sequence": 10,
                "translation": 11,
                "example": 12,
            }
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.unk_token = unk_token
        self.unk_id = vocab.get(unk_token, 1)
        self.pad_token = pad_token
        self.pad_id = vocab.get(pad_token, 0)
        self.max_vocab_size = max_vocab_size

    def tokenize(self, text):
        """
        Naive whitespace split + lowercase. Real usage: advanced subword merges.
        """
        tokens = text.lower().split()
        token_ids = []
        for t in tokens:
            token_ids.append(self.vocab.get(t, self.unk_id))
        return token_ids

    def decode(self, token_ids):
        """
        Convert token IDs back to strings (space-separated).
        """
        return " ".join([self.inv_vocab.get(tid, self.unk_token) for tid in token_ids])


##############################################################################
# 2) COAEncoder: The Transformer Encoder Side
##############################################################################
class COAMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_out = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (B, T, d_model)
        mask: optional (B, T, T) or (B, 1, T, T)
        """
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape for multi-head
        Q = Q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        att_out = probs @ V
        att_out = att_out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.w_out(att_out)
        out = self.resid_dropout(out)
        return out


class COAPositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class COAEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = COAMultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = COAPositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        norm_x = self.ln1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + attn_out

        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x


class COAEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            COAEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, src_ids, src_mask=None):
        """
        src_ids: (B, T_src)
        src_mask: optional (B, T_src, T_src)
        returns: (B, T_src, d_model)
        """
        B, T_src = src_ids.shape
        assert T_src <= self.max_seq_len

        x = self.token_emb(src_ids)
        positions = torch.arange(T_src, device=src_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)

        x = x + pos_emb
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=src_mask)

        x = self.ln_f(x)
        return x


##############################################################################
# 3) COADecoder: The Transformer Decoder Side
##############################################################################
class COADecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = COAMultiHeadAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = COAMultiHeadAttention(d_model, n_heads, dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = COAPositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        """
        x: (B, T_tgt, d_model)
        enc_out: (B, T_src, d_model)
        tgt_mask: (B, T_tgt, T_tgt)
        memory_mask: (B, T_tgt, T_src)
        """
        # 1) Masked Self-Attn
        norm_x = self.ln1(x)
        self_attn_out = self.self_attn(norm_x, norm_x, norm_x, mask=tgt_mask)
        x = x + self_attn_out

        # 2) Cross-Attn
        norm_x = self.ln2(x)
        cross_attn_out = self.cross_attn(norm_x, enc_out, enc_out, mask=memory_mask)
        x = x + cross_attn_out

        # 3) FFN
        norm_x = self.ln3(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out

        return x


class COADecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            COADecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, tgt_ids, enc_out, tgt_mask=None, memory_mask=None):
        """
        tgt_ids: (B, T_tgt)
        returns: (B, T_tgt, vocab_size)
        """
        B, T_tgt = tgt_ids.shape
        assert T_tgt <= self.max_seq_len

        x = self.token_emb(tgt_ids)
        positions = torch.arange(T_tgt, device=tgt_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)

        x = x + pos_emb
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits


##############################################################################
# 4) COAGPT: Full Encoder-Decoder
##############################################################################
class COAGPT(nn.Module):
    """
    A seq2seq Transformer named 'COA-GPT' with:
      - COAEncoder
      - COADecoder
    """

    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=256,
            n_heads=4,
            d_ff=1024,
            num_encoder_layers=3,
            num_decoder_layers=3,
            max_seq_len=512,
            dropout=0.1
    ):
        super().__init__()
        self.encoder = COAEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        self.decoder = COADecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src_ids: (B, T_src)
        tgt_ids: (B, T_tgt)
        returns: (B, T_tgt, tgt_vocab_size)
        """
        enc_out = self.encoder(src_ids, src_mask=src_mask)
        logits = self.decoder(tgt_ids, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return logits


##############################################################################
# 5) DATASET & DATALOADER: Reading from a JSON lines file
##############################################################################
class COADataset(Dataset):
    """
    Expects a JSON lines file with records like:
      {"source": "Hello world", "target": "Bonjour le monde"}
    """

    def __init__(self, json_file_path, src_tokenizer, tgt_tokenizer, max_len=64):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.samples = []

        with open(json_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        src_text = item["source"]
        tgt_text = item["target"]
        # Tokenize
        src_ids = self.src_tokenizer.tokenize(src_text)[: self.max_len]
        tgt_ids = self.tgt_tokenizer.tokenize(tgt_text)[: self.max_len]

        # Convert to torch
        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
        return (src_ids, tgt_ids)


def collate_fn(batch):
    src_batch = [b[0] for b in batch]
    tgt_batch = [b[1] for b in batch]
    max_src_len = max(x.size(0) for x in src_batch)
    max_tgt_len = max(x.size(0) for x in tgt_batch)

    padded_src = []
    padded_tgt = []
    for s, t in zip(src_batch, tgt_batch):
        ps = F.pad(s, (0, max_src_len - s.size(0)), value=0)
        pt = F.pad(t, (0, max_tgt_len - t.size(0)), value=0)
        padded_src.append(ps.unsqueeze(0))
        padded_tgt.append(pt.unsqueeze(0))

    padded_src = torch.cat(padded_src, dim=0)  # (B, max_src_len)
    padded_tgt = torch.cat(padded_tgt, dim=0)  # (B, max_tgt_len)
    return padded_src, padded_tgt


##############################################################################
# 6) TRAINING LOOP
##############################################################################
def create_causal_mask(sz):
    """
    Create an upper-triangle mask (causal) of shape (sz, sz).
    1 in the lower triangle, 0 above the diagonal.
    """
    mask = torch.ones(sz, sz)
    mask = torch.tril(mask)
    return mask


def train_coagpt(
        json_file_path,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab_size,
        tgt_vocab_size,
        epochs=5,
        batch_size=4,
        lr=1e-4,
        device="cuda"
):
    # Build dataset/dataloader
    dataset = COADataset(json_file_path, src_tokenizer, tgt_tokenizer, max_len=64)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    # Create COA-GPT model
    model = COAGPT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_seq_len=128,
        dropout=0.1
    )
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (src_ids, tgt_ids) in enumerate(loader):
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            # We'll do next-token prediction. Typically you shift the target by 1 position
            # for the labels, but here we keep it simple for demonstration.
            labels = tgt_ids.clone()

            B, T_tgt = tgt_ids.shape
            # Build the causal mask
            causal_mask = create_causal_mask(T_tgt).to(device)
            # Expand to (B, T_tgt, T_tgt) if needed:
            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)

            logits = model(src_ids, tgt_ids, src_mask=None, tgt_mask=causal_mask, memory_mask=None)
            # (B, T_tgt, tgt_vocab_size)

            # Flatten for cross-entropy
            logits_flat = logits.view(-1, tgt_vocab_size)
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch}, step {step}, Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} done, Avg Loss = {avg_loss:.4f}")

    # Save the COA-GPT model
    torch.save(model.state_dict(), "coagpt_model.pth")
    print("COA-GPT model saved to coagpt_model.pth")

    return model


##############################################################################
# 7) MAIN
##############################################################################
if __name__ == "__main__":
    # Example: data.jsonl lines:
    #  {"source": "hello world", "target": "bonjour le monde"}
    #  {"source": "this is a test", "target": "c est un test"}

    # 1) Instantiate advanced tokenizers
    src_tokenizer = COATextTokenizer()
    tgt_tokenizer = COATextTokenizer()  # separate if you want different vocabs

    # 2) Hardcode or dynamically compute vocab sizes
    src_vocab_size = len(src_tokenizer.vocab)
    tgt_vocab_size = len(tgt_tokenizer.vocab)

    # 3) Train COA-GPT on your file (e.g. 'data.jsonl')
    data_path = "data.jsonl"  # path to your JSON lines file
    trained_coagpt = train_coagpt(
        json_file_path=data_path,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        epochs=2,
        batch_size=2,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 4) Reload for inference
    inference_model = COAGPT(src_vocab_size, tgt_vocab_size)
    inference_model.load_state_dict(torch.load("coagpt_model.pth"))
    inference_model.eval()
    print("COA-GPT reloaded for inference. Done.")
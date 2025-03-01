import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class COADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        """
        data: A list of dict, each with 'input_text' and 'target_text'
        tokenizer: Your COA-GPT tokenizer
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

        # Tokenize inputs
        input_enc = self.tokenizer.encode(
            input_text,
            add_special_tokens=True,
        )

        # Tokenize targets
        target_enc = self.tokenizer.encode(
            target_text,
            add_special_tokens=True,
        )

        # Convert to PyTorch tensors, ensure max_length
        input_ids = torch.tensor(
            input_enc.ids[: self.max_length], dtype=torch.long
        )
        target_ids = torch.tensor(
            target_enc.ids[: self.max_length], dtype=torch.long
        )

        # Optionally, create an attention mask if your model needs it
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }

class COAGPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers):
        super(COAGPTModel, self).__init__()
        # Your Transformer blocks (embedding, positional enc, multi-head attention, etc.)
        # We'll just pretend we have a single "transformer" object here:
        self.transformer = MyTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(self, input_ids, labels=None):
        # Forward pass
        logits = self.transformer(input_ids)  # shape: (batch_size, seq_len, vocab_size)
        return logits

from torch.optim import AdamW
import torch.nn.functional as F

def train_coa_gpt(
    model,
    dataloader,
    epochs=3,
    lr=1e-4,
    device="cuda",
):
    """
    model: COAGPTModel (or your custom Transformer)
    dataloader: PyTorch DataLoader with training data
    epochs: Number of training epochs
    lr: learning rate
    device: 'cuda' or 'cpu'
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # 1. Prepare data
            input_ids = batch["input_ids"].to(device)     # (B, seq_len)
            target_ids = batch["target_ids"].to(device)   # (B, seq_len)

            # 2. Forward pass
            logits = model(input_ids)  # (B, seq_len, vocab_size)

            # 3. Compute loss: cross-entropy across all time steps
            #    Flatten logits to (B * seq_len, vocab_size)
            #    Flatten targets to (B * seq_len)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0  # adjust if you have PAD tokens with ID=0
            )

            # 4. Backprop and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 5. Logging / print
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx+1}/{len(dataloader)}] | Loss: {avg_loss:.4f}")

        print(f"Epoch {epoch+1} finished. Average Loss: {total_loss/len(dataloader):.4f}")

    print("Training complete!")

# 1. Build / Load your tokenizer
# tokenizer = build_coa_gpt_tokenizer(...)

# 2. Prepare your training data
train_data = [
    {
      "input_text": "Mission: Surveil area near Bridge Bobcat for threat forces.",
      "target_text": "COA: Deploy UAV to x:75 y:26; coordinate recon with ground units..."
    },
    ...
]
train_dataset = COADataset(train_data, tokenizer=tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 3. Initialize the model
vocab_size = tokenizer.get_vocab_size()
model = COAGPTModel(vocab_size=vocab_size,d_model=512,num_heads=8,hidden_dim=2048,num_layers=6)

# 4. Train
train_coa_gpt(model=model,dataloader=train_loader,epochs=3,lr=1e-4,device="cuda")

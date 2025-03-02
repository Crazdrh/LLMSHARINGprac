"""
train.py

A script that uses torch.distributed for multi-GPU training.
We wrap the GPT model with DistributedDataParallel and read data via a distributed sampler.
"""

import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt_config import GPT70BConfig
from gpt_model import GPT70B
from data_loader import create_csv_dataloader


def train_one_epoch(
        model: DDP,
        dataloader,
        optimizer,
        scheduler,
        device: torch.device,
        grad_accum_steps: int = 1
):
    """
    Train the model for one epoch using gradient accumulation with DDP.

    model: a DDP-wrapped GPT70B
    dataloader: yields (input_ids, target_ids) for each batch
    optimizer: e.g. AdamW
    scheduler: e.g. LambdaLR
    device: 'cuda' or 'cpu'
    grad_accum_steps: accumulate gradients over this many mini-batches
    """
    model.train()
    total_loss = 0.0
    total_steps = 0

    optimizer.zero_grad()

    for step, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        logits, loss = model(input_ids, targets=target_ids)
        loss = loss / grad_accum_steps
        loss.backward()

        total_loss += loss.item() * grad_accum_steps
        total_steps += 1

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / total_steps


def main():
    # 1) Initialize distributed environment
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 2) Set device
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # 3) Build config
    config = GPT70BConfig(
        vocab_size=50257,
        max_seq_len=256,
        embed_dim=512,  # smaller for demonstration
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=0.01
    )
    if rank == 0:
        print(config.summary())

    # 4) Instantiate GPT model, move to device
    model = GPT70B(config).to(device)

    # 5) Wrap in DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 6) Create DataLoader with distributed sampler
    train_loader = create_csv_dataloader(
        csv_path="data.csv",
        batch_size=4,
        max_seq_len=256,
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )

    # 7) Optimizer & LR scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def lr_lambda(step: int):
        max_steps = 1000
        return max(0.0, 1.0 - (step / max_steps))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 8) Training
    epochs = 3
    grad_accum_steps = 2

    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=grad_accum_steps
        )

        # If you want an average across ranks, do dist.all_reduce on the avg_loss
        if rank == 0:
            print(f"[Epoch {epoch + 1}/{epochs}] (Rank {rank}) Loss = {avg_loss:.4f}")
            # Save checkpoint from rank 0 only
            torch.save(model.module.state_dict(), f"gpt70b_ddp_epoch{epoch + 1}.pth")

    dist.destroy_process_group()
    print(f"Rank {rank} finished training.")


if __name__ == "__main__":
    main()

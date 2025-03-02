"""
data_loader.py

Reads CSV lines, uses ds_token.encode(...) to tokenize each line,
and returns (input_ids, target_ids). Includes a DistributedSampler
for multi-GPU training.
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Tuple
from gpt_tokenizer import ds_token

class CSVDataset(Dataset):
    """
    CSVDataset
    ==========
    Loads text from column 0 in a CSV file, uses ds_token.encode(...)
    to produce token IDs.
    """

    def __init__(self, csv_path: str, max_seq_len: int = 256):
        self.csv_path = csv_path
        self.max_seq_len = max_seq_len
        self.samples = []
        self._load_csv()

    def _load_csv(self):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:
                    text = row[0]
                    self.samples.append(text)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        text = self.samples[idx]
        token_ids = ds_token.encode(text)

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]

        # For language modeling, input == target
        return token_ids, token_ids


def collate_fn(batch: List[Tuple[List[int], List[int]]]):
    """
    Collate function to pad sequences to the same length within a batch.
    """
    input_ids_list, target_ids_list = zip(*batch)
    max_len = max(len(x) for x in input_ids_list)

    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(input_ids_list, target_ids_list):
        pad_len = max_len - len(inp)
        inp_padded = inp + [0] * pad_len
        tgt_padded = tgt + [0] * pad_len
        padded_inputs.append(inp_padded)
        padded_targets.append(tgt_padded)

    input_ids_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    target_ids_tensor = torch.tensor(padded_targets, dtype=torch.long)
    return input_ids_tensor, target_ids_tensor


def create_csv_dataloader(
    csv_path: str,
    batch_size: int = 8,
    max_seq_len: int = 256,
    shuffle: bool = True,
    num_replicas: int = 1,
    rank: int = 0
) -> DataLoader:
    """
    Creates a DataLoader for CSV data. If you're using DDP, pass num_replicas > 1
    and the current rank to use a DistributedSampler.

    csv_path : Path to the CSV file
    batch_size : how many samples per batch
    max_seq_len : maximum tokens to keep
    shuffle : whether to shuffle (distributed sampler can handle this)
    num_replicas : total number of processes in DDP
    rank : this process's rank (0..num_replicas-1)

    returns: a torch DataLoader
    """
    dataset = CSVDataset(csv_path, max_seq_len=max_seq_len)

    sampler = None
    if num_replicas > 1:
        sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # If we use a distributed sampler, disable shuffle in the DataLoader constructor
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        collate_fn=collate_fn,
        sampler=sampler
    )
    return loader

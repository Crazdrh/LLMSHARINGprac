from tokenizers import Tokenizer, trainers, models, normalizers, pre_tokenizers, processors


def build_coa_gpt_tokenizer(
        domain_corpus_paths,  # e.g., multiple .txt or .json lines files
        vocab_size=32000,  # large enough for domain coverage
        special_tokens=["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]
):
    # 1. Initialize empty Byte-Pair Encoding (BPE) or WordPiece model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Normalizers & Pre-tokenizers
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),  # Normalize unicode
        normalizers.Lowercase(),  # Lowercase if domain usage allows
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        # Possibly add a specialized pre-tokenizer that retains punctuation for coords, e.g. "x:125"
    ])

    # 3. Trainer with domain-specific expansions
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # 4. Collect raw lines from domain corpus
    corpus_files = []
    for path in domain_corpus_paths:
        corpus_files.append(path)

    # 5. Train the tokenizer on all domain text
    tokenizer.train(files=corpus_files, trainer=trainer)

    # 6. Optional: Post-processing for standard templates
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]"))
        ]
    )

    return tokenizer

# Example usage in training
domain_corpus_paths = [
    "/data/military_corpus/logs.txt",
    "/data/military_corpus/doctrine_manuals.txt",
    # Additional sets with JSON lines, etc.
]
vocab_size = 32000
coa_tokenizer = build_coa_gpt_tokenizer(domain_corpus_paths, vocab_size)

# Save for deployment
coa_tokenizer.save("path\coa_gpt_bpe_tokenizer.json")
encoded_input = coa_tokenizer.encode("Mission: Secure OBJ_Lion at x:125, y:75.")
token_ids = encoded_input.ids  # e.g. [101, 2345, 12, ... 102]
#The token_ids are used as the input to the Transformer.
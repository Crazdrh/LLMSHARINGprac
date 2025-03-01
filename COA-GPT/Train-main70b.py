def train_coa_gpt(model, dataloader, epochs=1, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {total_loss/(batch_idx+1):.4f}")
        print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss/len(dataloader):.4f}")

def main():
    corpus_files = ["path/to/military_data_1.txt", "path/to/military_data_2.txt"]
    if not os.path.exists("coa_gpt_bpe_tokenizer.json"):
        tokenizer_obj = build_coa_gpt_tokenizer(corpus_files, vocab_size=50000)
        tokenizer_obj.save("coa_gpt_bpe_tokenizer.json")
    else:
        tokenizer_obj = Tokenizer.from_file("coa_gpt_bpe_tokenizer.json")
    train_data = [
        {"input_text": "Mission: Secure area near Bridge Alpha.", "target_text": "COA: Deploy infantry and armored units to secure Bridge Alpha."},
        {"input_text": "Mission: Reconnaissance near Sector 7.", "target_text": "COA: Dispatch UAVs and recon teams to gather intel in Sector 7."}
    ]
    dataset = COADataset(train_data, tokenizer_obj, max_length=1024)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    vocab_size = tokenizer_obj.get_vocab_size()
    model = COAGPT70B(vocab_size=vocab_size)
    train_coa_gpt(model, dataloader, epochs=1, lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu")
    torch.save(model.state_dict(), "coa_gpt_70b.pt")

if __name__ == "__main__":
    main()

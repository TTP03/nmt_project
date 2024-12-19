import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from preprocess import TranslationDataset, load_data
from model import get_model_and_tokenizer

def train(model, tokenizer_src, tokenizer_tgt, train_loader, optimizer, device):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    model_name = "t5-small"  # Sử dụng mô hình T5 nhỏ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_src, test_src, train_tgt, test_tgt = load_data("data/en_sents.txt", "data/vi_sents.txt")
    model, tokenizer = get_model_and_tokenizer(model_name)
    model.to(device)

    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    train(model, tokenizer, tokenizer, train_loader, optimizer, device)

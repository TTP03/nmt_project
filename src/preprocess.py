import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, tokenizer_src, tokenizer_tgt, max_len=128):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]

        src_encoded = self.tokenizer_src(
            src, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        tgt_encoded = self.tokenizer_tgt(
            tgt, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )

        return {
            "input_ids": src_encoded["input_ids"].squeeze(),
            "attention_mask": src_encoded["attention_mask"].squeeze(),
            "labels": tgt_encoded["input_ids"].squeeze(),
        }

def load_data(en_path, vi_path, test_size=0.2):
    with open(en_path, "r", encoding="utf-8") as f:
        en_sentences = f.read().splitlines()
    with open(vi_path, "r", encoding="utf-8") as f:
        vi_sentences = f.read().splitlines()

    return train_test_split(en_sentences, vi_sentences, test_size=test_size)

"""
model/train.py
Trains the CaptionLSTM on data/dataset.txt and saves model/model.pth.
Usage: python model/train.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import CaptionLSTM, build_vocab, save_vocab

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "C:/Users/dvkuk/project/data/dataset.txt"
VOCAB_PATH = "C:/Users/dvkuk/project/model/vocab.json"
MODEL_PATH = "C:/Users/dvkuk/project/model/model.pth"

EMBED_DIM   = 64
HIDDEN_SIZE = 256
EPOCHS      = 300
BATCH_SIZE  = 128
SEQ_LEN     = 200      # characters per training window
LR          = 3e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────────

class CharDataset(Dataset):
    def __init__(self, encoded: list[int], seq_len: int):
        self.data    = encoded
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx      : idx + self.seq_len],     dtype=torch.long)
        y = torch.tensor(self.data[idx + 1  : idx + self.seq_len + 1], dtype=torch.long)
        return x, y


# ── Main ───────────────────────────────────────────────────────────────────────

def train():
    # 1. Load raw text
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"📄 Dataset loaded: {len(text):,} chars")

    # 2. Build & save vocab
    char_to_idx, idx_to_char = build_vocab(text)
    save_vocab(char_to_idx, idx_to_char, VOCAB_PATH)
    vocab_size = len(char_to_idx)

    # 3. Encode entire corpus
    encoded = [char_to_idx[ch] for ch in text]

    # 4. Build DataLoader
    dataset    = CharDataset(encoded, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"🗂  Batches per epoch: {len(dataloader)}")

    # 5. Model, loss, optimiser
    model     = CaptionLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")

    # 6. Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            hidden = model.init_hidden(x_batch.size(0), DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x_batch, hidden)          # (B, T, V)

            # Reshape for CrossEntropyLoss: (B*T, V) vs (B*T,)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                y_batch.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={avg_loss:.4f}")

    # 7. Save weights
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
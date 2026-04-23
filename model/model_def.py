"""
model/model.py
Character-level LSTM model for Instagram caption generation.
Vocab helpers: build_vocab(), save_vocab(), load_vocab()
"""

import json
import os
import torch
import torch.nn as nn


# ── Vocabulary helpers ─────────────────────────────────────────────────────────

def build_vocab(text: str) -> tuple[dict, dict]:
    """Build char↔idx mappings from raw text."""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def save_vocab(char_to_idx: dict, idx_to_char: dict,
               path: str = "model/vocab.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "char_to_idx": char_to_idx,
        # JSON keys must be strings; store int keys as strings
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Vocab saved → {path}  (size={len(char_to_idx)})")


def load_vocab(path: str = "model/vocab.json") -> tuple[dict, dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    char_to_idx = payload["char_to_idx"]
    idx_to_char = {int(k): v for k, v in payload["idx_to_char"].items()}
    return char_to_idx, idx_to_char


# ── Model ──────────────────────────────────────────────────────────────────────

class CaptionLSTM(nn.Module):
    """
    Character-level LSTM caption generator.

    Architecture:
        Embedding(vocab_size, embed_dim=32)
        → LSTM(embed_dim, hidden_size=128, num_layers=1, batch_first=True)
        → Linear(128, vocab_size)

    Parameter count (vocab≈100): 32*100 + 4*(128*(32+128)+128) + 128*100
        ≈ 3200 + 135168 + 12800 ≈ 151 000  >> 1 000 ✓
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size

        self.embed   = nn.Embedding(vocab_size, embed_dim)
        self.lstm    = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                          # (B, T) int
        hidden: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        emb = self.embed(x)                        # (B, T, E)
        out, hidden = self.lstm(emb, hidden)       # (B, T, H)
        logits = self.fc(out)                      # (B, T, V)
        return logits, hidden

    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h, c
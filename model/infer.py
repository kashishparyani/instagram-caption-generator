"""
model/infer.py
Loads trained model and vocab; exposes generate_caption().
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .model_def import CaptionLSTM, load_vocab

# ── Config ─────────────────────────────────────────────────────────────────────
VOCAB_PATH  = "model/vocab.json"
MODEL_PATH  = "model/model.pth"
EMBED_DIM   = 64
HIDDEN_SIZE = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokens
END_TOKEN = "<END>"

# ── Lazy-load globals (populated once by load_model()) ────────────────────────
_model        = None
_char_to_idx  = None
_idx_to_char  = None


def load_model():
    """Load vocab and model weights into module-level globals."""
    global _model, _char_to_idx, _idx_to_char

    _char_to_idx, _idx_to_char = load_vocab(VOCAB_PATH)
    vocab_size = len(_char_to_idx)

    _model = CaptionLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(DEVICE)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _model.eval()
    print("✅ Model loaded for inference.")


# ── Core generation ────────────────────────────────────────────────────────────

def generate_caption(
    segment: str,
    description: str,
    temperature: float = 0.8,
    max_new_chars: int = 300,
) -> str:
    """
    Generate an Instagram caption.

    Args:
        segment:     e.g. "Fitness"
        description: e.g. "Promote gym membership"
        temperature: sampling temperature (lower = more predictable)
        max_new_chars: hard stop on generated length

    Returns:
        Clean caption string (system tokens stripped).
    """
    if _model is None:
        load_model()

    prompt = f"<SEG> {segment} <DESC> {description} ->"

    # Encode prompt — skip unknown chars
    encoded = [_char_to_idx[ch] for ch in prompt if ch in _char_to_idx]
    if not encoded:
        return "Could not encode prompt."

    input_tensor = torch.tensor([encoded], dtype=torch.long, device=DEVICE)  # (1, T)

    generated_chars: list[str] = []

    with torch.no_grad():
        # Prime the hidden state with the prompt
        hidden = _model.init_hidden(1, DEVICE)
        logits, hidden = _model(input_tensor, hidden)

        # Sample next char from last position
        next_logits = logits[0, -1, :]  # (V,)

        for _ in range(max_new_chars):
            # Temperature sampling
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = _idx_to_char[next_idx]

            generated_chars.append(next_char)
            raw_so_far = "".join(generated_chars)

            # Stop if <END> token appears
            if END_TOKEN in raw_so_far:
                break

            # Feed next char back in
            x = torch.tensor([[next_idx]], dtype=torch.long, device=DEVICE)
            next_logits, hidden = _model(x, hidden)
            next_logits = next_logits[0, -1, :]

    raw_output = "".join(generated_chars)

    # ── Clean output ───────────────────────────────────────────────────────────
    # Remove everything from <END> onward
    if END_TOKEN in raw_output:
        raw_output = raw_output[: raw_output.index(END_TOKEN)]

    # Strip leading/trailing whitespace and arrows
    caption = raw_output.strip().lstrip("->").strip()

    return caption if caption else "Try again with a different description."


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    result = generate_caption("Fitness", "Morning workout routine")
    print("Generated caption:", result)
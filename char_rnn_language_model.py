
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. LOAD DATA (Toy + real file optional)
# ------------------------------------------------------------

# Larger toy dataset (works for training)
toy_text = """
hello hello help hello helicopter hell help
hello world this is a tiny character level rnn test
machine learning is fun and deep learning is powerful
char rnn models learn to predict the next character
"""

# Optional: load a real text file
# with open("textfile.txt", "r", encoding="utf-8") as f:
#     toy_text = f.read().lower()

text = toy_text.lower()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(indices):
    return "".join([itos[int(i)] for i in indices])

data = encode(text)

# Train/validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ------------------------------------------------------------
# 2. BATCH FUNCTION
# ------------------------------------------------------------

def get_batch(split, seq_len=5, batch_size=64):
    data = train_data if split == "train" else val_data

    # Make sure index range is valid
    max_index = len(data) - seq_len - 1
    if max_index <= 0:
        raise ValueError("Dataset too small for this seq_len. Reduce seq_len.")

    ix = torch.randint(0, max_index, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y

# ------------------------------------------------------------
# 3. MODEL
# ------------------------------------------------------------

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, rnn_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# ------------------------------------------------------------
# 4. TRAINING LOOP
# ------------------------------------------------------------

embed_dim = 64
hidden_size = 128
seq_len = 5
batch_size = 64
epochs = 10
learning_rate = 0.003

model = CharRNN(vocab_size, embed_dim, hidden_size, rnn_type="LSTM")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    xb, yb = get_batch("train", seq_len, batch_size)
    logits, _ = model(xb)
    
    loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        xb, yb = get_batch("val", seq_len, batch_size)
        logits, _ = model(xb)
        val_loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# ------------------------------------------------------------
# 5. GENERATION FUNCTION
# ------------------------------------------------------------

def generate_text(model, start="h", length=300, temperature=1.0):
    model.eval()
    input_idx = torch.tensor([[stoi[start]]], dtype=torch.long)
    hidden = None
    result = [start]

    for _ in range(length):
        logits, hidden = model(input_idx, hidden)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        
        result.append(itos[idx])
        input_idx = torch.tensor([[idx]])

    return "".join(result)

# Generate samples
print("\n=== SAMPLE (T = 0.7) ===")
print(generate_text(model, temperature=0.7))

print("\n=== SAMPLE (T = 1.0) ===")
print(generate_text(model, temperature=1.0))

print("\n=== SAMPLE (T = 1.2) ===")
print(generate_text(model, temperature=1.2))

# ------------------------------------------------------------
# 6. LOSS PLOT
# ------------------------------------------------------------

plt.figure(figsize=(7,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()

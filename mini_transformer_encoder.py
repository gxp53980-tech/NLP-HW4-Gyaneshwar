import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------
# 1. Small toy dataset (10 sentences)
# ------------------------------------------------------
sentences = [
    "hello how are you",
    "i love machine learning",
    "this is a mini transformer",
    "pytorch makes deep learning fun",
    "attention is all you need",
    "we are learning transformers",
    "natural language processing is powerful",
    "deep models understand context",
    "words create meaning",
    "transformers use self attention"
]

# ------------------------------------------------------
# 2. Build vocabulary
# ------------------------------------------------------
words = set(" ".join(sentences).split())
word2idx = {w: i+1 for i, w in enumerate(words)}
word2idx["<PAD>"] = 0
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(word2idx)

# ------------------------------------------------------
# 3. Tokenize and pad sentences
# ------------------------------------------------------
def tokenize(sentence):
    return [word2idx[w] for w in sentence.split()]

tokenized = [tokenize(s) for s in sentences]
max_len = max(len(s) for s in tokenized)

# pad
for s in tokenized:
    while len(s) < max_len:
        s.append(0)

tokens = torch.tensor(tokenized)
batch_size, seq_len = tokens.shape
print("Input tokens:\n", tokens, "\n")

# ------------------------------------------------------
# 4. Sinusoidal positional encoding
# ------------------------------------------------------
def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / d_model)
    angles = pos * angle_rates

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe

# ------------------------------------------------------
# 5. Multi-Head Self Attention
# ------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq, d_model = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # split into heads
        Q = Q.view(batch, seq, self.num_heads, self.dk).transpose(1,2)
        K = K.view(batch, seq, self.num_heads, self.dk).transpose(1,2)
        V = V.view(batch, seq, self.num_heads, self.dk).transpose(1,2)

        # scaled dot product attention
        scores = (Q @ K.transpose(-2,-1)) / (self.dk**0.5)
        attention = torch.softmax(scores, dim=-1)
        out = attention @ V

        # merge heads
        out = out.transpose(1,2).contiguous().view(batch, seq, d_model)
        out = self.out(out)

        return out, attention

# ------------------------------------------------------
# 6. Feed Forward Network
# ------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model=64, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# ------------------------------------------------------
# 7. Mini Transformer Encoder Block
# ------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("positional", positional_encoding(50, d_model))

        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        emb = self.embedding(x) + self.positional[:x.size(1)]
        att_out, att_weights = self.attention(emb)
        out1 = self.norm1(emb + att_out)
        ff_out = self.ff(out1)
        out2 = self.norm2(out1 + ff_out)
        return out2, att_weights

# ------------------------------------------------------
# 8. Run the Mini Transformer Encoder
# ------------------------------------------------------
d_model = 64
encoder = TransformerEncoder(vocab_size, d_model=d_model, num_heads=4)

contextual_embeddings, attention_weights = encoder(tokens)

print("\n=== FINAL CONTEXTUAL EMBEDDINGS ===")
print(contextual_embeddings.shape)    # (batch, seq, d_model)
print(contextual_embeddings)

print("\n=== ATTENTION WEIGHTS (HEAD 1) ===")
print(attention_weights[0][0])  # For 1st sentence, head 1

import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: (batch, seq_len, d_k)
    returns: attention_output, attention_weights
    """

    # 1️⃣ Raw attention scores = Q K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    print("\n=== RAW SCORES (Unscaled QKᵀ) ===")
    print(scores)

    # 2️⃣ Scaling for stability
    d_k = Q.size(-1)
    scaled_scores = scores / math.sqrt(d_k)

    print("\n=== SCALED SCORES (QKᵀ / sqrt(d_k)) ===")
    print(scaled_scores)

    # 3️⃣ Softmax over last dimension (attention distribution)
    attention_weights = F.softmax(scaled_scores, dim=-1)

    print("\n=== ATTENTION WEIGHT MATRIX (Softmax) ===")
    print(attention_weights)

    # 4️⃣ Output = softmax * V
    output = torch.matmul(attention_weights, V)

    print("\n=== OUTPUT VECTORS (Attention × V) ===")
    print(output)

    return output, attention_weights


# ----------------------------
# TEST WITH RANDOM INPUTS
# ----------------------------

torch.manual_seed(42)

batch = 1
seq_len = 4
d_k = 8

# Random Q, K, V
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Run attention
output, weights = scaled_dot_product_attention(Q, K, V)


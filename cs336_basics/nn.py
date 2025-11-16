import torch
import torch.nn as nn
from jaxtyping import Float, Int, Bool
from torch import Tensor

class Linear(nn.Module):
    """Linear transformation without bias."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self._init_weights()
    
    def _init_weights(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
    

class Embedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        
        return (self.weight * x_norm).to(in_dtype)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], 
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss."""
    batch_size = inputs.shape[0]
    
    # Numerical stability: subtract max
    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    inputs_shifted = inputs - inputs_max
    
    # Log-sum-exp trick
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_shifted), dim=-1))
    
    # Get logits for correct classes
    correct_logits = inputs_shifted[torch.arange(batch_size), targets]
    
    # Cross entropy = -log(softmax(correct_class))
    losses = log_sum_exp - correct_logits
    
    return torch.mean(losses)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        
        # Precompute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angles = positions[:, None] * freqs[None, :]  # (max_seq_len, d_k//2)
        
        # Register as buffers (non-trainable)
        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Apply RoPE to input tensor.
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        # Get cos/sin for the positions
        cos = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k//2)
        
        # Split into pairs and rotate
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * cos + x1 * sin
        
        # Interleave back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated.flatten(-2)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """Scaled dot-product attention."""
    d_k = Q.size(-1)
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    # Apply mask if provided (True = attend, False = mask out)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Softmax
    attn_weights = softmax(scores, dim=-1)
    
    # Apply to values
    return attn_weights @ V


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: W2(SiLU(W1(x)) * W3(x))
        return self.w2(self._silu(self.w1(x)) * self.w3(x))
    
    @staticmethod
    def _silu(x: Tensor) -> Tensor:
        """SiLU activation: x * sigmoid(x)"""
        return x * torch.sigmoid(x)
    
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""
    
    def __init__(self, d_model: int, num_heads: int, rope=None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        
        # Projections
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x: Tensor, token_positions: Tensor = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Project and reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE if provided
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            Q = self.rope(Q, token_positions.unsqueeze(1))  # Broadcast over heads
            K = self.rope(K, token_positions.unsqueeze(1))
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        causal_mask = ~causal_mask  # Flip: True = attend, False = mask
        
        # Attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        
        # Reshape back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        return self.o_proj(attn_output)

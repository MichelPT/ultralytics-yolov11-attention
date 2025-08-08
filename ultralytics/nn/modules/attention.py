# In a new file, e.g., ultralytics/nn/modules/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """A simple Cross-Attention module for feature fusion."""
    def __init__(self, query_dim, key_value_dim, head_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Projections for Q from the query stream
        self.to_q = nn.Linear(query_dim, head_dim * num_heads, bias=False)
        # Projections for K, V from the key-value stream
        self.to_kv = nn.Linear(key_value_dim, head_dim * num_heads * 2, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(head_dim * num_heads, query_dim)

    def forward(self, query, key_value):
        # query shape: (batch, num_tokens_q, query_dim)
        # key_value shape: (batch, num_tokens_kv, key_value_dim)
        
        # Get query projection
        q = self.to_q(query) # -> (batch, num_tokens_q, head_dim * num_heads)

        # Get key and value projections
        kv = self.to_kv(key_value).chunk(2, dim=-1)
        k, v = kv[0], kv[1] # -> (batch, num_tokens_kv, head_dim * num_heads) each

        # Reshape for multi-head attention
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(context.shape[0], -1, self.num_heads * self.head_dim)
        
        # Project back to original dimension
        out = self.to_out(context)
        
        return out

class CrossAttentionBlock(nn.Module):
    """
    A robust, Conv2d-based module that uses Cross-Attention to fuse two feature maps
    and then concatenates them. This is a drop-in replacement for a standard Concat layer.
    """
    def __init__(self, c1, c2, n_heads=8):
        """
        Initializes the CrossAttentionConcat module.
        Args:
            c1 (int): Channels of the first input (x1), which will be the query.
            c2 (int): Channels of the second input (x2), which will be the key/value.
            n_heads (int): Number of attention heads.
        """
        super().__init__()
        # Ensure n_heads is a valid divisor of c1
        if c1 % n_heads != 0:
            valid_heads = [h for h in range(n_heads, 0, -1) if c1 % h == 0]
            n_heads = valid_heads[0] if valid_heads else 1

        # Use Conv2d for projections to keep tensors as 2D feature maps
        self.q_conv = nn.Conv2d(c1, c1, 1, bias=False)
        self.k_conv = nn.Conv2d(c2, c1, 1, bias=False)
        self.v_conv = nn.Conv2d(c2, c1, 1, bias=False)
        
        self.mha = nn.MultiheadAttention(embed_dim=c1, num_heads=n_heads, batch_first=True)
        self.out_conv = nn.Conv2d(c1, c1, 1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (list[torch.Tensor]): A list of two tensors, [x1, x2].
        """
        x1, x2 = x  # x1 is the main path (query), x2 is from the backbone (key/value)
        B, C1, H, W = x1.shape
        
        # Get projections
        q = self.q_conv(x1)
        k = self.k_conv(x2)
        v = self.v_conv(x2)
        
        # Reshape for MHA
        q = q.flatten(2).transpose(1, 2)  # (B, H*W, C1)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        
        # Apply attention
        attn_output, _ = self.mha(q, k, v)
        
        # Reshape back to 2D feature map and process
        fused_x1 = self.out_conv(attn_output.transpose(1, 2).reshape(B, C1, H, W))
        
        # Add residual connection to the query path
        refined_x1 = x1 + fused_x1
        
        # Concatenate the refined query path with the original key/value path
        return torch.cat([refined_x1, x2], dim=1)
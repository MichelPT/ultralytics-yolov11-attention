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
    A wrapper block to handle the 2D feature map -> 1D sequence conversion
    for using CrossAttention in a CNN.
    """
    def __init__(self, c1, c2):  # c1 = query_channels, c2 = key_value_channels
        super().__init__()
        # This part is fine
        self.ca = CrossAttention(query_dim=c1, key_value_dim=c2, head_dim=c1 // 8, num_heads=8)
        self.conv = nn.Conv2d(c1, c1, 1, bias=False)  # To fuse the output
        self.bn = nn.BatchNorm2d(c1)

    # --- THIS METHOD NEEDS TO BE CHANGED ---
    def forward(self, x):  # Change signature to accept one argument 'x'
        x_query, x_kv = x  # Unpack the list of two tensors
        # --- END OF CHANGES ---

        # The rest of your code remains the same
        B, C1, H, W = x_query.shape
        
        # Flatten feature maps to sequences
        query_seq = x_query.flatten(2).transpose(1, 2)
        kv_seq = x_kv.flatten(2).transpose(1, 2)
        
        # Apply cross-attention
        attended_seq = self.ca(query_seq, kv_seq)
        
        # Reshape back to 2D feature map
        attended_map = attended_seq.transpose(1, 2).view(B, C1, H, W)
        
        # Combine with original query via a residual-like connection
        output = self.bn(self.conv(attended_map)) + x_query
        return output
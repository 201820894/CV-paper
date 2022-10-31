import torch
import torch.nn as nn

# Data Preprocessing
# Get (N)x(P^2*C) size input image


class LinearProjection(nn.Module):
    def __init__(self, patch_vec_size, num_patches, embedded_vec_dim, drop_rate):
        # patch_vec size: int, P
        # embedded_vec_dim: int, D
        # drop_rate: float, dropout rate
        super(LinearProjection, self).__init__()
        self.linear_proj = nn.Linear(patch_vec_size, embedded_vec_dim)
        self.class_token = nn.Parameter(torch.randn(1, embedded_vec_dim))
        self.pose_embedding = nn.Parameter(
            torch.randn(1, num_patches+1, embedded_vec_dim))
        self.dropout = nn.Dropout(drop_rate)

    # x: (N)x(P^2*C)
    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([self.class_token.repeat(
            batch_size, 1, 1), self.linear_proj(x)])
        x += self.pose_embedding
        x = self.dropout(x)
        return x


class MultiheadedSelfAttention(nn.Module):
    def __init__(self, embedded_vec_dim, num_heads, drop_rate):
        super(MultiheadedSelfAttention, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.embedded_vec_dim = embedded_vec_dim

        self.head_dim = int(embedded_vec_dim / num_heads)  # Add assert
        assert embedded_vec_dim // num_heads == 0, "Embedded vector dimension should be divided by #heads"

        # n_head x head_dim = embedded_vec_dim. Can calculate once
        self.query = nn.Linear(embedded_vec_dim, embedded_vec_dim)
        self.key = nn.Linear(embedded_vec_dim, embedded_vec_dim)
        self.value = nn.Linear(embedded_vec_dim, embedded_vec_dim)

        self.scale = torch.sqrt(embedded_vec_dim*torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Original: (B, N, embedded_vec_dim)
        # After transformation: (B, num_heads, N, head_dims)
        q = q.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # (B, num_heads, N, N)
        # Calculation with multi-head
        attn_prob = torch.softmax(q.dot(k.transpose(2, 3))/self.scale, dim=-1)

        # (B, num_heads, N, head_dims)
        x = self.dropout(attn_prob).dot(v)

        # (B, N, embedded_vec_dim)
        x = x.permute(0, 2, 1, 3).view(batch_size, -1, self.embedded_vec_dim)

        return x, attn_prob

# Single block


class EncoderBlock(nn.Module):
    def __init__(self, embedded_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super(EncoderBlock, self).__init__
        self.ln1 = nn.LayerNorm(embedded_vec_dim)
        self.ln2 = nn.LayerNorm(embedded_vec_dim)
        self.msa = MultiheadedSelfAttention(
            embedded_vec_dim=embedded_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(nn.Linear(embedded_vec_dim, mlp_hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, embedded_vec_dim),
                                 nn.Dropout(drop_rate))

    # With residual blocks
    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, att


class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, embedded_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        self.patchembedding = LinearProjection(
            patch_vec_size=patch_vec_size, num_patches=num_patches, embedded_vec_dim=embedded_vec_dim, drop_rate=drop_rate)
        self.transformer = nn.ModuleList([EncoderBlock(embedded_vec_dim=embedded_vec_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                          for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(embedded_vec_dim),
                                      nn.Linear(embedded_vec_dim, num_classes))

    def forward(self, x):
        att_list = []
        x = self.patchembedding(x)
        for layer in self.transformer:
            x, attn_prob = layer(x)
            att_list.append(attn_prob)
        x = self.mlp_head(x[0, :])

        return x, att_list

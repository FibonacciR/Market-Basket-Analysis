

import torch
import torch.nn as nn


class NCF(nn.Module):
    """Simple Neural Collaborative Filtering (NCF) model used for inference.

    Inputs:
    - num_users: number of users (vocabulary size)
    - num_products: number of products (vocabulary size)

    Forward accepts two 1-D LongTensors of equal length: user_ids and product_ids
    and returns a 1-D FloatTensor of scores (len == batch_size).
    """

    def __init__(self, num_users: int = 131209, num_products: int = 47913, extra_input_dim: int = 18):
        super().__init__()
        # Embedding dims (match checkpoint that was used to train)
        user_embedding_dim = 50
        product_embedding_dim = 50

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        self.product_embedding = nn.Embedding(num_products, product_embedding_dim)

        # Input dim is concatenation of user and product embeddings plus any extra features
        # The training checkpoint expects 118 (=50+50+18), so default extra_input_dim=18
        self.extra_input_dim = int(extra_input_dim)
        input_dim = user_embedding_dim + product_embedding_dim + self.extra_input_dim

        # Hidden layers and batch norms per checkpoint
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
        ])
        self.output_layer = nn.Linear(32, 1)

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(32),
        ])

    def forward(self, user_ids: torch.LongTensor, product_ids: torch.LongTensor, extra_features: torch.Tensor = None) -> torch.FloatTensor:
        # Expect user_ids and product_ids to have same shape: (batch,)
        # extra_features (optional) should be float tensor with shape (batch, extra_input_dim)
        user_emb = self.user_embedding(user_ids)
        product_emb = self.product_embedding(product_ids)

        if self.extra_input_dim > 0:
            if extra_features is None:
                # create zeros on the same device/dtype as embeddings
                batch_size = user_emb.shape[0]
                extra_features = torch.zeros(batch_size, self.extra_input_dim, device=user_emb.device, dtype=user_emb.dtype)
            else:
                # ensure shape matches
                if extra_features.dim() == 1:
                    extra_features = extra_features.unsqueeze(1)
                if extra_features.shape[-1] != self.extra_input_dim:
                    raise ValueError(f"extra_features must have last dim {self.extra_input_dim}, got {extra_features.shape}")
            x = torch.cat([user_emb, product_emb, extra_features], dim=-1)
        else:
            x = torch.cat([user_emb, product_emb], dim=-1)

        x = self.layers[0](x)
        x = self.batch_norms[0](x)
        x = torch.relu(x)

        x = self.layers[1](x)
        x = self.batch_norms[1](x)
        x = torch.relu(x)

        x = self.layers[2](x)
        x = self.batch_norms[2](x)
        x = torch.relu(x)

        out = self.output_layer(x)
        return out.squeeze(-1)
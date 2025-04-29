import torch
from torch import Tensor, nn
from typing import Optional
from dataclasses import dataclass

#_____ Configuration dataclass for model hyperparameters_______#
@dataclass
class Config:
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    max_seq_len: int
    dropout_prob: float = 0.1

#_____ Single Transformer block as used in GPT-2: Pre-LayerNorm, Multi-head Self-Attention, and Feed-Forward._______#
class GPT2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        #_____ Multi-head Self-Attention with Pre-LN_______#
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_output)

        #_____ Feed-Forward with Pre-LN_______#
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x

#_____ Simplified GPT-2 language model with token & positional embeddings, stacked blocks, final norm, and weight tying._______#
class GPT2Simple(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        #_____ Embeddings_______#
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        #_____ Transformer blocks_______#
        self.blocks = nn.ModuleList([
            GPT2Block(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout_prob
            )
            for _ in range(config.num_layers)
        ])

        #_____ Final layer norm & output projection with weight tying_______#
        self.ln_final = nn.LayerNorm(config.d_model)
        self.output_linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_linear.weight = self.token_embedding.weight

        #_____ Store config for reference_______#
        self.config = config

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device) -> Tensor:
        #_____ Generate a causal mask for self-attention: future tokens masked with -inf_______#
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask.to(device)

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        #_____ Forward pass through the GPT-2 model_______#
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        #_____ Embedding + positional encoding_______#
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        #_____ Causal mask if none provided_______#
        if attn_mask is None:
            attn_mask = self._generate_causal_mask(seq_len, device)

        #_____ Transformer blocks_______#
        for block in self.blocks:
            x = block(x, attn_mask)

        #_____ Final norm and output projection_______#
        x = self.ln_final(x)
        logits = self.output_linear(x)
        return logits

#_____ Example usage with Config and random input generation_______#

config = Config(
        vocab_size=10000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout_prob=0.1
    )

model = GPT2Simple(config)
batch_size, seq_len = 2, 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
random_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device
    )

print(random_input.shape)
logits = model(random_input)
print(f"Logits shape: {logits.shape}")  

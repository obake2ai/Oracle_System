import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert (n_heads * self.head_dim == d_model)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape

        # Project the input embeddings into Q, K, and V
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(self.dropout(attention_weights), V)

        # Concatenate heads and put them back to the original shape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        # Apply the final linear transformation
        out = self.fc_out(attention_output)

        return out



class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model) -> None:
        super().__init__()
        # Create a matrix of shape (context_length, d_model) to store the positional encodings
        pe = torch.zeros(context_length, d_model)

        # Create a vector with positions [0, 1, 2, ..., context_length-1] of shape (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)

        # Create a vector with the divisor terms based on the dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)

        # Register pe as a buffer, so it is not considered a parameter but is part of the module's state
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the positional encodings to the input embeddings
        return x + self.pe[:,:x.size(1), :]


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, logits):
        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        return logits


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_length, tokenizer):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
        self.wpe = PositionalEncoding(context_length, d_model) # word position encodings
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in  range(n_layers)])
        self.linear1 = nn.Linear(d_model, vocab_size)
        self.context_length = context_length
        self.tokenizer = tokenizer

        self.wte.weight = self.linear1.weight

    def forward(self, inputs, targets = None):
        logits = self.wte(inputs) # dim -> batch_size, sequence_length, d_model
        logits = self.wpe(logits)
        for block in self.blocks:
            logits = block(logits)
        logits = self.linear1(logits)
        loss = None
        if targets != None:
            batch_size, sequence_length, d_model = logits.shape
            # to calculate loss for all token embeddings in a batch
            # kind of a requirement for cross_entropy
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # this will store the model outputs along with the initial input sequence
        # make a copy so that it doesn't interfare with model
        output = inputs.clone()
        for _ in range(max_new_tokens):
            current_seq_length = inputs.size(1)
            # Truncate inputs if it exceeds context_length
            if current_seq_length > self.context_length:
                inputs = inputs[:, -self.context_length:]
            # we only pass targets on training to calculate loss
            logits, _ = self(inputs)
            # for all the batches, get the embeds for last predicted sequence
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            # get the probable token based on the input probs
            idx_next = torch.multinomial(probs, num_samples=1)

            inputs = torch.cat([inputs, idx_next], dim=1)
            output = torch.cat([output, idx_next], dim=1)
        return [self.tokenizer.decode(out.tolist()) for out in output]

import torch.nn as nn
from .embedding import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.embedding = BERTEmbedding(dropout=dropout)

        # Transformer para processar a sequência de embeddings ao longo dos anos
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,           # Tamanho do embedding
            nhead=8,               # Número de cabeças de atenção
            dim_feedforward=1024,  # Tamanho do feedforward
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Camada final de regressão
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        # Passa pelo embedding
        x = self.embedding(x)

        # Passa pelo Transformer
        x = x.permute(1, 0, 2)  # [batch_size, years, embed_size] -> [years, batch_size, embed_size]
        x = self.transformer(x)

        # Pega a média ao longo da dimensão dos anos
        x = x.mean(dim=0)

        # Passa pela camada de regressão
        output = self.output_layer(x)
        return output


# d_model = 256  # Tamanho do embedding
# seq_length = 75  # Número de passos no tempo
# batch_size = 16  # Tamanho do batch

# # Supondo que 'time' seja uma matriz com os índices das posições temporais para o batch
# time = torch.randint(0, seq_length, (batch_size, seq_length))  # Exemplo de índices de tempo aleatórios

# # Garantir que time seja do tipo long (int64)
# print(time.dtype)  # Verificar tipo, deve ser torch.int64 após .long()

# positional_encoding = PositionalEncoding(d_model, max_len=366)
# output = positional_encoding(time)

# print(output.shape)  # [batch_size, seq_length, d_model]
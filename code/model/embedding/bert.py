import torch.nn as nn
from .position import PositionalEncoding


class BERTEmbedding(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        channel_size = (32, 64)

        # Primeira convolução
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=10,              # Bandas espectrais
                out_channels=channel_size[0],
                kernel_size=(3, 3, 3),       # Temporal, Altura, Largura
                stride=(1, 1, 1),            # Mantém dimensões
                padding=(1, 1, 1)            # Preserva tamanhos
            ),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[0]),
        )

        # Primeiro pooling
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Segunda convolução
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel_size[0],
                out_channels=channel_size[1],
                kernel_size=(3, 2, 2),       # Temporal, Altura, Largura
                stride=(1, 1, 1),            # Mantém dimensões temporais
                padding=(1, 1, 1)            # Preserva tamanhos
            ),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[1]),
        )

        # Segundo pooling
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Camada linear final
        self.linear = nn.Linear(
            in_features=channel_size[1],  # Final após convoluções
            out_features=256             # Tamanho do embedding final
        )

        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding
        self.position = PositionalEncoding(
            d_model=256,                   # Tamanho do embedding
            max_len=75                      # Seq_length = 75
        )

    def forward(self, input_sequence):
        batch_size, years, seq_length, bands, w, h = input_sequence.shape

        # Agrupa anos e observações temporais em uma única dimensão para convolução
        x = input_sequence.view(batch_size, years * seq_length, bands, w, h).permute(0, 2, 1, 3, 4)

        # Passa pela primeira convolução e pooling
        x = self.conv1(x)
        x = self.pool1(x)

        # Passa pela segunda convolução e pooling
        x = self.conv2(x)
        x = self.pool2(x)

        # Verifique a forma de x após a convolução e pooling
        print(f"Shape após convolução e pooling: {x.shape}")

        # Pega as dimensões: [batch_size, channels, seq_len, height, width]
        _, c, t, w, h = x.shape

        # Ajusta a forma para [batch_size, seq_len, embed_size]
        x = x.view(batch_size, t, -1)  # Agrupa as características espaciais em um vetor por cada ponto no tempo

        # Passa pela camada linear
        x = self.linear(x)

        # Adiciona o Positional Encoding
        position_embed = self.position(x)
        x = x + position_embed

        return self.dropout(x)
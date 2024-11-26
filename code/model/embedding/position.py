"""
Reference: https://github.com/codertimo/BERT-pytorch
Author: Junseong Kim
"""
import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.requires_grad_(False)

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()  # Exponenciação de termos para codificação posicional

        # Atribuindo seno e cosseno alternadamente para diferentes dimensões
        pe[1:, 0::2] = torch.sin(position * div_term)  # Seno para dimensões ímpares
        pe[1:, 1::2] = torch.cos(position * div_term)  # Cosseno para dimensões pares

        self.register_buffer('pe', pe)

    def forward(self, time):
        # Garantir que o 'time' seja no formato [batch_size, seq_length]
        # time: índices da sequência de entrada

        # Certifique-se de que o tipo de 'time' seja long (int64)
        time = time.long()

        # Achar as posições correspondentes para cada elemento do batch
        return self.pe[time]

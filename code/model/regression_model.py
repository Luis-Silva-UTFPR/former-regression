import torch
import torch.nn as nn
from .bert import BERT


class BERTRegression(nn.Module):
    """
    Downstream task: Satellite Time Series Regression
    """

    def __init__(self, bert: BERT):
        """
        :param bert: the BERT-Former model
        """
        super().__init__()
        self.bert = bert
        self.regression = RegressionHead(self.bert.hidden)

    def forward(self, x, doy, mask):
        x = self.bert(x, doy, mask)  # [batch_size, seq_length, embed_size]
        return self.regression(x, mask)


class RegressionHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, 1)


    def forward(self, x, mask): # 
        mask = (1 - mask.unsqueeze(-1)) * 1e6
        x = x - mask  # mask invalid timesteps

        ema = x[:, 0]
        for t in range(1, x.size(1)):
            ema = 0.1 * x[:, t] + (1 - 0.1) * ema

        x_mean = torch.mean(x, dim=1)
        x = self.weight_ema * ema + self.weight_mean * x_mean
        x = self.linear(x)
        
        return x.squeeze(-1)

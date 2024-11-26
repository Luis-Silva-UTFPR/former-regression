import torch.nn as nn
from .bert import BERT

class BERTPrediction(nn.Module):
    def __init__(self, bert: BERT, num_features=10):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden, num_features)

    def forward(self, x, doy, mask):
        print(f'Before view: {x.shape}')
        # Verificando as dimensões de x antes de tentar redimensionar
        if x.dim() == 5:  # Verifica se x tem 5 dimensões
            x = x.view(x.size(0), x.size(1), -1)  # Redimensiona para [128, 75, 250]
        print(f'After view: {x.shape}')
        
        # Máscara onde 0 é para ser mascarado (True), e 1 é para não ser mascarado (False)
        mask = mask == 0
        
        # Passando as entradas pelo modelo BERT (que já contém a camada de embedding)
        x = self.bert(x, doy, mask)
        
        # Saída do Transformer, podemos aplicar a camada linear para prever os valores de produtividade
        x = self.linear(x)

        return x


import pickle
import numpy as np

# Caminho para o arquivo pickle
caminho_pickle = 'conf_mat.pkl'

# Abrindo o arquivo pickle em modo leitura binária
with open(caminho_pickle, 'rb') as arquivo:
    objeto = pickle.load(arquivo)
    objeto = (objeto / 100).astype(np.int16)

# Agora você pode usar o objeto carregado
print(objeto)

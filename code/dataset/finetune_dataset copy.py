import torch
from torch.utils.data import Dataset
import numpy as np
import concurrent.futures
import geopandas as gpd
from tqdm import tqdm
from dataset.data_augmentation import transform


# Função para preparar amostra
# Ajuste na função prepare_sample
def prepare_sample(sample_paths, max_length, norm, augment, year_range):
    """
    Prepara amostras concatenando séries temporais de múltiplos anos e seus respectivos valores de produtividade.
    """
    ts_origin_list = []
    doy_list = []
    productivity_list = []  # Lista para armazenar produtividade para cada ano
    for sample_path in sample_paths:
        with np.load(sample_path) as sample:
            # produtividade contínua para o respectivo ano
            productivity = sample[f"productivity"]  # Substitua se necessário
            productivity_list.append(productivity)  # Armazena o valor de produtividade

            # cloudfree image patch time series
            ts_origin = sample["ts"]  # [seq_Length, band_nums, patch_size, patch_size]
            if norm is not None:
                m, s = norm
                m = np.expand_dims(m, axis=-1)
                s = np.expand_dims(s, axis=-1)
                shape = ts_origin.shape
                ts_origin = ts_origin.reshape((shape[0], shape[1], -1))
                ts_origin = (ts_origin - m) / s
                ts_origin = ts_origin.reshape(shape)
            else:
                ts_origin = ts_origin / 10000.0

            ts_origin = ts_origin.astype(np.float32)

            if augment:
                ts_origin = transform(ts_origin)

            ts_origin_list.append(ts_origin)

            # acquisition dates of this time series
            doy = sample["doy"]  # [seq_Length, ]
            doy_list.append(doy)

    # Concatena ao longo do eixo temporal (primeira dimensão)
    ts_origin_concat = np.concatenate(ts_origin_list, axis=0)
    doy_concat = np.concatenate(doy_list, axis=0)
    print(doy_concat)

    # Obtém o comprimento total da série concatenada
    ts_length = ts_origin_concat.shape[0]

    if ts_length > max_length:
        # Recorte se exceder max_length
        ts_origin_concat = ts_origin_concat[:max_length]
        doy_concat = doy_concat[:max_length]
        ts_length = max_length
    else:
        # Padding time series para o mesmo comprimento
        ts_origin_concat = np.pad(
            ts_origin_concat,
            ((0, max_length - ts_length), (0, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        doy_concat = np.pad(
            doy_concat,
            (0, max_length - ts_length),
            mode="constant",
            constant_values=0,
        )


    # Máscara de observações válidas
    bert_mask = np.zeros((max_length,), dtype=np.int16)
    bert_mask[:ts_length] = 1
    return ts_origin_concat, bert_mask, doy_list, productivity_list


class FinetuneDataset(Dataset):
    def __init__(
        self,
        file_path,
        num_features,
        patch_size,
        max_length,
        norm=None,
        only_column="",
        start_year=2019,
        end_year=2023
    ):
        """
        Dataset para ajuste fino (fine-tuning) com múltiplos anos de produtividade (2019-2022).
        :param file_path: Caminho para o arquivo de dados em parquet.
        :param num_features: Número de features.
        :param patch_size: Tamanho do patch.
        :param max_length: Comprimento máximo para padding.
        :param norm: Normalização (média e desvio padrão).
        :param only_column: Filtra as amostras de uma coluna específica.
        :param start_year: Ano inicial.
        :param end_year: Ano final.
        """
        self.file_path = file_path
        self.max_length = max_length
        self.dimension = num_features
        self.patch_size = patch_size

        # Lê o arquivo e aplica o filtro da coluna se necessário
        if only_column:
            gdf = gpd.read_parquet(file_path)
            gdf = gdf[gdf[only_column]].reset_index(drop=True)
        else:  # Predicting
            gdf = gpd.read_parquet(file_path)

        # Armazena os caminhos dos arquivos
        self.FileList = [gdf[f"downloaded_filepath_{year}"].to_list() for year in range(start_year, end_year)]

        self.TS_num = len(gdf)  # Número de amostras (uma linha por amostra)
        self.norm = norm
        self.year_range = list(range(start_year, end_year))

        # Inicializa arrays para armazenar as informações de produtividade
        self.ts_origins = np.zeros(
            (self.TS_num, max_length, num_features, patch_size, patch_size),
            dtype=np.float32,
        )
        self.bert_masks = np.zeros((self.TS_num, max_length), dtype=np.int16)
        self.timestamps = np.zeros((self.TS_num, len(self.year_range)), dtype=np.int16)
        self.productivities = np.zeros((self.TS_num, len(self.year_range)), dtype=np.float32)

        # Processamento paralelo para cada linha de amostra (representando uma área)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for n in range(self.TS_num):
                # Para cada linha, iteramos sobre os arquivos de cada ano
                sample_paths = [self.FileList[year][n] for year in range(len(self.FileList))]
                
                # Submete o trabalho de preparação das amostras para execução paralela
                futures.append(
                    executor.submit(
                        prepare_sample,
                        sample_paths,
                        self.max_length,
                        self.norm,
                        only_column == only_column,
                        self.year_range
                    )
                )

            # Coleta os resultados conforme o processamento vai acontecendo
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                miniters=100,
                desc=f"Building{' ' + only_column} dataset...",
            ):
                # Desempacotando o índice e os resultados da amostra
                (ts_origin, bert_mask, doy, productivity_list) = future.result()
                
                # Agora você pode atribuir os valores corretamente
                self.ts_origins[n] = ts_origin
                self.bert_masks[n] = bert_mask
                self.timestamps[n] = np.array(doy)
                self.productivities[n] = np.array(productivity_list)

    def __len__(self):
        return self.TS_num

    def __getitem__(self, idx):
        """
        Retorna o item no índice especificado.
        """
        return {
            "bert_input": torch.from_numpy(self.ts_origins[idx]),
            "bert_mask": torch.from_numpy(self.bert_masks[idx]),
            "timestamp": torch.from_numpy(self.timestamps[idx]),
            "bert_target": torch.tensor(self.productivities[idx], dtype=torch.float32),
        }

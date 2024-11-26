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
    # Modificação para lidar com múltiplos anos
    ts_origin_list = []
    doy_list = []
    productivity_list = []
    bert_mask_list = []

    for sample_path in sample_paths:
        with np.load(sample_path) as sample:
            # Carregando produtividade
            productivity = sample[f"productivity"]
            productivity_list.append(productivity)

            # Carregando série temporal
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

            # Carregando DOY
            doy = sample["doy"]  # [seq_Length, ]
            doy_list.append(doy)

            # Criando a máscara (1 para observações válidas, 0 para padding)
            ts_length = ts_origin.shape[0]
            mask = np.zeros((max_length,), dtype=np.int16)
            mask[:ts_length] = 1
            bert_mask_list.append(mask)

    # Mantém as listas separadas
    return ts_origin_list, bert_mask_list, doy_list, productivity_list


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
        self.norm = norm
        self.year_range = list(range(start_year, end_year + 1))

        # Lê o arquivo e aplica o filtro da coluna se necessário
        if only_column:
            gdf = gpd.read_parquet(file_path)
            gdf = gdf[gdf[only_column]].reset_index(drop=True)
        else:  # Predicting
            gdf = gpd.read_parquet(file_path)

        # Armazena os caminhos dos arquivos
        self.FileList = [gdf[f"downloaded_filepath_{year}"].to_list() for year in self.year_range]
        self.TS_num = len(gdf)  # Número de amostras (uma linha por amostra)

        # Armazena informações temporais por ano
        self.ts_origins = {year: [] for year in self.year_range}
        self.bert_masks = {year: [] for year in self.year_range}
        self.timestamps = {year: [] for year in self.year_range}
        self.productivities = {year: [] for year in self.year_range}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for n in range(self.TS_num):
                # Para cada linha, iteramos sobre os arquivos de cada ano
                sample_paths = [self.FileList[year_idx][n] for year_idx in range(len(self.FileList))]
                
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
            for n, future in enumerate(
                tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    miniters=100,
                    desc=f"Building{' ' + only_column} dataset...",
                )
            ):
                # Desempacota o resultado do futuro
                (ts_origin, bert_mask, doy_list, productivity_list) = future.result()

                # Armazena os resultados para cada ano
                for i, year in enumerate(self.year_range):
                    self.ts_origins[year].append(ts_origin[i])
                    self.bert_masks[year].append(bert_mask[i])
                    self.timestamps[year].append(doy_list[i])
                    self.productivities[year].append(productivity_list[i])

        # Convertendo listas para arrays NumPy
        max_shape = (self.max_length, self.dimension, self.patch_size, self.patch_size)
        for year in self.year_range:
            # Padroniza `ts_origins`
            padded_ts_list = [
                np.pad(
                    ts,
                    [(0, max_shape[0] - ts.shape[0])] + [(0, 0)] * (len(max_shape) - 1),
                    mode="constant",
                    constant_values=0,
                ) if ts.shape[0] < max_shape[0] else ts[:max_shape[0]]
                for ts in self.ts_origins[year]
            ]
            self.ts_origins[year] = np.array(padded_ts_list, dtype=np.float32)

            # Padroniza `bert_masks`
            self.bert_masks[year] = np.array(
                [np.pad(mask, (0, self.max_length - len(mask)), mode="constant", constant_values=0)
                 for mask in self.bert_masks[year]],
                dtype=np.int16,
            )

            # Padroniza e converte `timestamps`
            padded_timestamps = [
                np.pad(
                    ts,
                    (0, self.max_length - len(ts)),
                    mode="constant",
                    constant_values=0
                ) if len(ts) < self.max_length else ts[:self.max_length]
                for ts in self.timestamps[year]
            ]
            self.timestamps[year] = np.array(padded_timestamps, dtype=np.int16)

            # Converte `productivities` para NumPy
            self.productivities[year] = np.array(self.productivities[year], dtype=np.float32)

    def __len__(self):
        return self.TS_num

    def __getitem__(self, idx):
        bert_input = torch.stack([torch.tensor(self.ts_origins[year][idx], dtype=torch.float32) for year in self.year_range], dim=0)
        bert_mask = torch.stack([torch.tensor(self.bert_masks[year][idx], dtype=torch.float32) for year in self.year_range], dim=0)
        timestamp = torch.stack([torch.tensor(self.timestamps[year][idx], dtype=torch.float32) for year in self.year_range], dim=0)
        bert_target = torch.stack([torch.tensor(self.productivities[year][idx], dtype=torch.float32) for year in self.year_range], dim=0)

        return {
            "bert_input": bert_input,
            "bert_mask": bert_mask,
            "timestamp": timestamp,
            "bert_target": bert_target,
        }

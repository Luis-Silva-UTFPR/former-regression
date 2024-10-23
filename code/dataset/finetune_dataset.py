import torch
from torch.utils.data import Dataset
import numpy as np
import concurrent.futures
import geopandas as gpd
from tqdm import tqdm
from dataset.data_augmentation import transform


# Função para preparar amostra
def prepare_sample(sample_paths, max_length, norm, augment):
    """
    Prepara amostras concatenando séries temporais de múltiplos anos e seus respectivos valores de produtividade.
    """
    ts_origin_list = []
    doy_list = []
    productivity_list = []  # Lista para armazenar produtividade para cada ano

    for sample_path in sample_paths:
        with np.load(sample_path) as sample:
            # produtividade contínua para o respectivo ano
            productivity = sample["productivity"]  # <- Substituir pela coluna correta
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

    # Obtém o comprimento total da série concatenada
    ts_length = ts_origin_concat.shape[0]

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

    return ts_origin_concat, bert_mask, doy_concat, productivity_list


# Classe FinetuneDataset para múltiplos anos
class FinetuneDataset(Dataset):
    def __init__(
        self,
        file_path,
        num_features,
        patch_size,
        max_length,
        norm=None,
        only_column="",
        start_year=2017,
        end_year=2023
    ):
        """
        Dataset para ajuste fino (fine-tuning) em múltiplos anos de séries temporais.
        :param file_path: Caminho para o arquivo de dados em parquet.
        :param num_features: Número de features.
        :param patch_size: Tamanho do patch.
        :param max_length: Comprimento máximo para padding.
        :param norm: Normalização (media e std).
        :param only_column: Filtra as amostras de uma coluna específica.
        :param start_year: Ano inicial.
        :param end_year: Ano final.
        """
        self.file_path = file_path
        self.max_length = max_length
        self.dimension = num_features
        self.patch_size = patch_size

        # Lê o arquivo e aplica o filtro da coluna se necessário
        gdf = gpd.read_parquet(file_path)
        if only_column:
            gdf = gdf[gdf[only_column]].reset_index(drop=True)

        # Lista de arquivos para múltiplos anos (de 2017 a 2023, por exemplo)
        self.FileList = [
            [gdf[f"downloaded_filepath_{year}"].to_list() for year in range(start_year, end_year + 1)]
        ]

        self.TS_num = len(self.FileList[0])  # número de amostras
        self.norm = norm

        self.ts_origins = np.zeros(
            (self.TS_num, max_length, num_features, patch_size, patch_size),
            dtype=np.float32,
        )
        self.bert_masks = np.zeros((self.TS_num, max_length), dtype=np.int16)
        self.timestamps = np.zeros((self.TS_num, max_length), dtype=np.int16)

        # Armazena a produtividade contínua para múltiplos anos
        self.productivities = np.zeros((self.TS_num, len(range(start_year, end_year + 1))), dtype=np.float32)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for n, sample_paths in enumerate(self.FileList):
                futures.append(
                    executor.submit(
                        lambda x_sample_paths, x_max_length, x_norm, x_augment, x_n: [
                            x_n,
                            prepare_sample(
                                x_sample_paths, x_max_length, x_norm, x_augment
                            ),
                        ],
                        [f"../{path}" for path in sample_paths],  # adaptado para múltiplos anos
                        self.max_length,
                        self.norm,
                        only_column == "train",
                        n,
                    )
                )

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                miniters=100,
                desc=f"Building{' ' + only_column} dataset...",
            ):
                n, (ts_origin, bert_mask, doy, productivity_list) = future.result()
                self.ts_origins[n] = ts_origin
                self.bert_masks[n] = bert_mask
                self.timestamps[n] = doy
                self.productivities[n] = productivity_list  # Armazena a lista de produtividades

    def __len__(self):
        return self.TS_num

    def __getitem__(self, idx):
        """
        Retorna o item no índice especificado.
        """
        return {
            "ts_origin": torch.from_numpy(self.ts_origins[idx]),
            "bert_mask": torch.from_numpy(self.bert_masks[idx]),
            "timestamps": torch.from_numpy(self.timestamps[idx]),
            "productivity": torch.tensor(self.productivities[idx], dtype=torch.float32),
        }

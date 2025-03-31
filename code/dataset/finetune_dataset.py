from torch.utils.data import Dataset
import torch
import numpy as np
import geopandas as gpd
from dataset.data_augmentation import transform
import concurrent.futures
from tqdm.std import tqdm

import numpy as np
from scipy.signal import savgol_filter

# def prepare_sample(sample_path, max_length, norm, augment, prod, area_ha, id):
#     try:
#         with np.load(sample_path) as sample:
#             # class label of this time series
#             class_label = sample["class_label"]
            
#             # cloudfree image patch time series
#             ts_origin = sample["ts"]  # [seq_Length, band_nums, patch_size, patch_size]
#             class_label = (prod / area_ha)

#             B3 = ts_origin[:, 1, :, :]  # Banda do verde (Green)
#             B4 = ts_origin[:, 2, :, :]  # Banda do vermelho (Red)
#             B8 = ts_origin[:, 6, :, :]  # Banda do infravermelho próximo (NIR)

#             # NDVI = (B8 - B4) / (B8 + B4)
#             ndvi = (B8 - B4) / (B8 + B4 + 1e-10)  # Adiciona 1e-10 para evitar divisão por zero
#             ndvi_mean = np.mean(ndvi, axis=(1, 2))  # Média ao longo dos pixels (5x5)

#             # Suavizar NDVI com filtro Savitzky-Golay
#             if len(ndvi_mean) >= 5:  # Para Savitzky-Golay o comprimento mínimo é 5
#                 ndvi_smooth = savgol_filter(ndvi_mean, window_length=5, polyorder=2)  # Janela de 5 pontos e polinômio de ordem 2
#             else:
#                 ndvi_smooth = ndvi_mean

#             # Encontrar o índice do pico (máximo valor de NDVI)
#             peak_idx = np.argmax(ndvi_smooth)
#             doy = sample["doy"]  # [seq_Length, ]

#             # Determinar os limites de corte (60 dias antes e depois do pico)
#             if len(doy) > 0:
#                 peak_doy = doy[peak_idx]
#                 start_doy = peak_doy - 60
#                 end_doy = peak_doy + 60

#                 # Filtrar série temporal para manter apenas o intervalo desejado
#                 mask = (doy >= start_doy)
                
#                 ts_origin = ts_origin[mask]
#                 doy = doy[mask]
#                 ndvi_mean = ndvi_mean[mask]
            
#             # Normalização
#             if norm is not None:
#                 m, s = norm
#                 m = np.expand_dims(m, axis=-1)
#                 s = np.expand_dims(s, axis=-1)
#                 shape = ts_origin.shape
#                 ts_origin = ts_origin.reshape((shape[0], shape[1], -1))
#                 ts_origin = (ts_origin - m) / s
#                 ts_origin = ts_origin.reshape(shape)
#             else:
#                 ts_origin = ts_origin / 10000.0

#             ts_origin = ts_origin.astype(np.float32)
#             if augment:
#                 ts_origin = transform(ts_origin)

#             # Tamanho da série temporal após corte
#             ts_length = ts_origin.shape[0]

#             # Padding da série temporal para o comprimento máximo
#             ts_origin = np.pad(
#                 ts_origin,
#                 ((0, max_length - ts_length), (0, 0), (0, 0), (0, 0)),
#                 mode="constant",
#                 constant_values=0.0
#             )

#             # Padding do DOY
#             doy = np.pad(
#                 doy,
#                 (0, max_length - ts_length),
#                 mode="constant",
#                 constant_values=0
#             )

#             # Padding do NDVI
#             ndvi_mean = np.pad(
#                 ndvi_mean,
#                 (0, max_length - ts_length),
#                 mode="constant",
#                 constant_values=0
#             )

#             # Máscara de observação válida
#             bert_mask = np.zeros((max_length,), dtype=np.int16)
#             bert_mask[:ts_length] = 1

#             return (
#                 ts_origin,
#                 bert_mask,
#                 doy,
#                 class_label
#             )
#     except Exception as e:
#         print(e)
#         print(id)



def prepare_sample(sample_path, max_length, norm, augment, prod, area_ha, id):
    try:
        with np.load(sample_path) as sample:
            # class label of this time series
            class_label = sample["class_label"]
            
            # cloudfree image patch time series
            ts_origin = sample["ts"]  # [seq_Length, band_nums, patch_size, patch_size]
            class_label = (prod / area_ha)
            # class_label = prod

            B3 = ts_origin[:, 1, :, :]  # Banda do verde (Green)
            B4 = ts_origin[:, 2, :, :]  # Banda do vermelho (Red)
            B8 = ts_origin[:, 6, :, :]  # Banda do infravermelho próximo (NIR)

            # NDVI = (B8 - B4) / (B8 + B4)
            ndvi = (B8 - B4) / (B8 + B4 + 1e-10)  # Adiciona 1e-10 para evitar divisão por zero

            # NDWI = (B3 - B8) / (B3 + B8)
            ndwi = (B3 - B8) / (B3 + B8 + 1e-10)  # Adiciona 1e-10 para evitar divisão por zero

            ndvi_mean = np.mean(ndvi, axis=(1, 2))  # Média ao longo dos pixels (5x5)
            ndwi_mean = np.mean(ndwi, axis=(1, 2))  # Média ao longo dos pixels (5x5)

            ndvi_list = ndvi_mean.tolist()  
            ndwi_list = ndwi_mean.tolist()


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

            # length of this time series (varies for each sample)
            ts_length = ts_origin.shape[0]
            
            # padding time series to the same length
            try:
                ts_origin = np.pad(
                    ts_origin,
                    ((0, max_length - ts_length), (0, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            except:
                print(id)

            # acquisition dates of this time series
            doy = sample["doy"]  # [seq_Length, ]
            try:
                doy = np.pad(
                    doy,
                    (0, max_length - ts_length),
                    mode="constant",
                    constant_values=0,
                )
            except:
                print(id)

            ndvi_list = np.pad(
                ndvi_list,
                (0, max_length - ts_length),
                mode="constant",
                constant_values=0,
            )
            ndwi_list = np.pad(
                ndwi_list,
                (0, max_length - ts_length),
                mode="constant",
                constant_values=0,
            )

            # mask of valid observations
            bert_mask = np.zeros((max_length,), dtype=np.int16)
            bert_mask[:ts_length] = 1

            return (
                ts_origin,
                bert_mask,
                doy,
                class_label,
                # area_ha,
                # ndvi_list,
                # ndwi_list,
            )
    except Exception as e:
        print(e)
        print(id)


class FinetuneDataset(Dataset):
    def __init__(
        self,
        file_path,
        num_features,
        patch_size,
        max_length,
        norm=None,
        is_train=False,
        only_column=""
    ):
        """
        :param file_path: path to the folder of the pre-training dataset
        :param num_features: dimension of each pixel
        :param patch_size: patch size
        :param max_length: padded sequence length
        :param norm: mean and std used to normalize the input reflectance
        """

        self.file_path = file_path
        self.max_length = max_length
        self.dimension = num_features
        self.patch_size = patch_size
        self.is_train = is_train

        if only_column:  # Train, validate or test
            gdf: gpd.GeoDataFrame = gpd.read_parquet(file_path)
            gdf = gdf[gdf[only_column]].reset_index(drop=True)
            self.FileList = gdf.downloaded_filepath.to_list()
        else:  # Predicting
            gdf: gpd.GeoDataFrame = gpd.read_parquet(file_path)
            self.FileList = gpd.read_parquet(
                file_path
            ).downloaded_filepath.to_list()

        if only_column:
            self.productivity = gdf["prod"].to_numpy()
            self.area_ha = gdf["area_ha"].to_numpy()
        else:
            self.productivity = None

        self.productivity = gdf["prod"].to_numpy()
        self.area_ha = gdf["area_ha"].to_numpy()
        self.id = gdf["id"].to_numpy()

        self.TS_num = len(self.FileList)  # number of labeled samples
        self.norm = norm
        self.ts_origins = np.zeros(
            (self.TS_num, max_length, num_features, patch_size, patch_size),
            dtype=np.float32,
        )
        self.bert_masks = np.zeros((self.TS_num, max_length), dtype=np.int16)
        self.timestamps = np.zeros((self.TS_num, max_length), dtype=np.int16)
        self.class_labels = np.zeros((self.TS_num, 1), dtype=np.float32)
        self.areas_ha = np.zeros((self.TS_num, 1), dtype=np.float32)
        self.ndvi_timestamps = np.zeros((self.TS_num, max_length), dtype=np.float32)
        self.ndwi_timestamps = np.zeros((self.TS_num, max_length), dtype=np.float32)

        

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for n, sample_path in enumerate(self.FileList):
                futures.append(
                    executor.submit(
                        lambda x_sample_path, x_max_length, x_norm, x_augment, x_n, x_prod, area_ha, id: [
                            x_n,
                            prepare_sample(
                                x_sample_path, x_max_length, x_norm, x_augment, x_prod[x_n], area_ha[x_n], id[x_n]
                            ),
                        ],
                        f"../{sample_path}",
                        self.max_length,
                        self.norm,
                        only_column == "train",
                        n,
                        self.productivity,
                        self.area_ha,
                        self.id
                    )
                )

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                miniters=100,
                desc=f"Building{' ' + only_column} dataset...",
            ):
                n, (ts_origin, bert_mask, doy, class_label) = future.result()
                self.ts_origins[n] = ts_origin
                self.bert_masks[n] = bert_mask
                self.timestamps[n] = doy
                self.class_labels[n] = class_label
                # n, (ts_origin, bert_mask, doy, class_label, area_ha, ndvi, ndwi) = future.result()
                # self.ts_origins[n] = ts_origin
                # self.bert_masks[n] = bert_mask
                # self.timestamps[n] = doy
                # self.class_labels[n] = class_label
                # self.areas_ha[n] = np.float32(area_ha)
                # self.ndvi_timestamps[n] = ndvi
                # self.ndwi_timestamps[n] = ndwi

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):
        output = {
            "bert_input": self.ts_origins[item],
            "bert_mask": self.bert_masks[item],
            "timestamp": self.timestamps[item],
            "class_label": self.class_labels[item]
            # "area_ha": self.areas_ha[item],
            # "ndvi": self.ndvi_timestamps[item],
            # "ndwi": self.ndwi_timestamps[item]
        }

        torch_tensors = {key: torch.from_numpy(value).float() for key, value in output.items()}
        return torch_tensors

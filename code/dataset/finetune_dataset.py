from torch.utils.data import Dataset
import torch
import numpy as np
import geopandas as gpd
from dataset.data_augmentation import transform
import concurrent.futures
from tqdm.std import tqdm


def prepare_sample(sample_path, max_length, norm, augment):
    with np.load(sample_path) as sample:
        # class label of this time series
        class_label = sample["class_label"]

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

        # length of this time series (varies for each sample)
        ts_length = ts_origin.shape[0]

        # padding time series to the same length
        ts_origin = np.pad(
            ts_origin,
            ((0, max_length - ts_length), (0, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        # acquisition dates of this time series
        doy = sample["doy"]  # [seq_Length, ]
        doy = np.pad(
            doy,
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
        )


class FinetuneDataset(Dataset):
    def __init__(
        self, file_path, num_features, patch_size, max_length, norm=None, only_column=""
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

        if only_column:  # Train, validate or test
            gdf = gpd.read_parquet(file_path)
            gdf = gdf[gdf[only_column]].reset_index(drop=True)
            self.FileList = gdf.downloaded_filepath_2015_2016.to_list()
        else:  # Predicting
            self.FileList = gpd.read_parquet(
                file_path
            ).downloaded_filepath_2015_2016.to_list()

        self.TS_num = len(self.FileList)  # number of labeled samples
        self.norm = norm

        self.ts_origins = np.zeros(
            (self.TS_num, max_length, num_features, patch_size, patch_size),
            dtype=np.float32,
        )
        self.bert_masks = np.zeros((self.TS_num, max_length), dtype=np.int16)
        self.timestamps = np.zeros((self.TS_num, max_length), dtype=np.int16)
        self.class_labels = np.zeros((self.TS_num, 1), dtype=np.int16)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for n, sample_path in enumerate(self.FileList):
                futures.append(
                    executor.submit(
                        lambda x_sample_path, x_max_length, x_norm, x_augment, x_n: [
                            x_n,
                            prepare_sample(
                                x_sample_path, x_max_length, x_norm, x_augment
                            ),
                        ],
                        f"../{sample_path}",
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
                n, (ts_origin, bert_mask, doy, class_label) = future.result()
                self.ts_origins[n] = ts_origin
                self.bert_masks[n] = bert_mask
                self.timestamps[n] = doy
                self.class_labels[n] = class_label

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):
        output = {
            "bert_input": self.ts_origins[item],
            "bert_mask": self.bert_masks[item],
            "timestamp": self.timestamps[item],
            "class_label": self.class_labels[item],
        }

        return {key: torch.from_numpy(value) for key, value in output.items()}
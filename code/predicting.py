import torch
from torch.utils.data import DataLoader
from model import BERT
from trainer import BERTFineTuner
from dataset import FinetuneDataset
import numpy as np
import random
import os
import argparse
import geopandas as gpd
from sklearn.metrics import classification_report, confusion_matrix


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def Config():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        required=True,
        help="Path to the labeled dataset.",
    )
    parser.add_argument(
        "--pretrain_path",
        default="../checkpoints/pretrain",
        type=str,
        required=False,
        help="The storage path of the pre-trained model parameters.",
    )
    parser.add_argument(
        "--finetune_path",
        default="../checkpoints/finetune",
        type=str,
        required=False,
        help="The output directory where the fine-tuning checkpoints will be written.",
    )
    parser.add_argument(
        "--with_cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether cuda is available.",
    )
    parser.add_argument(
        "--cuda_devices",
        default=None,
        type=int,
        help="List of cuda devices.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of loader worker processes.",
    )
    parser.add_argument(
        "--max_length",
        default=75,
        type=int,
        help="The maximum length of input time series. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--patch_size",
        default=5,
        type=int,
        help="Size of the input patches.",
    )
    parser.add_argument(
        "--num_features",
        default=10,
        type=int,
        help="The dimensionality of satellite observations.",
    )
    parser.add_argument(
        "--num_classes",
        default=15,
        type=int,
        help="Number of classes.",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="Number of hidden neurons of the Transformer network.",
    )
    parser.add_argument(
        "--layers",
        default=3,
        type=int,
        help="Number of layers of the Transformer network.",
    )
    parser.add_argument(
        "--attn_heads",
        default=8,
        type=int,
        help="Number of attention heads of the Transformer network.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-4,
        type=float,
        help="",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
        help="",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_seed(0)
    config = Config()

    pred_path = os.path.join(config.dataset_path)

    print("Loading datasets...")
    pred_dataset = FinetuneDataset(
        pred_path,
        config.num_features,
        config.patch_size,
        config.max_length,
        only_column="",
    )

    print("Predicting samples: %d" % (pred_dataset.TS_num))

    print("Creating dataloader...")
    pred_data_loader = DataLoader(
        pred_dataset,
        shuffle=False,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        drop_last=False,
    )

    print("Initialing SITS-Former...")
    bert = BERT(
        num_features=config.num_features,
        hidden=config.hidden_size,
        n_layers=config.layers,
        attn_heads=config.attn_heads,
        dropout=config.dropout,
    )
    if config.pretrain_path is not None:
        print("Loading pre-trained model parameters...")
        bert_path = os.path.join(config.pretrain_path, "checkpoint.bert.tar")
        if os.path.exists(bert_path):
            bert.load_state_dict(
                torch.load(bert_path, map_location=torch.device("cpu"))
            )
        else:
            print("Cannot find the pre-trained parameter file, please check the path!")

    # Work from here, don't have classification metrics anymore
    trainer = BERTFineTuner(
        bert,
        config.num_classes,
        train_loader=None,
        valid_loader=None,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        with_cuda=config.with_cuda,
        cuda_devices=config.cuda_devices,
    )

    print("Testing SITS-Former...")
    trainer.load(config.finetune_path)
    y_pred = trainer.predict(pred_data_loader)

    df = gpd.read_parquet(pred_path)
    df["y_pred"] = y_pred
    df["crop_number"] = pred_dataset.class_labels

    print(
        classification_report(
            df["crop_number"], df["y_pred"], labels=list(range(config.num_classes))
        )
    )

    print(
        confusion_matrix(
            df["crop_number"], df["y_pred"], labels=list(range(config.num_classes))
        )
    )

    df.to_parquet("../data/output.parquet")
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from model import BERT
from trainer import BERTFineTuner
from dataset import FinetuneDataset
import numpy as np
import random
import os
import argparse
from tqdm.auto import tqdm
import pandas as pd

import torch
import numpy as np

class JitterTransform:
    def __init__(self, sigma=0.05):
        self.sigma = sigma  # Define a intensidade do ruído

    def __call__(self, sample):
        x, y = sample  # x = entrada, y = rótulo
        noise = torch.randn_like(x) * self.sigma  # Gera ruído gaussiano
        return x + noise, y  # Adiciona o ruído à entrada
    

class RandomMaskingTransform:
    def __init__(self, mask_prob=0.1):
        self.mask_prob = mask_prob  # Probabilidade de mascarar um valor

    def __call__(self, sample):
        x, y = sample
        mask = torch.rand_like(x) > self.mask_prob  # Máscara booleana
        return x * mask, y  # Aplica a máscara


class CombinedTransform:
    def __init__(self, jitter_sigma=0.05, mask_prob=0.1):
        self.jitter = JitterTransform(sigma=jitter_sigma)
        self.masking = RandomMaskingTransform(mask_prob=mask_prob)

    def __call__(self, sample):
        sample = self.jitter(sample)  # Primeiro aplica jitter
        sample = self.masking(sample)  # Depois aplica masking
        return sample


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def setup_seed(seed):
#     # torch.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     # torch.backends.cudnn.deterministic = True


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
        # default=None,
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
        default=16,
        type=int,
        help="Number of loader worker processes.",
    )
    parser.add_argument(
        "--max_length",
        default=95,
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
        default=1,
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
        default=64,
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
    # Verifica se a CUDA (GPU) está disponível e define o dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_path = os.path.join(config.dataset_path)
    valid_path = os.path.join(config.dataset_path)
    test_path = os.path.join(config.dataset_path)

    print("Loading datasets...")
    train_dataset = FinetuneDataset(
        train_path,
        config.num_features,
        config.patch_size,
        config.max_length,
        only_column="train",
        is_train=False
    )
    valid_dataset = FinetuneDataset(
        valid_path,
        config.num_features,
        config.patch_size,
        config.max_length,
        only_column="validate",
    )
    test_dataset = FinetuneDataset(
        test_path,
        config.num_features,
        config.patch_size,
        config.max_length,
        only_column="test",
    )
    print(
        "Training samples: %d, validation samples: %d, testing samples: %d"
        % (train_dataset.TS_num, valid_dataset.TS_num, test_dataset.TS_num)
    )

    print("Creating dataloader...")
    train_data_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        drop_last=False,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        drop_last=False,
    )
    test_data_loader = DataLoader(
        test_dataset,
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
    ).to(device)
    if config.pretrain_path is not None:
        print("Loading pre-trained model parameters...")
        bert_path = os.path.join(config.pretrain_path, "checkpoint.bert.tar")
        if os.path.exists(bert_path):
            bert.load_state_dict(
                torch.load(bert_path, map_location=torch.device("cpu")), strict=False
            )
        else:
            print("Cannot find the pre-trained parameter file, please check the path!")

    print("Creating downstream task trainer...")
    trainer = BERTFineTuner(
        bert,
        config.num_classes,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        with_cuda=config.with_cuda,
        cuda_devices=config.cuda_devices,
    )

    print("Training/Fine-tuning SITS-Former...")

    # Inicialização para o critério de melhores desempenhos
    Best_MAE = float('inf')
    Best_MSE = float('inf')
    Best_R2 = float('-inf')
    Best_Vloss = float('inf')
    best_yp = []
    best_y = []
    best_Tr2 = float('-inf')
    best_Tmae = float('-inf')

    for epoch in tqdm(range(config.epochs), miniters=1, unit="epoch"):
        train_loss, valid_loss, valid_metrics, y_p, y, train_r2, train_mae = trainer.train(epoch)
        valid_mae = valid_metrics["MAE"]
        valid_mse = valid_metrics["MSE"]
        valid_r2 = valid_metrics["R2"]

        
        # if valid_loss < Best_Vloss and train_r2 < 1.5 * valid_r2:
        # if valid_loss < Best_Vloss:
        if valid_r2 > Best_R2:
            Best_Vloss = valid_loss
            Best_MAE = valid_mae
            Best_MSE = valid_mse
            Best_R2 = valid_r2
            best_yp = y_p
            best_y = y
            best_Tr2 = train_r2
            best_Tmae = train_mae
            trainer.save(epoch, config.finetune_path)
    
    print("Train metrics: MAE: %.4f, R²: %.4f" % (best_Tmae, best_Tr2))

    if isinstance(best_yp, torch.Tensor):
        best_yp = best_yp.cpu().numpy()  # Converter para NumPy se estiver na GPU
    if isinstance(best_y, torch.Tensor):
        best_y = best_y.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(best_y, best_yp, alpha=0.5, label="Predições vs valores reais")
    plt.xlim(min(best_y), max(best_y))  # Ajusta o eixo X para focar em valores menores
    plt.ylim(min(best_y), max(best_y))

    # Adicionar a linha de referência (y = x)
    plt.plot([min(best_y), max(best_y)], [min(best_y), max(best_y)], color='red', linestyle='--', label="y = x")

    # Adicionar rótulos e título
    plt.xlabel("Y true")
    plt.ylabel("Y pred")
    plt.title("Y true vs Y pred")
    plt.legend()

    # Mostrar o gráfico
    plt.savefig('../plots_validation/estim.png')
    plt.close()


    try:
        trainer.plot_metrics()
    except:
        pass

    print(
        "Best performance on the validation set: MAE = %.4f, MSE = %.4f, R2 = %.4f"
        % (Best_MAE, Best_MSE, Best_R2)
    )

    print("\n")
    print("Testing SITS-Former...")
    trainer.load(config.finetune_path)

    # Supondo que o método 'test' retorne as métricas como um dicionário com as chaves 'MAE', 'MSE', 'R2'
    test_metrics, y_true, y_pred = trainer.test(test_data_loader)

    # y_true = 
    test_mae = test_metrics["MAE"]
    test_mse = test_metrics["MSE"]
    test_r2 = test_metrics["R2"]


    print(
        "Best performance on the test set KG/HA: MAE = %.4f, MSE = %.4f, R2 = %.4f"
        % (test_mae, test_mse, test_r2)
    )

    df = pd.DataFrame(
        data = {
            "y_pred": y_pred,
            "y_true": y_true
        }
    )
    df.to_csv("/mnt/c3691c07-b3dd-4b44-a41f-1a86553ea058/luis/the_best_ones/the_new_one/test_data.csv")

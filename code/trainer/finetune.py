import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from model import BERT, BERTPrediction

# from .metric import Average_Accuracy, Kappa_Coefficient
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)
from tqdm.auto import tqdm

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Absolute Error: {mae}, R² Score: {r2}")
    return mae, r2


class BERTFineTuner:
    def __init__(
        self,
        bert: BERT,
        num_features: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        decay_gamma: float = 0.99,
        gradient_clipping_value=5.0,
        with_cuda: bool = True,
        cuda_devices=None,
    ):
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(
            "cuda"
            if cuda_condition
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )

        print(f"Running on {self.device}...")

        self.bert = bert
        self.model = BERTPrediction(bert, num_features).to(self.device)
        self.num_classes = num_features

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.warmup_epochs = warmup_epochs
        self.optim_schedule = lr_scheduler.ExponentialLR(self.optim, gamma=decay_gamma)
        self.gradient_clippling = gradient_clipping_value
        self.criterion = nn.MSELoss(reduction="none")

        if with_cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(
                    "Using %d GPUs for model pre-training" % torch.cuda.device_count()
                )
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            torch.backends.cudnn.benchmark = True

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        number_parameters = (
            sum([p.nelement() for p in self.model.parameters()]) / 1000000
        )
        print("Total Parameters: %.2f M" % number_parameters)

    def train(self, epoch):
        self.model.train()

        data_iter = tqdm(
            enumerate(self.train_loader),
            desc="EP_%s:%d" % ("train", epoch),
            total=len(self.train_loader),
            bar_format="{l_bar}{r_bar}",
        )

        train_loss = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            mask_prediction = self.model(
                data["bert_input"].float(),
                data["timestamp"].long(),
                data["bert_mask"].long(),
            )

            loss = self.criterion(mask_prediction, data["bert_target"].float())
            mask = data["loss_mask"].unsqueeze(-1)
            loss = (loss * mask.float()).sum() / mask.sum()

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clippling)
            self.optim.step()

            train_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": train_loss / (i + 1),
                "loss": loss.item(),
            }

            if i % 10 == 0:
                data_iter.write(str(post_fix))

        train_loss = train_loss / len(data_iter)
        # self.writer.add_scalar('train_loss', train_loss, global_step=epoch)

        valid_loss = self.validate()
        # self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)

        if epoch >= self.warmup_epochs:
            self.optim_schedule.step()
        # self.writer.add_scalar('cosine_lr_decay', self.optim_schedule.get_lr()[0], global_step=epoch)

        print(
            "EP%d, train_loss=%.5f, validate_loss=%.5f"
            % (epoch, train_loss, valid_loss)
        )
        return train_loss, valid_loss

    def validate(self):
        self.model.eval()

        valid_loss = 0.0
        counter = 0
        
        for data in self.valid_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                mask_prediction = self.model(
                    data["bert_input"].float(),
                    data["timestamp"].long(),
                    data["bert_mask"].long(),
                )

                loss = self.criterion(mask_prediction, data["bert_target"].float())

            mask = data["loss_mask"].unsqueeze(-1)
            loss = (loss * mask.float()).sum() / mask.sum()

            valid_loss += loss.item()
            counter += 1

        valid_loss /= counter

        return valid_loss

    def test(self, data_loader):
        self.model.eval()

        valid_loss = 0.0
        counter = 0
        y_true_list = []
        y_pred_list = []
        
        for data in tqdm(data_loader, miniters=1, unit="test"):
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                mask_prediction = self.model(
                    data["bert_input"].float(),
                    data["timestamp"].long(),
                    data["bert_mask"].long(),
                )

                loss = self.criterion(mask_prediction, data["bert_target"].float())

            mask = data["loss_mask"].unsqueeze(-1)
            loss = (loss * mask.float()).sum() / mask.sum()

            valid_loss += loss.item()
            counter += 1
            
            # Coletar valores reais e preditos para avaliação
            y_true_list += data["bert_target"].cpu().numpy().tolist()
            y_pred_list += mask_prediction.cpu().numpy().tolist()

        valid_loss /= counter

        # Avaliar as predições após o loop
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        mae, r2 = evaluate_predictions(y_true, y_pred)

        print(f"Validation Loss: {valid_loss:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")
        return valid_loss


    def save(self, epoch, path):
        if not os.path.exists(path):
            os.makedirs(path)

        output_path = os.path.join(path, "checkpoint.tar")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            output_path,
        )

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, path):
        input_path = os.path.join(path, "checkpoint.tar")

        try:
            checkpoint = torch.load(input_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            self.model.train()

            print("EP:%d Model loaded from:" % epoch, input_path)
            return input_path
        except IOError:
            print("Error: parameter file does not exist!")

    # Work on predict
    def predict(self, data_loader):
        self.model.eval()
        y_true_list = []
        y_pred_list = []

        with torch.inference_mode():
            for data in tqdm(data_loader, desc="Predicting..."):
                data = {key: value.to(self.device) for key, value in data.items()}

                result = self.model(
                    data["bert_input"].float(),
                    data["timestamp"].long(),
                    data["bert_mask"].long(),
                )

                # Coletar predições e valores reais
                y_pred_list += result.cpu().numpy().tolist()
                y_true_list += data["bert_target"].cpu().numpy().tolist()

        # Avaliar as predições
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        mae, r2 = evaluate_predictions(y_true, y_pred)

        return np.array(y_pred), mae, r2
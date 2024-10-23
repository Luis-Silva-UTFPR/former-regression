import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from model import BERT, BERTPrediction
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm

def evaluate_predictions(y_true, y_pred):
    """Avalia as predições utilizando MAE e R²"""
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

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.warmup_epochs = warmup_epochs
        self.optim_schedule = lr_scheduler.ExponentialLR(self.optim, gamma=decay_gamma)
        self.gradient_clippling = gradient_clipping_value
        self.criterion = nn.MSELoss(reduction="none")

        if with_cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs for model pre-training")
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            torch.backends.cudnn.benchmark = True

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        number_parameters = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Total Parameters: {number_parameters:.2f} M")

    def train(self, epoch):
        self.model.train()

        data_iter = tqdm(
            enumerate(self.train_loader),
            desc=f"EP_{epoch}:train",
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

        train_loss /= len(data_iter)
        valid_loss = self.validate()
        if epoch >= self.warmup_epochs:
            self.optim_schedule.step()

        print(f"EP{epoch}, train_loss={train_loss:.5f}, validate_loss={valid_loss:.5f}")
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
            y_true_list += data["bert_target"].cpu().numpy().tolist()
            y_pred_list += mask_prediction.cpu().numpy().tolist()

        valid_loss /= len(data_loader)

        # Avaliar predições após o loop
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        mae, r2 = evaluate_predictions(y_true, y_pred)
        return valid_loss, mae, r2

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

        print(f"EP:{epoch} Model Saved on:", output_path)
        return output_path

    def load(self, path):
        input_path = os.path.join(path, "checkpoint.tar")
        try:
            checkpoint = torch.load(input_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            self.model.train()
            print(f"EP:{epoch} Model loaded from:", input_path)
            return input_path
        except IOError:
            print("Error: parameter file does not exist!")

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

                y_pred_list += result.cpu().numpy().tolist()
                y_true_list += data["bert_target"].cpu().numpy().tolist()

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        mae, r2 = evaluate_predictions(y_true, y_pred)
        return np.array(y_pred), mae, r2

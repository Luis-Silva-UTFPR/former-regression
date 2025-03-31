import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import BERT, BERTRegression
from tqdm.auto import tqdm
from torch_optimizer import AdaBelief, Lamb
from torch.optim import AdamW, Adam, SGD, Adamax, RMSprop, Adagrad, Adadelta, ASGD, LBFGS, NAdam, Rprop

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true))
        return loss
    

class MAPELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8)))


class BERTFineTuner:
    def __init__(
        self,
        bert: BERT,
        num_classes: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion="MAELoss",
        lr: float = 5e-5,
        weight_decay=0,
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

        # # Funciona bem 0.34 0.08 0.08
        for param in self.bert.parameters():
            param.requires_grad = True
        # for param in self.bert.embedding.parameters():
        #     param.requires_grad = False

        self.model = BERTRegression(bert).to(self.device)

        params = list(self.model.parameters()) + list(self.bert.parameters())
        self.num_classes = num_classes

        self.lr = lr
        self.best_loss = float("inf")

        if with_cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs for model pre-training")
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            torch.backends.cudnn.benchmark = True

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.weight_decay = weight_decay
        betas = (0.9, 0.999)  # Usado em otimizadores que exigem beta1 e beta2


        # self.optim = Adam(params, lr=lr, betas=betas, weight_decay=weight_decay) # -0.0 0.07 0.04
        # self.optim = AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay) # -0.06 0.06 0.02
        # self.optim = SGD(params, lr=lr, weight_decay=weight_decay) # xxxx
        # self.optim = RMSprop(params, lr=lr, weight_decay=weight_decay) # xxxx
        # self.optim = Adamax(params, lr=lr, betas=betas, weight_decay=weight_decay) # -0.06 0.04 0.02
        # self.optim = ASGD(params, lr=lr, weight_decay=weight_decay) # xxxx
        # self.optim = NAdam(params, lr=lr, betas=betas, weight_decay=weight_decay) # -0.04 0.05 0.01
        # self.optim = Rprop(params, lr=lr) # overfitting 0.24 0.09 -0.12
        self.optim = AdaBelief(params, lr=lr, betas=betas, weight_decay=weight_decay) # 0.14 0.10 0.07
        # self.optim = Lamb(params, lr=lr, betas=betas, weight_decay=weight_decay) # -0.38 -0.00 -0.00

        self.train_maes = []
        self.valid_maes = []
        self.train_mses = []
        self.valid_mses = []
        self.train_r2s = []
        self.valid_r2s = []


        if criterion == "MSELoss": # 0.40 0.20 0.05
            self.criterion = nn.MSELoss()
        elif criterion == "MAELoss": # 0.38 0.14 0.12
            self.criterion = nn.L1Loss()
        elif criterion == "RMSELoss": # 0.40 0.19 0.0
            self.criterion = RMSELoss()
        elif criterion == "HuberLoss": # 0.39 0.14 0.08
            self.criterion = nn.HuberLoss(delta=1.0)
        elif criterion == "SmoothL1Loss": # 0.42 0.20 0.09
            self.criterion = nn.SmoothL1Loss(beta=1.0)
        elif criterion == "QuantileLoss": # 0.37 0.12 0.02
            self.criterion = QuantileLoss(quantile=0.5)
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")


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
        train_loss = 0.0
        counter = 0

        all_preds = []
        all_labels = []

        for data in self.train_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            predict = self.model(
                data["bert_input"].float(),
                data["timestamp"].long(),
                data["bert_mask"].long()
                # data["area_ha"].squeeze().float(),
                # data["ndvi"],
                # data["ndwi"],
            )
            target = data["class_label"].squeeze(-1).float()
            loss = self.criterion(predict, target)

            if torch.isnan(loss):  # PROBLEM!
                print("Detected NaN in loss!")
                print(f"Labels: {data['class_label'].squeeze().cpu()}")
                print(f"Predictions: {predict.cpu()}")
                exit(1)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            train_loss += loss.item()

            preds = predict.detach().cpu().numpy()
            labels = data["class_label"].detach().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

            counter += 1
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        train_r2 = r2_score(all_labels, all_preds)
        train_mae = mean_absolute_error(all_labels, all_preds)
        train_mse = mean_squared_error(all_labels, all_preds)

        self.train_r2s.append(train_r2)
        self.train_maes.append(train_mae)
        self.train_mses.append(train_mse)

        train_loss /= counter
        valid_loss, metrics, y_p, y = self.validate()

        self.valid_maes.append(metrics["MAE"])
        self.valid_mses.append(metrics["MSE"])
        self.valid_r2s.append(metrics["R2"])


        print(
            "EP%d, Train Loss: %.4f, Train R2: %.4f, Valid Loss: %.4f, Valid MAE: %.4f, Valid MSE: %.4f, Valid R2: %.4f"
            % (epoch, train_loss, train_r2, valid_loss, metrics["MAE"], metrics["MSE"], metrics["R2"])
        )
        return train_loss, valid_loss, metrics, y_p, y, train_r2, train_mae

    def validate(self):
        self.model.eval()

        valid_loss = 0.0
        counter = 0
        y_pred = []
        y_true = []

        for data in self.valid_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                y_p = self.model(
                    data["bert_input"].float(),
                    data["timestamp"].long(),
                    data["bert_mask"].long()
                    # data["area_ha"].squeeze().float(),
                    # data["ndvi"],
                    # data["ndwi"],
                )

                y = data["class_label"].view(-1).float()
                loss = self.criterion(y_p, y)

            valid_loss += loss.item()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(y_p.cpu().tolist())

            counter += 1

        valid_loss /= counter
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        composite_metric = mae - r2

        return valid_loss, {"MAE": mae, "MSE": mse, "R2": r2, "COMPOSITE": composite_metric}, y_p, y

    def test(self, data_loader):
        self.model.eval()

        y_pred = []
        y_true = []
        area_ha = []

        for data in tqdm(data_loader, desc="Testing..."):
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                y_p = self.model(
                    data["bert_input"].float(),
                    data["timestamp"].long(),
                    data["bert_mask"].long()
                    # data["area_ha"].squeeze().float(),
                    # data["ndvi"],
                    # data["ndwi"],
                )

                y = data["class_label"].view(-1).float()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(y_p.cpu().tolist())
            # area_ha.extend(data["area_ha"].cpu().tolist())

        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)
        # area_ha = np.array(area_ha) 

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }, y_true, y_pred

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

    def predict(self, data_loader):
        # Coloca o modelo em modo de avaliação
        self.model.eval()
        y_preds = []

        self.model.to(self.device)


        with torch.inference_mode():  # Certifique-se de usar o modo de inferência
            for data in tqdm(data_loader, desc="Predicting..."):
                # Move os dados para o mesmo dispositivo do modelo
                data = {key: value.to(self.device) for key, value in data.items()}
                
                # Garantia de que o modelo também está no dispositivo correto
                self.model.to(self.device)

                # Passa os dados pelo modelo
                result = self.model(
                    data["bert_input"].float(),
                    data["timestamp"].long(),
                    data["bert_mask"].long()
                    # data["area_ha"].squeeze().float(),
                    # data["ndvi"],
                    # data["ndwi"],
                )
                
                # Move o resultado para CPU antes de adicionar à lista
                y_preds.extend(result.cpu().tolist())

        # Retorna o modelo ao modo de treinamento
        self.model.train()
        return np.array(y_preds)

    
    def plot_metrics(self, output_dir="SGD"):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "MAE"), exist_ok=True)
        abs_path = os.path.join(output_dir, "MAE")

        # Plot MAE
        plt.figure(figsize=(10, 6))
        plt.plot(self.valid_maes[0:], label="Valid MAE")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("Valid MAE")
        plt.legend()
        plt.grid()
        plt.savefig(f"{abs_path}/valid_mae.png")
        plt.close()

        # Plot MSE
        plt.figure(figsize=(10, 6))
        plt.plot(self.valid_mses[0:], label="Valid mse")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.title("Valid mse")
        plt.legend()
        plt.grid()
        plt.savefig(f"{abs_path}/valid_mse.png")
        plt.close()

        # Plot R2
        plt.figure(figsize=(10, 6))
        plt.plot(self.valid_r2s[0:], label="Valid R²")
        plt.xlabel("Epochs")
        plt.ylabel("R²")
        plt.title("Valid R²")
        plt.legend()
        plt.grid()
        plt.savefig(f"{abs_path}/valid_r2.png")
        plt.close()

        # Plot MAE
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_maes[0:], label="Train MAE")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("Train MAE")
        plt.legend()
        plt.grid()
        plt.savefig(f"{abs_path}/train_mae.png")
        plt.close()

        # Plot MSE
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_mses[0:], label="Train mse")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.title("Train mse")
        plt.legend()
        plt.grid()
        plt.savefig(f"{abs_path}/train_mse.png")
        plt.close()

        # Plot R2
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_r2s[0:], label="Train R²")
        plt.xlabel("Epochs")
        plt.ylabel("R²")
        plt.title("Train R²")
        plt.legend()
        plt.grid()
        plt.savefig(f"{abs_path}/train_r2.png")
        plt.close()

# stdlib
from typing import Any, List, Tuple, Union

# third party
import numpy as np
import math, sys, argparse
import pandas as pd
import torch
from torch import nn
from functools import partial
import time, os, json
from utils import NativeScaler, MAEDataset, adjust_learning_rate, get_dataset
import model_mae
from torch.utils.data import DataLoader, RandomSampler
import sys
import timm.optim.optim_factory as optim_factory
from utils import get_args_parser

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin
from sklearn.datasets import load_iris
from hyperimpute.utils.benchmarks import compare_models
from hyperimpute.plugins.imputers import Imputers
from tqdm import tqdm
eps = 1e-8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ReMasker:
    def __init__(self):
        args = get_args_parser().parse_args()

        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.norm_field_loss = args.norm_field_loss
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = 20  # originally 20
        self.model = None
        self.norm_parameters = None

        self.embed_dim = args.embed_dim
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.mlp_ratio = args.mlp_ratio
        self.max_epochs = 50
        self.mask_ratio = 0.5
        self.encode_func = args.encode_func
        self.dim = None  # Add dim as an attribute

    def save_model(self, path: str):
        """Save the model and optimizer state to the given path."""
        state = {
            'model_state_dict': self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict(),
            'norm_parameters': self.norm_parameters,
            'dim': self.dim  # Save dim
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the model and optimizer state from the given path."""
        state = torch.load(path, map_location=device)
        self.dim = state.get('dim')  # Load dim

        if self.dim is None:
            raise ValueError("Dimension 'dim' not found in the state dictionary.")


        # Initialize the model
        self.model = model_mae.MaskedAutoencoder(
            rec_len=self.dim,  # Use self.dim
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )

        # Use DataParallel for multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(device)

        # Load the state dict
        self.model.load_state_dict(state['model_state_dict'])
        self.norm_parameters = state['norm_parameters']
        self.model.to(device)
        print(f"Model loaded from {path}")

    def fit(self, X_raw: pd.DataFrame, save_path: str = None):
        X = X_raw.copy()
        self.dim = X.shape[1]  # Initialize dim
        no = len(X)

        min_val = np.zeros(self.dim)
        max_val = np.zeros(self.dim)
        eps = 1e-7

        for i in range(self.dim):
            min_val[i] = np.nanmin(X.iloc[:, i])
            max_val[i] = np.nanmax(X.iloc[:, i])
            X.iloc[:, i] = (X.iloc[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        self.norm_parameters = {"min": min_val, "max": max_val}
        np_array = X.to_numpy()
        X = torch.tensor(np_array, dtype=torch.float32)
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(device)
        X = torch.nan_to_num(X)
        X = X.to(device)

        self.model = model_mae.MaskedAutoencoder(
            rec_len=self.dim,  # Use self.dim
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )

        # Use DataParallel for multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(device)

        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:
            self.lr = self.blr * eff_batch_size / 64
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        self.model.train()

        for epoch in range(self.max_epochs):
            print(epoch)
            self.optimizer.zero_grad()
            total_loss = 0

            for iter, (samples, masks) in tqdm(enumerate(dataloader), total=len(dataloader)):
                if iter % self.accum_iter == 0:
                    adjust_learning_rate(self.optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
                                         self.max_epochs, self.warmup_epochs)

                samples = samples.unsqueeze(dim=1)
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = loss.mean()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss /= self.accum_iter

                loss_scaler(loss.mean(), self.optimizer, parameters=self.model.parameters(),
                            update_grad=(iter + 1) % self.accum_iter == 0)

                if (iter + 1) % self.accum_iter == 0:
                    self.optimizer.zero_grad()

            total_loss = (total_loss / (iter + 1)) ** 0.5
            print((epoch + 1), ',', total_loss)

        if save_path:
            self.save_model(save_path)
        return self

    def transform(self, X_raw: torch.Tensor):
        if not torch.is_tensor(X_raw):
            X_raw = torch.tensor(X_raw.values)
        X = X_raw.clone()

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        no, dim = X.shape
        X = X.cpu()

        for i in range(dim):
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)

        X = torch.from_numpy(X).to(device).float()
        M = M.to(device).float()

        self.model.eval()

        with torch.no_grad():
            for i in range(no):
                sample = torch.reshape(X[i], (1, 1, -1))
                mask = torch.reshape(M[i], (1, -1))
                _, pred, _, _ = self.model(sample, mask)
                pred = pred.squeeze(dim=2)
                if i == 0:
                    imputed_data = pred
                else:
                    imputed_data = torch.cat((imputed_data, pred), 0)

        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()
        return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.tensor(X.values, dtype=torch.float32)
        return self.fit(X).transform(X).detach().cpu().numpy()
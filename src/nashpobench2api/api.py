import json
import random
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

CATEGORICAL_FEATURES = ["optimizer"]
FEATURES = [
    "cellcode_0",
    "cellcode_1",
    "cellcode_2",
    "cellcode_3",
    "cellcode_4",
    "cellcode_5",
    "optimizer",
    "batch_size",
    "initial_lr",
    "weight_decay",
    "randomflip",
    "momentum",
    "epoch",
]

LOG_TRANSFORM_FEATURES = ["initial_lr", "weight_decay", "batch_size"]


class RankingMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout: float = 0.1,
        batch_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class API:
    def __init__(
        self,
        log_dir="train_log",
        model_dir=None,
        inference_log_dir=None,
        device_name="raspi-cpu",
        need_inference_time=True,
        mode="valid",
        metric="acc",
        iepoch=200,
        epoch=200,
        inference_device="auto",
    ):
        base_path = Path(__file__).parent
        if model_dir is None:
            model_dir = base_path / ".apidata"
        if inference_log_dir is None:
            inference_log_dir = base_path / "onnx_log"

        self.log_dir = log_dir
        self.model_dir = Path(model_dir)
        self.inference_log_dir = inference_log_dir
        self.device_name = device_name
        self.need_inference_time = need_inference_time
        self.mode = mode
        self.metric = metric
        self.iepoch = iepoch
        self.epoch = epoch

        if inference_device == "auto":
            self.inference_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.inference_device = inference_device

        self.valid_keys = [
            "initial_lr",
            "weight_decay",
            "momentum",
            "params",
            "optimizer",
            "batch_size",
            "randomflip",
        ] + [f"cellcode_{i}" for i in range(6)]
        self.categorical_keys = ["optimizer"] + [f"cellcode_{i}" for i in range(6)]

        self._load_models()

        if self.need_inference_time:
            self._load_inference_data()

        self.INDEX4DATA = 4
        self._init_logdata()

    def _init_logdata(self):
        ## acc (validation)
        self.acc_trans = []
        self.best_acc = -1
        self.best_acc_trans = []
        ## test acc
        self.test_acc_trans = []
        self.best_test_acc = -1
        self.best_test_acc_trans = []
        ## inference time
        if self.need_inference_time:
            self.inftime_trans = []
            self.best_inftime = 1e10
            self.best_inftime_trans = []
        ## cost
        self.total_cost = 0.0
        self.total_cost_trans = []
        self.best_cost = None
        ## key
        self.key_trans = []
        self.best_key = None
        self.best_key_trans = []
        self._config_budget_cache = {}
        return

    def _write_log(
        self,
        acc: float,
        test_acc: float,
        inftime: float,
        cost: float,
        key: dict,
    ):
        self.acc_trans.append(acc)
        self.test_acc_trans.append(test_acc)
        self.key_trans.append(key)
        self.total_cost += cost
        self.total_cost_trans.append(self.total_cost)
        if self.need_inference_time:
            self.inftime_trans.append(inftime)

        if self.need_inference_time:
            if self.best_key is None:
                (
                    self.best_acc,
                    self.best_test_acc,
                    self.best_inftime,
                    self.best_cost,
                    self.best_key,
                ) = acc, test_acc, inftime, cost, key
            else:
                dominates = (
                    acc >= self.best_acc
                    and (inftime is not None)
                    and self.best_inftime is not None
                    and inftime <= self.best_inftime
                ) and (acc > self.best_acc or inftime < self.best_inftime)
                if dominates:
                    (
                        self.best_acc,
                        self.best_test_acc,
                        self.best_inftime,
                        self.best_cost,
                        self.best_key,
                    ) = acc, test_acc, inftime, cost, key
        else:
            if self.best_key is None or self.best_acc < acc:
                self.best_acc, self.best_test_acc, self.best_cost, self.best_key = (
                    acc,
                    test_acc,
                    cost,
                    key,
                )
        self.best_acc_trans.append(self.best_acc)
        self.best_test_acc_trans.append(self.best_test_acc)
        self.best_key_trans.append(self.best_key)
        if self.need_inference_time:
            self.best_inftime_trans.append(self.best_inftime)
        return

    def reset_logdata(self):
        self._init_logdata()
        self.clear_budget_cache()
        return

    def get_results(self, epoch: Union[int, str] = "both", mode: str = "test"):
        if epoch == "both":
            epochs = [12, 200]
        else:
            epochs = [int(epoch)]

        self.final_accs = []
        self.final_inftimes = []
        for e in epochs:
            if self.need_inference_time:
                final_acc, inference_time, _ = self.query_by_key(
                    epoch=e, mode=mode, enable_log=False, **self.best_key
                )
                self.final_accs.append(final_acc)
                self.final_inftimes.append(inference_time)
            else:
                final_acc, _ = self.query_by_key(
                    epoch=e, mode=mode, enable_log=False, **self.best_key
                )
                self.final_accs.append(final_acc)

        nlist = [
            "acc_trans",
            "test_acc_trans",
            "key_trans",
            "best_acc_trans",
            "best_test_acc_trans",
            "best_key_trans",
            "total_cost_trans",
            "final_accs",
        ]
        if self.need_inference_time:
            nlist += [
                "inftime_trans",
                "best_inftime",
                "best_inftime_trans",
                "final_inftimes",
            ]

        return {n: eval("self." + n, {"self": self}) for n in nlist}

    def get_random_key(self):
        cellcode = [random.randint(0, 3) for _ in range(6)]

        initial_lr = 10 ** random.uniform(np.log10(0.003125), np.log10(0.4))
        weight_decay = 10 ** random.uniform(np.log10(1e-4), np.log10(5e-3))
        batch_size = 2 ** random.randint(4, 9)
        key = {
            "initial_lr": float(initial_lr),
            "weight_decay": float(weight_decay),
            "momentum": random.uniform(0.7, 0.9),
            "optimizer": random.sample(["SGD", "RMSprop", "Adam"], 1)[0],
            "batch_size": batch_size,
            "randomflip": random.uniform(0.25, 0.75),
            "cellcode_0": cellcode[0],
            "cellcode_1": cellcode[1],
            "cellcode_2": cellcode[2],
            "cellcode_3": cellcode[3],
            "cellcode_4": cellcode[4],
            "cellcode_5": cellcode[5],
        }
        return key

    def get_total_cost(self):
        return self.total_cost

    def _load_models(self):
        self.models = {}
        self.model_configs = {}
        self.xgb_models = {}
        self.xgb_configs = {}

        for epoch in [12, 200]:
            config_path = self.model_dir / f"mlp_config_epoch{epoch}.json"
            if not config_path.exists():
                raise FileNotFoundError(f"MLP config file not found: {config_path}")

            with open(config_path, "r") as f:
                self.model_configs[epoch] = json.load(f)

            xgb_config_path = (
                self.model_dir / f"xgboost_total_time_epoch{epoch}_config.json"
            )
            if not xgb_config_path.exists():
                raise FileNotFoundError(
                    f"XGBoost config file not found: {xgb_config_path}"
                )

            with open(xgb_config_path, "r") as f:
                self.xgb_configs[epoch] = json.load(f)

            self.models[epoch] = {}

            for target in ["valid_acc", "test_acc"]:
                if target not in self.model_configs[epoch]:
                    continue

                config = self.model_configs[epoch][target]
                weights_path = (
                    self.model_dir / f"mlp_ranking_epoch{epoch}_{target}_weights.pt"
                )

                if not weights_path.exists():
                    raise FileNotFoundError(
                        f"MLP weights file not found: {weights_path}"
                    )

                model = RankingMLP(
                    input_dim=config["input_dim"],
                    hidden_dims=config["hidden_dims"],
                    dropout=config["dropout"],
                    batch_norm=config["batch_norm"],
                    activation=config["activation"],
                )

                state_dict = torch.load(
                    weights_path, map_location=self.inference_device, weights_only=True
                )
                model.load_state_dict(state_dict)
                model.to(self.inference_device)
                model.eval()

                self.models[epoch][target] = model

            xgb_model_path = self.model_dir / f"xgboost_total_time_epoch{epoch}.json"
            if not xgb_model_path.exists():
                raise FileNotFoundError(
                    f"XGBoost model file not found: {xgb_model_path}"
                )

            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(xgb_model_path)
            self.xgb_models[epoch] = xgb_model

        return

    def _load_inference_data(self):
        base_path = Path(__file__).parent
        inference_data_path = base_path / ".apidata" / "onnx_combined_data.parquet"

        if not inference_data_path.exists():
            raise FileNotFoundError(
                f"Inference data file not found: {inference_data_path}"
            )

        self.inference_data = pd.read_parquet(inference_data_path)
        return

    def _predict_mlp(self, X: pd.DataFrame, epoch: int, target: str) -> float:
        config = self.model_configs[epoch][target]
        model = self.models[epoch][target]
        scalers = config["scalers"]
        output_range = config["output_range"]

        x_array = X.values.astype(np.float32)
        x_mean = np.array(scalers["x_mean"], dtype=np.float32)
        x_std = np.array(scalers["x_std"], dtype=np.float32)
        x_scaled = (x_array - x_mean) / x_std

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.inference_device)

        with torch.no_grad():
            y_scaled = model(x_tensor).cpu().numpy()

        y_mean = scalers["y_mean"]
        y_std = scalers["y_std"]
        y_pred = y_scaled * y_std + y_mean

        if output_range is not None:
            y_pred = np.clip(y_pred, output_range[0], output_range[1])

        return float(y_pred[0])

    def _predict_xgboost(self, X: pd.DataFrame, epoch: int) -> float:
        model = self.xgb_models[epoch]
        config = self.xgb_configs[epoch]
        scalers = config["scalers"]

        feature_names = config.get("feature_columns", X.columns.tolist())
        X_reordered = X.reindex(columns=feature_names, fill_value=0)

        x_array = X_reordered.values.astype(np.float32)
        x_mean = np.array(scalers["x_mean"], dtype=np.float32)
        x_std = np.array(scalers["x_std"], dtype=np.float32)
        x_scaled = (x_array - x_mean) / x_std

        y_scaled = model.predict(x_scaled)

        y_mean = scalers["y_mean"]
        y_std = scalers["y_std"]
        y_pred = y_scaled * y_std + y_mean

        return float(y_pred[0])

    def preprocess_features(self, df):
        df = df.copy()
        df.loc[df["optimizer"] == "Adam", "momentum"] = 0.0
        X = df[FEATURES].copy()

        for col in [c for c in X.columns if c.startswith("cellcode_")]:
            if X[col].dtype == "object":
                X[col] = X[col].astype(int)

        for col in LOG_TRANSFORM_FEATURES:
            if col in X.columns:
                X[col] = np.log10(X[col])

        optimizer_values = ["SGD", "RMSprop", "Adam"]
        optimizer_dummies = pd.get_dummies(X["optimizer"], prefix="optimizer")
        optimizer_dummies = optimizer_dummies.reindex(
            columns=[f"optimizer_{opt}" for opt in optimizer_values], fill_value=0
        )
        X = pd.concat([X.drop(columns=["optimizer"]), optimizer_dummies], axis=1)

        ordered_cols = [
            "cellcode_0",
            "cellcode_1",
            "cellcode_2",
            "cellcode_3",
            "cellcode_4",
            "cellcode_5",
            "batch_size",
            "initial_lr",
            "weight_decay",
            "randomflip",
            "momentum",
            "epoch",
            "optimizer_Adam",
            "optimizer_RMSprop",
            "optimizer_SGD",
        ]
        X = X.reindex(columns=ordered_cols, fill_value=0)
        return X

    def get_inference_time(self, cellcode):
        filtered_data = self.inference_data[
            (self.inference_data["cellcode"] == cellcode)
            & (self.inference_data["device"] == self.device_name)
        ]

        if filtered_data.empty:
            raise FileNotFoundError(
                f"No inference data found for cellcode: {cellcode} "
                f"with device: {self.device_name}"
            )

        return float(filtered_data["trial_mean"].iloc[0])

    def query_by_key(self, epoch=12, iepoch=None, mode="valid", enable_log=True, **key):
        if (iepoch is None) and (epoch is not None):
            iepoch = epoch

        df = pd.DataFrame([key | {"epoch": iepoch - 1}])
        X = self.preprocess_features(df)
        X_time = X.copy()
        X_testacc = X.copy()
        X_time["epoch"] = 0
        X_testacc["epoch"] = epoch - 1

        pred_acc = self._predict_mlp(X, epoch, f"{mode}_acc")

        pred_cost = self._predict_xgboost(X_time, epoch)
        if iepoch != epoch:
            pred_cost = pred_cost * iepoch / epoch

        if enable_log:
            pred_test_acc = self._predict_mlp(X_testacc, epoch, "test_acc")
        else:
            pred_test_acc = 0.0

        inference_time = None
        if self.need_inference_time:
            l_cell = [str(key[f"cellcode_{i}"]) for i in range(6)]
            cellcode = (
                l_cell[0]
                + "|"
                + l_cell[1]
                + l_cell[2]
                + "|"
                + l_cell[3]
                + l_cell[4]
                + l_cell[5]
            )

            inference_time = self.get_inference_time(cellcode)

        if enable_log:
            self._write_log(pred_acc, pred_test_acc, inference_time, pred_cost, key)

        if self.need_inference_time:
            return pred_acc, inference_time, pred_cost

        return pred_acc, pred_cost

    def _key_to_tuple(self, key: dict) -> tuple:
        sorted_items = sorted(key.items())
        return tuple((k, v) for k, v in sorted_items)

    def query_by_key_incremental(
        self, epoch=12, iepoch=None, mode="valid", enable_log=True, **key
    ):
        if (iepoch is None) and (epoch is not None):
            iepoch = epoch

        key_tuple = self._key_to_tuple(key)

        last_iepoch = self._config_budget_cache.get(key_tuple, 0)

        df = pd.DataFrame([key | {"epoch": iepoch - 1}])
        X = self.preprocess_features(df)
        X_time = X.copy()
        X_testacc = X.copy()
        X_time["epoch"] = 0
        X_testacc["epoch"] = epoch - 1

        pred_acc = self._predict_mlp(X, epoch, f"{mode}_acc")

        full_cost = self._predict_xgboost(X_time, epoch)

        cost_to_current = full_cost * iepoch / epoch
        cost_to_previous = full_cost * last_iepoch / epoch
        incremental_cost = max(0.0, cost_to_current - cost_to_previous)

        self._config_budget_cache[key_tuple] = iepoch

        pred_test_acc = 0.0
        if enable_log:
            pred_test_acc = self._predict_mlp(X_testacc, epoch, "test_acc")

        inference_time = None
        if self.need_inference_time:
            l_cell = [str(key[f"cellcode_{i}"]) for i in range(6)]
            cellcode = (
                l_cell[0]
                + "|"
                + l_cell[1]
                + l_cell[2]
                + "|"
                + l_cell[3]
                + l_cell[4]
                + l_cell[5]
            )
            inference_time = self.get_inference_time(cellcode)

        if enable_log:
            self._write_log(
                pred_acc, pred_test_acc, inference_time, incremental_cost, key
            )

        if self.need_inference_time:
            return pred_acc, inference_time, incremental_cost

        return pred_acc, incremental_cost

    def clear_budget_cache(self):
        self._config_budget_cache = {}

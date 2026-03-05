import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_input_matrix(pa: float, pe: float, va: float, ve: float, step: float) -> np.ndarray:

    # Number of compositions
    # MATLAB defines n=8 and then uses features(:,9)=p*v, so total columns = 9
    cmps = [
        np.arange(0.5, 1.0 + 1e-9, 0.05),     # Matrix
        np.arange(0.0, 0.2 + 1e-9, 0.01),     # other fillers
        np.arange(0.0, 0.1 + 1e-9, 0.02),
        np.arange(0.0, 0.1 + 1e-9, 0.01),
        np.arange(0.0, 0.1 + 1e-9, 0.02),
        np.arange(0.0, 0.1 + 1e-9, 0.02),
        np.arange(pa, pe + 1e-9, step),      # surface pressure
        np.arange(va, ve + 1e-9, step),      # slide speed
    ]
    grids = np.meshgrid(*cmps, indexing="ij")
    features = np.stack([g.reshape(-1) for g in grids], axis=1)  # (N, 8)

    # delete impossible permutations: sum(features(:,1:6),2) ~= 1
    s = features[:, :6].sum(axis=1)
    features = features[np.isclose(s, 1.0), :]

    pv = (features[:, 6] * features[:, 7]).reshape(-1, 1)
    features = np.concatenate([features, pv], axis=1)  # (N, 9)
    return features.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_layers: list[int], activation: str, positive_mode: str):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        act_cls = acts.get(activation.lower(), nn.Tanh)

        layers = []
        prev = in_dim
        # hidden layers include possibly a "1" in your MATLAB hiddenLayerSize, but in PyTorch
        # we treat the last layer as output layer separately. We'll mirror your structure by
        # using all entries except the last as hidden, and last as output dim (typically 1).
        for h in hidden_layers[:-1]:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(prev, hidden_layers[-1])

        self.positive_mode = positive_mode.lower()
        if self.positive_mode == "softplus":
            self.pos = nn.Softplus()
        else:
            self.pos = None

    def forward(self, x):
        z = self.backbone(x)
        y = self.out(z)
        if self.pos is not None:
            y = self.pos(y)
        return y


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_data_from_xlsx(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xlsx_path = cfg["data"]["xlsx_path"]
    sheet = cfg["data"]["sheet_name"]
    target_name = cfg["data"]["target_column_name"]

    df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
    # MATLAB used xlsread returning Tr_data numeric and Tr_title; here we use header row
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found. Available columns: {list(df.columns)[:30]} ...")

    # MATLAB x = Tr_data(:,1:9) => here we take first 9 columns by position (excluding header)
    # But safer: take by position exactly 1..9 (1-based) => 0..8 (0-based)
    start = cfg["data"]["input_cols"]["start"] - 1
    end = cfg["data"]["input_cols"]["end"]        # end is inclusive in MATLAB, exclusive in python slicing
    x = df.iloc[:, start:end].to_numpy(dtype=np.float32)
    y = df[target_name].to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y


def split_data(x: np.ndarray, y: np.ndarray, cfg: Dict[str, Any], seed: int) -> SplitData:
    tr = cfg["split"]["train_ratio"]
    va = cfg["split"]["val_ratio"]
    te = cfg["split"]["test_ratio"]
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=(1 - tr), random_state=seed, shuffle=True)
    # val and test split from tmp
    val_frac_of_tmp = va / (va + te)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=(1 - val_frac_of_tmp),
                                                    random_state=seed + 1, shuffle=True)
    return SplitData(x_train, y_train, x_val, y_val, x_test, y_test)


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def mse_both(y_true_orig: np.ndarray,
             y_pred_orig: np.ndarray,
             y_scaler: MinMaxScaler) -> tuple[float, float]:
    """
    Return (mse_orig, mse_scaled) using the SAME samples.
    """
    mse_orig = mse_np(y_true_orig, y_pred_orig)
    y_true_s = y_scaler.transform(y_true_orig)
    y_pred_s = y_scaler.transform(y_pred_orig)
    mse_scaled = mse_np(y_true_s, y_pred_s)
    return mse_orig, mse_scaled

def mre_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # MATLAB: abs((di-oi)/di), then mean omitnan
    di = y_true.reshape(-1)
    oi = y_pred.reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        re = np.abs((di - oi) / di)
    return float(np.nanmean(re))


def train_one(model: nn.Module,
              x_train_t: torch.Tensor, y_train_t: torch.Tensor,
              x_val_t: torch.Tensor, y_val_t: torch.Tensor,
              cfg: Dict[str, Any],
              device: torch.device) -> Tuple[nn.Module, Dict[str, float]]:
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    target_mse = float(cfg["train"]["target_mse"])

    criterion = nn.MSELoss()

    opt_name = cfg["train"]["optimizer"].lower()
    if opt_name == "lbfgs":
        lbfgs_max_iter = int(cfg["train"].get("lbfgs_max_iter", 20))
        lbfgs_history = int(cfg["train"].get("lbfgs_history_size", 10))
        lbfgs_line_search = cfg["train"].get("lbfgs_line_search", "strong_wolfe")
        lbfgs_line_search = None if (lbfgs_line_search is None or str(lbfgs_line_search).strip() == "") else str(
            lbfgs_line_search)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history,
            line_search_fn=lbfgs_line_search,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_losses = []
    val_losses = []
    best_epoch = 0

    # early stop
    patience = int(cfg["train"].get("patience", 50))
    no_improve = 0

    # print frequency
    log_every = int(cfg["train"].get("log_every", 20))

    best_val = float("inf")
    best_state = None

    if opt_name == "lbfgs":
        total_steps = int(cfg["train"].get("lbfgs_steps", epochs))
    else:
        total_steps = epochs

    for ep in range(1, total_steps + 1):
        model.train()

        if opt_name == "lbfgs":
            # IMPORTANT: closure must be deterministic and should not iterate over a DataLoader.
            # We use full-batch tensors (already prepared) for stable LBFGS behavior.
            def closure():
                optimizer.zero_grad(set_to_none=True)
                pred = model(x_train_t)
                loss_t = criterion(pred, y_train_t)
                loss_t.backward()
                return loss_t

            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

            # record loss curve
            # train loss：Adam -- loss.item()；LBFGS -- tensor/float
            try:
                train_loss_value = float(loss.item())
            except Exception:
                train_loss_value = float(loss)
            train_losses.append(train_loss_value)
            val_losses.append(float(val_loss))

            # progress print in terminal
            # if ep == 1 or ep % log_every == 0 or ep == total_steps:
            #     print(
            #         f"[Epoch {ep:4d}/{epochs}] "
            #         f"train_loss={train_loss_value:.4e}  val_loss={val_loss:.4e}  best_val={best_val:.4e}",
            #         flush=True
            #     )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # early stop if reached target mse
        if best_val <= target_mse:
            print(f"[Stop] Reached target_mse at epoch {ep} (best_val={best_val:.4e})", flush=True)
            break

        # early stop
        if no_improve >= patience:
            print(f"[Stop] Early stopping at epoch {ep} (best_epoch={best_epoch}, best_val={best_val:.4e})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
    }
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    set_seed(int(cfg["project"]["seed"]))

    tag = cfg["project"]["save_tag"]
    loops = int(cfg["project"]["loops"])

    device_cfg = cfg["train"]["device"].lower()
    if device_cfg == "cuda":
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_data_from_xlsx(cfg)

    # prepare output dir (similar to MATLAB Saving_path)
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(cfg["project"]["output_dir"]) / f"{ts}_{tag}_{loops}loops_{len(cfg['model']['hidden_layers'])}layers"
    nets_dir = run_dir / "nets"
    nets_dir.mkdir(parents=True, exist_ok=True)

    # Build grid for positivity check (MATLAB BuildInputMatrix)
    grid_cfg = cfg["validity_check"]["grid"]
    grid_X = build_input_matrix(grid_cfg["pa"], grid_cfg["pe"], grid_cfg["va"], grid_cfg["ve"], grid_cfg["step"])

    target_mse = float(cfg["train"]["target_mse"])
    valid_models = []
    ttl = 0  # total attempts including invalid

    # vectors like MATLAB
    records = []

    z = 1
    attempt_seed_base = int(cfg["project"]["seed"])
    max_attempts_total = 200
    while z <= loops and ttl < max_attempts_total:
    # while z <= loops:
        ttl += 1
        split_seed = attempt_seed_base + ttl * 101  # deterministic but different each attempt

        sd = split_data(X, y, cfg, seed=split_seed)

        # preprocess: minmax(-1,1)
        x_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train_s = x_scaler.fit_transform(sd.X_train)
        y_train_s = y_scaler.fit_transform(sd.y_train)
        x_val_s = x_scaler.transform(sd.X_val)
        y_val_s = y_scaler.transform(sd.y_val)
        x_test_s = x_scaler.transform(sd.X_test)
        y_test_s = y_scaler.transform(sd.y_test)

        # tensors
        x_train_t = torch.from_numpy(x_train_s).to(device)
        y_train_t = torch.from_numpy(y_train_s).to(device)
        x_val_t = torch.from_numpy(x_val_s).to(device)
        y_val_t = torch.from_numpy(y_val_s).to(device)

        model = MLP(
            in_dim=X.shape[1],
            hidden_layers=list(cfg["model"]["hidden_layers"]),
            activation=cfg["model"]["activation"],
            positive_mode=cfg["model"]["positive_mode"] if cfg["model"]["output_positive"] else "none",
        ).to(device)

        model, info = train_one(model, x_train_t, y_train_t, x_val_t, y_val_t, cfg, device)
        best_val_mse_scaled = float(info["best_val_mse"])

        # Evaluate on all sets (inverse transform to original scale)
        model.eval()
        with torch.no_grad():
            y_train_pred_s = model(torch.from_numpy(x_train_s).to(device)).cpu().numpy()
            y_val_pred_s = model(torch.from_numpy(x_val_s).to(device)).cpu().numpy()
            y_test_pred_s = model(torch.from_numpy(x_test_s).to(device)).cpu().numpy()

        y_train_pred = y_scaler.inverse_transform(y_train_pred_s)
        y_val_pred = y_scaler.inverse_transform(y_val_pred_s)
        y_test_pred = y_scaler.inverse_transform(y_test_pred_s)

        # train_mse = mse_np(sd.y_train, y_train_pred)
        # val_mse = mse_np(sd.y_val, y_val_pred)
        # test_mse = mse_np(sd.y_test, y_test_pred)

        train_mse, train_mse_s = mse_both(sd.y_train, y_train_pred, y_scaler)
        val_mse, val_mse_s = mse_both(sd.y_val, y_val_pred, y_scaler)
        test_mse, test_mse_s = mse_both(sd.y_test, y_test_pred, y_scaler)

        print(f"[Attempt {ttl}] MSE(orig): train={train_mse:.6g}, val={val_mse:.6g}, test={test_mse:.6g}")
        print(f"[Attempt {ttl}] MSE(scl ): train={train_mse_s:.6g}, val={val_mse_s:.6g}, test={test_mse_s:.6g}")

        # overall outputs like MATLAB outputs = net(inputs) (here all X)
        with torch.no_grad():
            y_all_pred_s = model(torch.from_numpy(x_scaler.transform(X)).to(device)).cpu().numpy()
        y_all_pred = y_scaler.inverse_transform(y_all_pred_s)
        overall_mse = mse_np(y, y_all_pred)
        mre = mre_np(y, y_all_pred)

        # positivity check on grid
        positive_ok = True
        grid_min = float("nan")
        grid_max = float("nan")
        if cfg["validity_check"]["require_positive_on_grid"]:
            with torch.no_grad():
                grid_pred_s = model(torch.from_numpy(x_scaler.transform(grid_X)).to(device)).cpu().numpy()
            grid_pred = y_scaler.inverse_transform(grid_pred_s)
            grid_min = float(np.min(grid_pred))
            grid_max = float(np.max(grid_pred))
            positive_ok = bool(np.all(grid_pred > 0))

        # validity criteria (match MATLAB: valPerformance <= targetMSE AND all(Pdt_output>0))
        if positive_ok and (best_val_mse_scaled <= target_mse):
        # if positive_ok and (best_val_mse_orig <= target_mse):       # use orig mse as criteria
            # save this net
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                },
                nets_dir / f"{z}.pt",
            )

            # Save loss curve ONLY for accepted nets
            fig_path = run_dir / f"loss_curve_net_{z}.png"
            plt.figure()
            plt.plot(info["train_losses"], label="train_loss")
            plt.plot(info["val_losses"], label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss (scaled)")
            plt.title(f"Loss Curve - accepted net {z} (attempt {ttl}, best_epoch={info.get('best_epoch', '-')})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

            records.append({
                "z": z,
                "ttl": ttl,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "overall_mse": overall_mse,
                "mre": mre,
                "positive_ok": positive_ok,
            })
            z += 1
            print(
                f"[ACCEPT] ttl={ttl} z={z} best_val_scaled={best_val_mse_scaled:.4e} "
                f"val_mse_orig={val_mse:.4e} positive={positive_ok} grid_min={grid_min:.4e} ",flush=True)
            print("", flush=True)
        else:
            # invalid -> continue training new attempt without increasing z (like MATLAB)
            records.append({
                "z": None,
                "ttl": ttl,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "overall_mse": overall_mse,
                "mre": mre,
                "positive_ok": positive_ok,
                "invalid": True,
            })
            print(f"[REJECT] ttl={ttl} best_val_scaled={best_val_mse_scaled:.4e} "
                  f"val_mse_orig={val_mse:.4e} positive={positive_ok} grid_min={grid_min:.4e} ", flush=True)
            print("", flush=True)
            continue

    if z <= loops:
        print(f"[WARN] Only collected {z - 1}/{loops} valid models within {max_attempts_total} attempts.")

    # pick best valid model by min val_mse
    valid = [r for r in records if r.get("z") is not None]
    if len(valid) == 0:
        raise RuntimeError("No valid models found. Consider relaxing target_mse or positivity constraint.")

    best = min(valid, key=lambda r: r["val_mse"])
    best_idx = best["z"]

    # load best model and save as trained_net
    ckpt = torch.load(nets_dir / f"{best_idx}.pt", map_location="cpu")
    torch.save(ckpt, run_dir / "trained_net.pt")

    # save scalers + metrics
    meta = {
        "best_net_z": int(best_idx),
        "target": cfg["data"]["target_column_name"],
        "hidden_layers": cfg["model"]["hidden_layers"],
        "loops_valid": loops,
        "attempts_total": ttl,
        "best_metrics": best,
        "all_records": records,
    }
    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # also write an info txt similar to MATLAB
    info_txt = run_dir / "00_InfoTXT.txt"
    info_txt.write_text(
        "\n".join([
            f"Number of valid nets: {loops}",
            f"Predicted variable: {cfg['data']['target_column_name']}",
            f"Hidden layers: {cfg['model']['hidden_layers']}",
            f"Total attempts including invalid: {ttl}",
            f"Best net z: {best_idx}",
            f"Best val MSE: {best['val_mse']}",
            f"Best overall MSE: {best['overall_mse']}",
            f"Best MRE: {best['mre']}",
        ]),
        encoding="utf-8",
    )

    print(f"[OK] Run saved to: {run_dir}")
    print(f"[OK] Best net z={best_idx}, val_mse={best['val_mse']:.6g}, positive_ok={best['positive_ok']}")

if __name__ == "__main__":
    main()
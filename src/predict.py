import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
import pandas as pd


# --------- model definition must match train.py ---------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_layers: list[int], activation: str, positive_mode: str):
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh}
        act_cls = acts.get(activation.lower(), nn.Tanh)

        layers = []
        prev = in_dim
        for h in hidden_layers[:-1]:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(prev, hidden_layers[-1])

        self.positive_mode = positive_mode.lower()
        self.pos = nn.Softplus() if self.positive_mode == "softplus" else None

    def forward(self, x):
        z = self.backbone(x)
        y = self.out(z)
        if self.pos is not None:
            y = self.pos(y)
        return y


def load_trained_model(run_dir: Path, model_file: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(run_dir / model_file, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    hidden_layers = list(model_cfg.get("hidden_layers", [12, 5, 1]))
    activation = model_cfg.get("activation", "tanh")
    positive_mode = model_cfg.get("positive_mode", "softplus") if model_cfg.get("output_positive", True) else "none"

    model = MLP(in_dim=9, hidden_layers=hidden_layers, activation=activation, positive_mode=positive_mode)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def frange(start: float, end: float, step: float) -> np.ndarray:
    # inclusive end (match MATLAB style)
    n = int(round((end - start) / step)) + 1
    return (start + step * np.arange(n)).astype(np.float32)


def enumerate_materials(cfg: Dict[str, Any]) -> np.ndarray:
    g = cfg["composition_grid"]
    a = frange(g["a"]["start"], g["a"]["end"], g["a"]["step"])
    b = frange(g["b"]["start"], g["b"]["end"], g["b"]["step"])
    c = frange(g["c"]["start"], g["c"]["end"], g["c"]["step"])
    d = frange(g["d"]["start"], g["d"]["end"], g["d"]["step"])
    e = frange(g["e"]["start"], g["e"]["end"], g["e"]["step"])
    f_min = float(g["f"]["start"])
    f_max = float(g["f"]["end"])
    step36 = float(g["f_grid_step"])

    # Upper bound similar to MATLAB
    upper = len(a) * len(b) * len(c) * len(d) * len(e)
    M = np.zeros((upper, 6), dtype=np.float32)
    k = 0

    for aa in a:
        for bb in b:
            for cc in c:
                for dd in d:
                    for ee in e:
                        ff = 1.0 - float(aa + bb + cc + dd + ee)
                        if ff < f_min - 1e-12 or ff > f_max + 1e-12:
                            continue
                        # f on 0.01 grid
                        if abs(round(ff / step36) * step36 - ff) > 1e-8:
                            continue
                        M[k, :] = np.array([aa, bb, cc, dd, ee, ff], dtype=np.float32)
                        k += 1

    return M[:k, :]


def build_condition_grid(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cg = cfg["condition_grid"]
    p = frange(cg["p"]["start"], cg["p"]["end"], cg["p"]["step"])
    v = frange(cg["v"]["start"], cg["v"]["end"], cg["v"]["step"])
    PP, VV = np.meshgrid(p, v, indexing="ij")  # shape (p_pts, v_pts)
    pv = (PP.reshape(-1) * VV.reshape(-1)).astype(np.float32)
    return PP.astype(np.float32), VV.astype(np.float32), pv


def build_masks(pv: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    r = cfg["pv_regions"]
    m1 = pv <= float(r["region1"]["value"])
    m2 = pv <= float(r["region2"]["value"])
    low = float(r["region3"]["low"])
    high = float(r["region3"]["high"])
    m3 = (pv >= low) & (pv <= high)
    return {"mask1": m1, "mask2": m2, "mask3": m3}


@torch.no_grad()
def predict_one_material(model: nn.Module,
                         cmp: np.ndarray,
                         PP: np.ndarray,
                         VV: np.ndarray,
                         pv: np.ndarray,
                         minibatch: int,
                         device: torch.device) -> np.ndarray:
    m = pv.shape[0]
    # Assemble X (m x 9): [6 components, p, v, pv]
    X = np.zeros((m, 9), dtype=np.float32)
    X[:, 0:6] = cmp.reshape(1, 6).repeat(m, axis=0)
    X[:, 6] = PP.reshape(-1)
    X[:, 7] = VV.reshape(-1)
    X[:, 8] = pv

    y = np.zeros((m,), dtype=np.float32)
    idx = 0
    while idx < m:
        j = min(idx + minibatch, m)
        Xi = torch.from_numpy(X[idx:j, :]).to(device)
        yi = model(Xi).detach().cpu().numpy().reshape(-1)
        y[idx:j] = yi.astype(np.float32)
        idx = j
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    run_dir = Path(cfg["project"]["run_dir"])
    model_file = cfg["project"]["model_file"]
    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device_cfg = cfg["runtime"]["device"].lower()
    if device_cfg == "cuda":
        device = torch.device("cuda")
    elif device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    verbose = bool(cfg["runtime"].get("verbose", True))
    minibatch = int(cfg["prediction"]["minibatch_size"])
    y_name = cfg["prediction"].get("output_name", "y")

    # Load model
    model, train_cfg = load_trained_model(run_dir, model_file, device)

    # Enumerate materials
    if verbose:
        print("[1/3] Enumerating valid materials...")
    M = enumerate_materials(cfg)
    nMat = M.shape[0]
    if verbose:
        print(f"      materials: {nMat}")

    # Condition grid and masks
    PP, VV, pv = build_condition_grid(cfg)
    masks = build_masks(pv, cfg)
    m = pv.shape[0]
    if verbose:
        print(f"[2/3] Condition grid: m={m} points (p={PP.shape[0]} x v={PP.shape[1]})")

    # Main loop
    if verbose:
        print("[3/3] Predicting and computing statistics...")
    R = np.zeros((nMat, 8), dtype=np.float32)

    for i in range(nMat):
        y = predict_one_material(model, M[i, :], PP, VV, pv, minibatch, device)

        # stats as in MATLAB
        R[i, 0] = float(np.mean(y[masks["mask1"]]))
        R[i, 1] = float(np.std(y[masks["mask1"]], ddof=0))
        R[i, 2] = float(np.mean(y[masks["mask2"]]))
        R[i, 3] = float(np.std(y[masks["mask2"]], ddof=0))
        R[i, 4] = float(np.mean(y[masks["mask3"]]))
        R[i, 5] = float(np.std(y[masks["mask3"]], ddof=0))
        R[i, 6] = float(np.mean(y))
        R[i, 7] = float(np.std(y, ddof=0))

        if verbose and (i + 1) % 2000 == 0:
            print(f"      {i+1}/{nMat} done")

    # Output table: 6 composition cols + 8 stats cols
    cols_cmp = ["a_PEEK", "b_SCF", "c", "d", "e", "f"]
    cols_stats = [
        f"{y_name}_m1", f"{y_name}_s1",
        f"{y_name}_m2", f"{y_name}_s2",
        f"{y_name}_m3", f"{y_name}_s3",
        f"{y_name}_mA", f"{y_name}_sA",
    ]
    df = pd.DataFrame(np.concatenate([M, R], axis=1), columns=cols_cmp + cols_stats)

    out_file = Path(cfg["output"]["file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = cfg["output"].get("format", "xlsx").lower()
    if fmt == "csv":
        df.to_csv(out_file, index=False)
    elif fmt == "parquet":
        df.to_parquet(out_file, index=False)
    else:
        df.to_excel(out_file, index=False)

    if verbose:
        print(f"[OK] Saved: {out_file}")


if __name__ == "__main__":
    main()
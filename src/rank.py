import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p}")


def compute_cost_index(df: pd.DataFrame, comp_cols: List[str], unit_prices: List[float],
                       w_n: float, w_price: float, eps: float = 0.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    comps = df[comp_cols].to_numpy(dtype=float)
    prices = np.array(unit_prices, dtype=float).reshape(1, -1)
    total_price = (comps * prices).sum(axis=1)

    # count components: content > eps
    n_components = (comps > eps).sum(axis=1).astype(float)

    denom = n_components * w_n + total_price * w_price
    cost_index = 1.0 / denom
    return pd.Series(cost_index, index=df.index), pd.Series(total_price, index=df.index), pd.Series(n_components, index=df.index)


def compute_tribo_index(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    t = cfg["tribo_index"]
    denom = (
        df[t["cof_region2_col"]].astype(float) * float(t["w_cof_r2"]) +
        df[t["wear_region2_col"]].astype(float) * float(t["w_wear_r2"]) +
        df[t["cof_region3_col"]].astype(float) * float(t["w_cof_r3"]) +
        df[t["wear_region3_col"]].astype(float) * float(t["w_wear_r3"])
    )
    tribo_index = 1.0 / denom
    return tribo_index


def pareto_front(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """
    Maximize x and y. Returns rows on Pareto front (non-dominated set).
    Simple O(n log n) approach: sort by x desc, keep those with y increasing.
    """
    d = df.sort_values(by=x, ascending=False).reset_index(drop=False)
    best_y = -np.inf
    keep = []
    for _, row in d.iterrows():
        if row[y] > best_y:
            keep.append(True)
            best_y = row[y]
        else:
            keep.append(False)
    pareto = d.loc[keep].copy()
    # restore original index as column
    pareto = pareto.rename(columns={"index": "_orig_index"})
    return pareto


def normalize_01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx - mn, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_cols = cfg["inputs"]["composition_columns"]

    # Load COF and wear prediction tables
    df_cof = load_table(cfg["inputs"]["cof_file"])
    df_wear = load_table(cfg["inputs"]["wear_file"])

    # Merge by composition columns (exact match)
    df = pd.merge(df_cof, df_wear, on=comp_cols, how="inner", suffixes=("", ""))

    # Compute cost index
    c_cfg = cfg["cost_index"]
    cost_index, total_price, n_components = compute_cost_index(
        df, comp_cols=comp_cols,
        unit_prices=list(c_cfg["unit_prices"]),
        w_n=float(c_cfg["weight_n_components"]),
        w_price=float(c_cfg["weight_price"]),
        eps=float(c_cfg.get("count_eps", 0.0))
    )
    df["total_price"] = total_price
    df["n_components"] = n_components
    df["cost_index"] = cost_index

    # Compute tribo index
    df["tribo_index"] = compute_tribo_index(df, cfg)

    # Best by tribo and best by cost
    best_tribo_row = df.loc[df["tribo_index"].idxmax()].copy()
    best_cost_row = df.loc[df["cost_index"].idxmax()].copy()

    # Pareto front and "best value" selection
    pareto = pareto_front(df, x="tribo_index", y="cost_index")
    alpha = float(cfg["selection"]["value_score"]["alpha"])
    pareto["tribo_norm"] = normalize_01(pareto["tribo_index"])
    pareto["cost_norm"] = normalize_01(pareto["cost_index"])
    pareto["value_score"] = alpha * pareto["tribo_norm"] + (1 - alpha) * pareto["cost_norm"]
    best_value_row = pareto.loc[pareto["value_score"].idxmax()].copy()

    # Save outputs
    ranking_file = Path(cfg["output"]["ranking_file"])
    pareto_file = Path(cfg["output"]["pareto_file"])
    ranking_file.parent.mkdir(parents=True, exist_ok=True)
    pareto_file.parent.mkdir(parents=True, exist_ok=True)

    df.sort_values(by=["tribo_index", "cost_index"], ascending=False).to_excel(ranking_file, index=False)
    pareto.sort_values(by=["tribo_index", "cost_index"], ascending=False).to_excel(pareto_file, index=False)

    summary = {
        "best_tribo": best_tribo_row.to_dict(),
        "best_cost": best_cost_row.to_dict(),
        "best_value": best_value_row.to_dict(),
        "notes": {
            "best_tribo": "max tribo_index",
            "best_cost": "max cost_index",
            "best_value": "selected from Pareto front by normalized weighted score"
        }
    }
    best_summary_file = Path(cfg["output"]["best_summary_file"])
    best_summary_file.parent.mkdir(parents=True, exist_ok=True)
    best_summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved ranking: {ranking_file}")
    print(f"[OK] Saved pareto:  {pareto_file}")
    print(f"[OK] Saved summary: {best_summary_file}")


if __name__ == "__main__":
    main()
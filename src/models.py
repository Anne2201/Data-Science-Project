# src/models.py
from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

import torch
import torch.nn as nn


# =========================
# ML
# =========================
def _build_ml_dataset(df: pd.DataFrame, include_distributor: bool = True):
    d = df.copy()
    base_features = ["Budget (in $)", "Running Time", "Release_Month", "Main_Genre"]
    if include_distributor and "Distributor" in d.columns:
        base_features.append("Distributor")

    d = d.dropna(subset=["World Wide Sales (in $)", "Budget (in $)", "Running Time", "Release_Month", "Main_Genre"])

    X = pd.get_dummies(
        d[base_features],
        columns=[c for c in ["Main_Genre", "Distributor"] if c in base_features],
        drop_first=True
    )
    y = np.log1p(d["World Wide Sales (in $)"].astype(float))
    return X, y


def train_compare_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    X, y = _build_ml_dataset(df, include_distributor=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    ridge = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
    lasso = make_pipeline(StandardScaler(with_mean=False), LassoCV(cv=5, random_state=42))
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(random_state=42)
    voting = VotingRegressor(estimators=[("lr", lr), ("rf", rf), ("gb", gb)])

    models = {
        "Linear Regression": lr,
        "Ridge": ridge,
        "LassoCV": lasso,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting Ensemble": voting
    }

    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        rows.append({"Model": name, "R2": float(r2_score(y_test, pred))})

    comparison_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)

    best_name = comparison_df.iloc[0]["Model"]
    best_model = models[best_name]
    cv = cross_val_score(best_model, X, y, cv=5, scoring="r2")

    metrics = {
        "best_model": str(best_name),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1])
    }

    artifacts: Dict[str, Any] = {"rf_importance_df": None, "lr_coef_df": None}

    # RF importance
    rf.fit(X_train, y_train)
    imp = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    artifacts["rf_importance_df"] = imp.reset_index(drop=True)

    # LR coef
    lr.fit(X_train, y_train)
    coef = pd.DataFrame({"Feature": X.columns, "Coefficient": lr.coef_}).sort_values("Coefficient", ascending=False)
    artifacts["lr_coef_df"] = coef.reset_index(drop=True)

    return comparison_df, metrics, artifacts


# =========================
# LSTM
# =========================
def prepare_lstm_data(df_ts: pd.DataFrame, window_size: int = 5):
    features = ["Log_Sales", "Log_Budget", "Release_Month"]
    data = df_ts[features].fillna(0).values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i + window_size, :])
        y.append(scaled[i + window_size, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    train_size = int(len(X) * 0.8)
    split_info = {"train_size": train_size, "window_size": window_size}
    return X, y, scaler, split_info


class MovieLSTM(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(X_seq: np.ndarray, y_seq: np.ndarray, train_size: int, epochs: int = 60, lr: float = 0.01):
    device = "cpu"

    X_train = torch.tensor(X_seq[:train_size], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_seq[:train_size], dtype=torch.float32).reshape(-1, 1).to(device)
    X_test = torch.tensor(X_seq[train_size:], dtype=torch.float32).to(device)
    y_test = y_seq[train_size:].copy()

    model = MovieLSTM(input_size=X_seq.shape[-1], hidden_size=64, num_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        preds_test = model(X_test).cpu().numpy().reshape(-1)

    artifacts = {"y_test": y_test, "preds": preds_test, "losses": losses}
    return model, artifacts


# =========================
# STRATEGIC SIMULATOR (comme ton notebook, simplifié mais cohérent)
# =========================
def strategic_forecast_simulator(df_ts: pd.DataFrame, model: nn.Module):
    """
    Simule revenus (Millions $) et ROI selon mois + budget.
    On garde genre "pseudo" (4 catégories) car ton CSV peut ne pas contenir les genres.
    """
    model.eval()
    np.random.seed(42)

    # Univers simplifié (comme notebook)
    genres = ["Action/Blockbuster", "Horror/Thriller", "Sci-Fi", "Comedy/Drama"]
    months = [5, 6, 7, 11, 12]
    budgets_m = [10, 50, 100, 200]

    max_log_sales = float(df_ts["Log_Sales"].max()) if "Log_Sales" in df_ts.columns else 18.0

    results = []
    # modèle: input = (1,5,3) features: [Log_Sales, Log_Budget, Release_Month] scaled
    for g in genres:
        for m in months:
            for b in budgets_m:
                # input seq neutre
                x = torch.zeros((1, 5, 3), dtype=torch.float32)
                # log budget & month "approx" dans l’espace [0..1] car on a entraîné en MinMax
                # -> on met des valeurs plausibles
                x[0, :, 1] = float(np.clip(np.log1p(b * 1e6) / 20.0, 0, 1))
                x[0, :, 2] = float(np.clip(m / 12.0, 0, 1))

                with torch.no_grad():
                    pred_norm = float(model(x).item())

                # rescale heuristique -> millions $
                predicted_log = pred_norm * max_log_sales
                pred_rev = float(np.expm1(predicted_log))
                pred_rev_m = pred_rev / 1e6

                # boosts / penalties façon notebook
                if g == "Action/Blockbuster" and m in [6, 7]:
                    pred_rev_m *= 1.25
                if g == "Horror/Thriller" and m in [10, 11]:
                    pred_rev_m *= 1.15
                if m == 7:
                    pred_rev_m *= 0.92  # competition penalty

                # floor realism
                if pred_rev_m < b * 0.8:
                    seasonal = 1.6 if m in [7, 12] else 1.2
                    base = 2.8 if b <= 50 else 2.2
                    pred_rev_m = b * base * seasonal

                roi = pred_rev_m / b

                results.append({
                    "Genre": g,
                    "Month": m,
                    "Budget_M": b,
                    "Forecasted_Revenue_M": round(pred_rev_m, 2),
                    "ROI": round(roi, 2)
                })

    sim_df = pd.DataFrame(results)
    best_rev = sim_df.loc[sim_df["Forecasted_Revenue_M"].idxmax()].to_dict()
    best_roi = sim_df.loc[sim_df["ROI"].idxmax()].to_dict()
    return best_rev, best_roi, sim_df


def generate_diversified_majors_plan(model: nn.Module) -> pd.DataFrame:
    """
    Reproduit l’idée "Diversified Strategic Matrix: Studio Specialization (2026-2035)"
    (plan 1 config optimale par studio).
    """
    model.eval()
    np.random.seed(42)

    studio_dna = {
        "DISNEY": {"preferred": "Action/Blockbuster", "budget_range": [200, 250]},
        "WARNER": {"preferred": "Sci-Fi", "budget_range": [150, 200]},
        "UNIVERSAL": {"preferred": "Horror/Thriller", "budget_range": [30, 80]},
        "PARAMOUNT": {"preferred": "Action/Blockbuster", "budget_range": [100, 150]},
        "SONY": {"preferred": "Comedy/Drama", "budget_range": [40, 70]},
        "LIONSGATE": {"preferred": "Horror/Thriller", "budget_range": [15, 40]},
    }

    genres = ["Action/Blockbuster", "Horror/Thriller", "Sci-Fi", "Comedy/Drama"]
    months = [5, 6, 7, 11, 12]

    results = []
    for studio, dna in studio_dna.items():
        best = None

        for g in genres:
            for b in dna["budget_range"]:
                for m in months:
                    x = torch.zeros((1, 5, 3), dtype=torch.float32)
                    x[0, :, 1] = float(np.clip(np.log1p(b * 1e6) / 20.0, 0, 1))
                    x[0, :, 2] = float(np.clip(m / 12.0, 0, 1))

                    with torch.no_grad():
                        pred_norm = float(model(x).item())

                    predicted_sales = float(np.expm1(pred_norm * 18.0)) / 1e6  # millions

                    # DNA bonus
                    if g == dna["preferred"]:
                        predicted_sales *= 1.4

                    # saturation penalty
                    if g == "Action/Blockbuster" and m == 7:
                        predicted_sales *= 0.85

                    # floor
                    if predicted_sales < b * 1.5:
                        base_mult = 3.5 if dna["budget_range"][0] < 50 else 2.5
                        predicted_sales = b * (base_mult + np.sin(m) * 0.5)

                    roi = predicted_sales / b
                    row = {
                        "Studio": studio,
                        "Strategic_Genre": g,
                        "Budget_M": float(b),
                        "Release_Window": "Summer" if m in [5, 6, 7] else "Holiday",
                        "Month": int(m),
                        "Projected_ROI": round(float(roi), 2),
                        "Forecasted_Rev_M": round(float(predicted_sales), 2),
                    }

                    if best is None or row["Projected_ROI"] > best["Projected_ROI"]:
                        best = row

        results.append(best)

    return pd.DataFrame(results)
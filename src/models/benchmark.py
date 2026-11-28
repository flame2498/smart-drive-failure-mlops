"""Multi-Model Benchmarking Engine for SMART Drive Failure Prediction.

Trains and evaluates multiple model families (Logistic Regression, Random Forest,
LightGBM Baseline, and LightGBM Optuna Champion) on the exact same temporal
hold-out test set, logging comprehensive comparison metrics to MLflow.
"""

from pathlib import Path
import json
import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from src.data.features import compute_telemetry_features


def load_and_split_data(config_path: str = "params.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = Path("data/raw/backblaze_telemetry.parquet")
    if not raw_path.exists():
        raw_path = Path("data/raw/telemetry_sample.parquet")

    print(f"Loading telemetry from {raw_path}...")
    df_raw = pl.read_parquet(raw_path)

    print("Computing feature transformations...")
    df_feat = compute_telemetry_features(
        df_raw,
        prediction_horizon_days=config["data"]["prediction_horizon_days"],
        rolling_window_days=config["data"]["rolling_window_days"],
        top_models=config["data"].get("top_models"),
    )

    dates = df_feat.select("date").unique().sort("date")["date"].to_list()
    test_days = config["training"]["test_size_days"]
    val_days = config["training"]["val_size_days"]

    train_end = dates[-(test_days + val_days)]
    val_end = dates[-test_days]

    train_df = df_feat.filter(pl.col("date") <= train_end)
    val_df = df_feat.filter((pl.col("date") > train_end) & (pl.col("date") <= val_end))
    test_df = df_feat.filter(pl.col("date") > val_end)

    target_col = config["training"]["target_col"]
    feature_cols = [
        col for col in df_feat.columns if col not in ["date", "serial_number", "model", "failure", target_col]
    ]

    print(f"Split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select(target_col).to_pandas().values.ravel()

    X_val = val_df.select(feature_cols).to_pandas()
    y_val = val_df.select(target_col).to_pandas().values.ravel()

    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(target_col).to_pandas().values.ravel()

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, config


def evaluate_predictions(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

    return {
        "test_recall": rec,
        "test_pr_auc": pr_auc,
        "test_f2": f2,
        "test_f1": f1,
        "test_precision": prec,
        "test_roc_auc": roc_auc,
    }


def run_benchmark():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("smart-drive-failure-prediction")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, config = load_and_split_data()

    models_to_benchmark = {
        "Logistic_Regression_Baseline": {
            "model": make_pipeline(StandardScaler(), LogisticRegression(max_iter=100, class_weight="balanced", random_state=42)),
            "params": {"model_family": "LogisticRegression", "class_weight": "balanced", "scaler": "StandardScaler"},
            "tag": "baseline_linear",
        },
        "Random_Forest_Ensemble": {
            "model": RandomForestClassifier(
                n_estimators=40,
                max_depth=8,
                class_weight="balanced_subsample",
                max_samples=0.25,
                n_jobs=-1,
                random_state=42,
            ),
            "params": {"model_family": "RandomForest", "n_estimators": 40, "max_depth": 8, "class_weight": "balanced_subsample"},
            "tag": "baseline_bagging",
        },
        "LightGBM_Default_Baseline": {
            "model": lgb.LGBMClassifier(
                learning_rate=0.05,
                n_estimators=200,
                max_depth=6,
                num_leaves=31,
                scale_pos_weight=50.0,
                random_state=42,
                verbosity=-1,
            ),
            "params": {"model_family": "LightGBM_Default", "scale_pos_weight": 50.0, "learning_rate": 0.05, "num_leaves": 31},
            "tag": "baseline_boosting",
        },
        "LightGBM_Optuna_Champion": {
            "model": lgb.LGBMClassifier(
                learning_rate=0.1426,
                n_estimators=250,
                max_depth=9,
                num_leaves=22,
                min_child_samples=157,
                subsample=0.725,
                colsample_bytree=0.673,
                scale_pos_weight=93.25,
                random_state=42,
                verbosity=-1,
            ),
            "params": {
                "model_family": "LightGBM_Optuna_Tuned",
                "scale_pos_weight": 93.25,
                "learning_rate": 0.1426,
                "num_leaves": 22,
                "max_depth": 9,
                "min_child_samples": 157,
            },
            "tag": "production_champion",
        },
    }

    results = []

    print("\n" + "=" * 70)
    print("STARTING MULTI-MODEL MLFLOW BENCHMARK")
    print("=" * 70)

    for run_name, spec in models_to_benchmark.items():
        print(f"\n---> Training: {run_name}...")
        clf = spec["model"]

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(spec["params"])
            mlflow.log_param("num_features", len(feature_cols))
            mlflow.set_tag("model_tier", spec["tag"])

            if "LightGBM" in run_name:
                clf.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
                )
            else:
                clf.fit(X_train, y_train)

            # Predict on hold-out test set
            y_prob = clf.predict_proba(X_test)[:, 1]
            metrics = evaluate_predictions(y_test, y_prob)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            print(f"Results for {run_name}:")
            print(f"  Test Recall:    {metrics['test_recall']*100:.2f}%")
            print(f"  Test PR-AUC:    {metrics['test_pr_auc']:.4f}")
            print(f"  Test F2-Score:  {metrics['test_f2']:.4f}")
            print(f"  Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")

            # If it's the champion, save it as primary artifact
            if spec["tag"] == "production_champion":
                model_dir = Path("models")
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "lgb_model.joblib"
                joblib.dump({"model": clf, "feature_cols": feature_cols, "metrics": metrics}, model_path)
                with open(model_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                mlflow.log_artifact(str(model_path))

            results.append({"Model": run_name, **metrics})

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY LEADERBOARD")
    print("=" * 70)
    summary_df = pl.DataFrame(results).sort("test_recall", descending=True)
    print(summary_df)
    return summary_df


if __name__ == "__main__":
    run_benchmark()

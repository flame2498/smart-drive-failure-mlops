"""Model training pipeline with temporal out-of-time split and MLflow tracking."""
import json
import os
from pathlib import Path
import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
import yaml

from src.data.features import compute_telemetry_features


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials: int = 15) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators": 250,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 50.0, 400.0, log=True),
            "random_state": 42,
            "verbosity": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, y_val_prob))
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        val_f2 = float(fbeta_score(y_val, y_val_pred, beta=2, zero_division=0))
        return val_pr_auc + 0.3 * val_f2

    print(f"\nStarting Optuna Hyperparameter Optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Optuna Best Trial Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    return study.best_params


def run_training_pipeline(config_path: str = "params.yaml", tune: bool = False, n_trials: int = 15) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_file = config["data"].get("raw_file", "data/raw/backblaze_telemetry.parquet")
    raw_path = Path(raw_file)
    if not raw_path.exists():
        fallback_path = Path("data/raw/telemetry_sample.parquet")
        if fallback_path.exists():
            raw_path = fallback_path
        else:
            from src.data.generate_sample import generate_benchmark_telemetry
            print("Raw telemetry not found. Generating benchmark dataset...")
            generate_benchmark_telemetry(output_path=str(fallback_path))
            raw_path = fallback_path

    print(f"Loading raw telemetry from {raw_path}...")

    # 1. Feature Engineering with Polars
    print("Computing rolling window features...")
    df_raw = pl.read_parquet(raw_path)
    df_feat = compute_telemetry_features(
        df_raw,
        prediction_horizon_days=config["data"]["prediction_horizon_days"],
        rolling_window_days=config["data"]["rolling_window_days"],
        top_models=config["data"].get("top_models"),
    )

    # 2. Temporal Out-of-Time Train/Val/Test Split
    dates = df_feat.select("date").unique().sort("date")["date"].to_list()
    total_days = len(dates)
    test_days = config["training"]["test_size_days"]
    val_days = config["training"]["val_size_days"]

    train_end = dates[-(test_days + val_days)]
    val_end = dates[-test_days]

    train_df = df_feat.filter(pl.col("date") <= train_end)
    val_df = df_feat.filter((pl.col("date") > train_end) & (pl.col("date") <= val_end))
    test_df = df_feat.filter(pl.col("date") > val_end)

    target_col = config["training"]["target_col"]
    feature_cols = [
        col for col in df_feat.columns
        if col not in ["date", "serial_number", "model", "failure", target_col]
    ]

    print(f"Features: {feature_cols}")
    print(f"Dataset split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select(target_col).to_pandas().values.ravel()

    X_val = val_df.select(feature_cols).to_pandas()
    y_val = val_df.select(target_col).to_pandas().values.ravel()

    X_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select(target_col).to_pandas().values.ravel()

    # Save reference features for Evidently drift monitoring
    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(ref_dir / "reference_features.parquet")

    # 3. Optional Optuna Hyperparameter Optimization
    train_params = dict(config["training"])
    if tune or config["training"].get("tune", False):
        best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=n_trials)
        train_params.update(best_params)

    # 4. Train LightGBM with MLflow Tracking
    mlflow.set_experiment("smart-drive-failure-prediction")
    with mlflow.start_run():
        mlflow.log_params(train_params)
        mlflow.log_param("num_features", len(feature_cols))

        scale_pos_weight = train_params.get("scale_pos_weight", 50.0)
        clf = lgb.LGBMClassifier(
            learning_rate=train_params.get("learning_rate", 0.05),
            n_estimators=train_params.get("n_estimators", 300),
            max_depth=train_params.get("max_depth", 6),
            num_leaves=train_params.get("num_leaves", 31),
            subsample=train_params.get("subsample", 0.8),
            colsample_bytree=train_params.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=train_params.get("random_state", 42),
            verbosity=-1,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # 5. Evaluation on Hold-Out Test Set
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        pr_auc = float(average_precision_score(y_test, y_prob))
        roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_test, y_pred, beta=2, zero_division=0))

        metrics = {
            "test_pr_auc": pr_auc,
            "test_roc_auc": roc_auc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_f2": f2,
        }

        print("\nTest Evaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            mlflow.log_metric(k, v)

        # Save model artifact
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "lgb_model.joblib"
        joblib.dump({"model": clf, "feature_cols": feature_cols, "metrics": metrics}, model_path)
        
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(model_path))
        print(f"Model saved to {model_path}")

        return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to config yaml")
    args = parser.parse_args()
    run_training_pipeline(config_path=args.config, tune=args.tune, n_trials=args.trials)

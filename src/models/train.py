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


def run_training_pipeline(config_path: str = "params.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = Path("data/raw/telemetry_sample.parquet")
    if not raw_path.exists():
        from src.data.generate_sample import generate_benchmark_telemetry
        print("Raw telemetry not found. Generating benchmark dataset...")
        generate_benchmark_telemetry(output_path=str(raw_path))

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

    # 3. Train LightGBM with MLflow Tracking
    mlflow.set_experiment("smart-drive-failure-prediction")
    with mlflow.start_run():
        mlflow.log_params(config["training"])
        mlflow.log_param("num_features", len(feature_cols))

        scale_pos_weight = config["training"].get("scale_pos_weight", 50.0)
        clf = lgb.LGBMClassifier(
            learning_rate=config["training"]["learning_rate"],
            n_estimators=config["training"]["n_estimators"],
            max_depth=config["training"]["max_depth"],
            num_leaves=config["training"]["num_leaves"],
            subsample=config["training"]["subsample"],
            colsample_bytree=config["training"]["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            random_state=config["training"]["random_state"],
            verbosity=-1,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # 4. Evaluation on Hold-Out Test Set
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
    run_training_pipeline()

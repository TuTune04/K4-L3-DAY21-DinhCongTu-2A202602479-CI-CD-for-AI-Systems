import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    positive_rate = float((y_train == 1).mean())
    if abs(positive_rate - REFERENCE_POSITIVE_RATE) > DRIFT_TOLERANCE:
        print(
            f"WARNING: ty le lop duong tap train = {positive_rate:.4f}, "
            f"lech qua 5 diem % so voi tham chieu {REFERENCE_POSITIVE_RATE:.3f}"
        )
    else:
        print(f"INFO: ty le lop duong tap train = {positive_rate:.4f}")
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("positive_rate", positive_rate)
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))
        proba = model.predict_proba(X_eval)[:, 1]
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.90 + 1e-9, 0.05)]
        threshold_f1s = [
            float(f1_score(y_eval, (proba >= t).astype(int), zero_division=0))
            for t in thresholds
        ]
        best_index = max(range(len(thresholds)), key=threshold_f1s.__getitem__)
        best_threshold = float(thresholds[best_index])
        best_threshold_f1 = float(threshold_f1s[best_index])
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_threshold_f1)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        os.makedirs("outputs", exist_ok=True)
        matrix = confusion_matrix(y_eval, preds, labels=[0, 1])
        precision = precision_score(y_eval, preds, labels=[0, 1], average=None, zero_division=0)
        recall = recall_score(y_eval, preds, labels=[0, 1], average=None, zero_division=0)
        with open("outputs/detail.txt", "w") as f:
            f.write("Confusion matrix: rows = actual, columns = predicted\n")
            f.write("Classes: 0 = thu_nhap_thap, 1 = thu_nhap_cao\n")
            f.write("Actual / Predicted      0      1\n")
            for label, row in enumerate(matrix):
                f.write(f"{label}                  {row[0]:6d} {row[1]:6d}\n")
            for label, name in enumerate(["thu_nhap_thap", "thu_nhap_cao"]):
                f.write(
                    f"Class {label} = {name}: precision = {precision[label]:.4f}, "
                    f"recall = {recall[label]:.4f}\n"
                )
            f.write(f"Default threshold (0.50): F1 = {f1:.4f}, accuracy = {acc:.4f}\n")
            f.write(f"Best threshold = {best_threshold:.2f}, F1 = {best_threshold_f1:.4f}\n")
        mlflow.log_artifact("outputs/detail.txt")
        report = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_threshold,
            "best_threshold_f1": best_threshold_f1,
            "positive_rate": positive_rate,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)

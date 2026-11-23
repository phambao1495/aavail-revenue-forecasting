import os
import csv
import uuid
from datetime import datetime


def update_train_log(tag, period, rmse, runtime, model_version, model_version_note, log_dir="logs", test=False):
    """Ghi nhật ký quá trình huấn luyện mô hình."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train-test.log" if test else "train.log")

    header = [
        "unique_id", "timestamp", "tag", "period",
        "rmse", "runtime", "model_version", "model_version_note"
    ]
    write_header = not os.path.exists(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            str(uuid.uuid4()),
            datetime.now().isoformat(),
            tag,
            period,
            round(float(rmse), 4),
            runtime,
            model_version,
            model_version_note,
        ])


def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, log_dir="logs", test=False):
    """Ghi nhật ký mỗi lần API thực hiện dự đoán (giám sát hiệu năng production)."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "predict-test.log" if test else "predict.log")

    header = [
        "unique_id", "timestamp", "country", "y_pred",
        "y_proba", "target_date", "runtime", "model_version"
    ]
    write_header = not os.path.exists(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            str(uuid.uuid4()),
            datetime.now().isoformat(),
            country,
            round(float(y_pred), 2),
            y_proba,
            target_date,
            runtime,
            model_version,
        ])
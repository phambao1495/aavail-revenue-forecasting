import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

from src.data_ingestion import fetch_ts
from src.logger import update_train_log, update_predict_log

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "Supervised time-series forecasting with lag features"


def engineer_features(df, training=True):
    """Tạo đặc trưng trễ (lag features) để dự đoán tổng doanh thu 30 ngày tiếp theo."""
    dates = df["date"].values
    eng_features = {"previous_7": [], "previous_14": [], "previous_28": [], "previous_70": [], "recent_views": []}
    y = []
    valid_dates = []

    for d in dates:
        current = pd.to_datetime(d)
        # Lấy các mốc thời gian quá khứ
        d_7 = current - pd.Timedelta(days=7)
        d_14 = current - pd.Timedelta(days=14)
        d_28 = current - pd.Timedelta(days=28)
        d_70 = current - pd.Timedelta(days=70)
        d_next_30 = current + pd.Timedelta(days=30)

        # Chỉ giữ các điểm có đủ 70 ngày lịch sử (và đủ 30 ngày tương lai nếu đang train)
        if d_70 < df["date"].min():
            continue
        if training and d_next_30 > df["date"].max():
            continue

        eng_features["previous_7"].append(df[(df["date"] >= d_7) & (df["date"] < current)]["revenue"].sum())
        eng_features["previous_14"].append(df[(df["date"] >= d_14) & (df["date"] < current)]["revenue"].sum())
        eng_features["previous_28"].append(df[(df["date"] >= d_28) & (df["date"] < current)]["revenue"].sum())
        eng_features["previous_70"].append(df[(df["date"] >= d_70) & (df["date"] < current)]["revenue"].sum())
        eng_features["recent_views"].append(df[(df["date"] >= d_28) & (df["date"] < current)]["total_views"].sum())
        valid_dates.append(current)

        if training:
            y.append(df[(df["date"] >= current) & (df["date"] < d_next_30)]["revenue"].sum())

    X = pd.DataFrame(eng_features)
    if training:
        return X, np.array(y), np.array(valid_dates)
    return X, np.array(valid_dates)


def _model_train(df, tag, model_dir="models", log_dir="logs", test=False):
    """Huấn luyện và so sánh nhiều mô hình (Baseline vs RF vs GB), lưu mô hình tốt nhất."""
    start_time = time.time()
    X, y, dates = engineer_features(df, training=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Định nghĩa các mô hình ứng viên (Ridge đóng vai trò Baseline tuyến tính)
    candidates = {
        "Baseline_Ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor(n_estimators=100, random_state=42))]),
        "GradientBoosting": Pipeline([("scaler", StandardScaler()), ("reg", GradientBoostingRegressor(n_estimators=100, random_state=42))]),
    }

    results = {}
    best_model_name = None
    best_model = None
    best_rmse = float("inf")

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        results[name] = round(float(rmse), 2)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = pipe

    # Huấn luyện lại mô hình tốt nhất trên toàn bộ dữ liệu
    best_model.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    prefix = "test-" if test else "sl-"
    model_path = os.path.join(model_dir, f"{prefix}{tag}-{re.sub(r'\\.', '_', str(MODEL_VERSION))}.joblib")

    # Lưu kèm phân phối dữ liệu huấn luyện để giám sát drift/novelty
    artifact = {
        "model": best_model,
        "best_model_name": best_model_name,
        "comparison_rmse": results,
        "feature_means": X.mean().to_dict(),
        "feature_stds": X.std().to_dict(),
    }
    joblib.dump(artifact, model_path)

    m, s = divmod(time.time() - start_time, 60)
    runtime = f"{int(m):03d}:{int(s):02d}"
    period = f"{pd.to_datetime(dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}"
    update_train_log(tag, period, best_rmse, runtime, MODEL_VERSION, f"{best_model_name} | {MODEL_VERSION_NOTE}", log_dir=log_dir, test=test)

    return results


def model_train(data_dir="data", model_dir="models", log_dir="logs", test=False):
    """Huấn luyện mô hình cho 'all' và các quốc gia hàng đầu."""
    ts_dict = fetch_ts(data_dir=data_dir, save_dir=model_dir)
    all_comparisons = {}

    # Nếu đang chạy unit test, chỉ train 2 nhóm ('all' và 'united_kingdom') cho nhanh
    keys_to_train = ["all", "united_kingdom"] if test else list(ts_dict.keys())

    for tag in keys_to_train:
        if tag in ts_dict:
            all_comparisons[tag] = _model_train(ts_dict[tag], tag, model_dir=model_dir, log_dir=log_dir, test=test)

    return all_comparisons


def model_load(prefix="sl", data_dir="data", model_dir="models"):
    """Tải toàn bộ mô hình đã lưu và dữ liệu chuỗi thời gian."""
    models = {}
    for f in os.listdir(model_dir):
        if f.startswith(prefix) and f.endswith(".joblib"):
            tag = f.split("-")[1]
            models[tag] = joblib.load(os.path.join(model_dir, f))

    ts_dict = fetch_ts(data_dir=data_dir, save_dir=model_dir)
    return ts_dict, models


def model_predict(country, year, month, day, all_models=None, all_data=None, model_dir="models", log_dir="logs", test=False):
    """Dự đoán doanh thu 30 ngày tiếp theo cho quốc gia và ngày mục tiêu."""
    start_time = time.time()
    prefix = "test" if test else "sl"

    if all_models is None or all_data is None:
        all_data, all_models = model_load(prefix=prefix, model_dir=model_dir)

    tag = country.replace(" ", "_").lower()
    if tag not in all_models:
        raise ValueError(f"Không tìm thấy mô hình cho quốc gia '{country}'. Các lựa chọn hợp lệ: {list(all_models.keys())}")

    target_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    target_dt = pd.to_datetime(target_date)

    df = all_data[tag]
    if target_dt not in set(df["date"]):
        raise ValueError(f"Ngày {target_date} nằm ngoài phạm vi dữ liệu ({df['date'].min().date()} -> {df['date'].max().date()}).")

    X, dates = engineer_features(df, training=False)
    idx = np.where(dates == np.datetime64(target_dt))[0]
    if len(idx) == 0:
        raise ValueError(f"Không đủ 70 ngày lịch sử trước ngày {target_date} để tạo đặc trưng.")

    query = X.iloc[[idx[0]]]
    artifact = all_models[tag]
    model = artifact["model"]
    y_pred = model.predict(query)[0]

    # Giám sát bất thường (Novelty/Outlier check dựa trên Z-score của previous_28)
    mean_28 = artifact["feature_means"]["previous_28"]
    std_28 = artifact["feature_stds"]["previous_28"] or 1.0
    z_score = abs(query["previous_28"].values[0] - mean_28) / std_28
    drift_warning = bool(z_score > 3.0)

    m, s = divmod(time.time() - start_time, 60)
    runtime = f"{int(m):03d}:{int(s):02d}"
    update_predict_log(tag, y_pred, None, target_date, runtime, MODEL_VERSION, log_dir=log_dir, test=test)

    return {
        "country": tag,
        "target_date": target_date,
        "y_pred": round(float(y_pred), 2),
        "model_used": artifact["best_model_name"],
        "drift_warning": drift_warning,
    }


if __name__ == "__main__":
    print("Đang huấn luyện và so sánh các mô hình...")
    comparisons = model_train(data_dir="data", model_dir="models", log_dir="logs", test=False)
    for k, v in comparisons.items():
        print(f"[{k}] RMSE các mô hình: {v}")
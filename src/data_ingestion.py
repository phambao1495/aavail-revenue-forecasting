import os
import re
import glob
import pandas as pd
import numpy as np


def fetch_data(data_dir="data"):
    """Đọc toàn bộ file JSON giao dịch và gộp thành một DataFrame chuẩn hóa."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục dữ liệu '{data_dir}' không tồn tại.")

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not file_list:
        raise ValueError(f"Không tìm thấy file JSON nào trong '{data_dir}'.")

    df_list = []
    # Chuẩn hóa tên cột không đồng nhất giữa các file JSON của AAVAIL
    col_map = {
        "StreamID": "stream_id",
        "TimesViewed": "times_viewed",
        "total_price": "price",
    }

    for file_path in file_list:
        df_temp = pd.read_json(file_path)
        df_temp.rename(columns=col_map, inplace=True)
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True, sort=False)

    # Tạo cột ngày chuẩn (invoice_date) từ year, month, day
    df["invoice_date"] = pd.to_datetime(
        df[["year", "month", "day"]].astype(str).agg("-".join, axis=1),
        errors="coerce",
    )
    df.dropna(subset=["invoice_date"], inplace=True)

    # Làm sạch chuỗi trong cột invoice (bỏ chữ cái đầu nếu có)
    if "invoice" in df.columns:
        df["invoice"] = df["invoice"].astype(str).apply(lambda x: re.sub(r"\D+", "", x))

    return df


def convert_to_ts(df_orig, country=None):
    """Chuyển đổi dữ liệu giao dịch thành chuỗi thời gian (time-series) liên tục theo ngày."""
    if country and country.lower() != "all":
        df = df_orig[df_orig["country"] == country].copy()
    else:
        df = df_orig.copy()

    if df.empty:
        raise ValueError(f"Không có dữ liệu cho quốc gia: {country}")

    # Tạo dải ngày liên tục từ ngày đầu tiên đến ngày cuối cùng
    start_date = df["invoice_date"].min()
    end_date = df["invoice_date"].max()
    all_days = pd.date_range(start=start_date, end=end_date, freq="D")

    # Tổng hợp theo ngày: doanh thu, số lượng hóa đơn, tổng lượt xem
    daily_agg = (
        df.groupby("invoice_date")
        .agg(
            revenue=("price", "sum"),
            purchases=("invoice", "nunique"),
            total_views=("times_viewed", "sum"),
        )
        .reindex(all_days, fill_value=0)
    )

    daily_agg.index.name = "date"
    daily_agg.reset_index(inplace=True)
    daily_agg["year_month"] = daily_agg["date"].dt.to_period("M").astype(str)

    return daily_agg


def fetch_ts(data_dir="data", clean=False, save_dir="models"):
    """Hàm tự động hóa toàn bộ quy trình ingest dữ liệu cho 'all' và top 10 quốc gia."""
    os.makedirs(save_dir, exist_ok=True)
    ts_files = glob.glob(os.path.join(save_dir, "ts-*.csv"))

    if ts_files and not clean:
        ts_dict = {}
        for f in ts_files:
            key = os.path.basename(f).replace("ts-", "").replace(".csv", "")
            ts_dict[key] = pd.read_csv(f, parse_dates=["date"])
        return ts_dict

    df = fetch_data(data_dir)
    # Lấy top 10 quốc gia có doanh thu cao nhất + toàn bộ ('all')
    top_countries = (
        df.groupby("country")["price"].sum().nlargest(10).index.tolist()
    )

    ts_dict = {"all": convert_to_ts(df, country="all")}
    for c in top_countries:
        key = c.replace(" ", "_").lower()
        ts_dict[key] = convert_to_ts(df, country=c)

    for key, ts_df in ts_dict.items():
        ts_df.to_csv(os.path.join(save_dir, f"ts-{key}.csv"), index=False)

    return ts_dict


if __name__ == "__main__":
    print("Đang chạy quá trình thu thập dữ liệu...")
    data_dict = fetch_ts(data_dir="data", clean=True)
    print(f"Hoàn tất! Đã tạo chuỗi thời gian cho các nhóm: {list(data_dict.keys())}")
    print(data_dict["all"].head())
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Thêm thư mục gốc vào sys.path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import fetch_data, fetch_ts
from src.model import model_train

sns.set_theme(style="whitegrid")
os.makedirs("notebooks/plots", exist_ok=True)


def create_eda_and_comparison_plots():
    # 1. Biểu đồ EDA 1: Tổng doanh thu theo Top 10 Quốc gia
    df = fetch_data("data")
    top_countries = df.groupby("country")["price"].sum().nlargest(10).reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_countries, x="price", y="country", hue="country", palette="viridis", legend=False)
    plt.title("EDA: Top 10 Countries by Total Revenue", fontsize=14, fontweight="bold")
    plt.xlabel("Total Revenue")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig("notebooks/plots/eda_revenue_by_country.png", dpi=150)
    plt.close()

    # 2. Biểu đồ EDA 2: Chuỗi thời gian doanh thu theo ngày (All countries)
    ts_dict = fetch_ts("data", save_dir="models")
    df_all = ts_dict["all"]

    plt.figure(figsize=(12, 5))
    plt.plot(df_all["date"], df_all["revenue"], color="#1f77b4", linewidth=1.2)
    plt.title("EDA: Daily Revenue Time-Series (All Countries)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Daily Revenue")
    plt.tight_layout()
    plt.savefig("notebooks/plots/eda_daily_revenue_ts.png", dpi=150)
    plt.close()

    # 3. Biểu đồ so sánh các mô hình với Baseline (RMSE)
    comparisons = model_train(data_dir="data", model_dir="models", log_dir="logs", test=False)
    rmse_all = comparisons["all"]

    comp_df = pd.DataFrame({
        "Model": list(rmse_all.keys()),
        "RMSE": list(rmse_all.values())
    })

    plt.figure(figsize=(8, 5))
    bars = sns.barplot(data=comp_df, x="Model", y="RMSE", hue="Model", palette=["#d9534f", "#5bc0de", "#5cb85c"], legend=False)
    plt.title("Model Comparison vs. Baseline (Lower RMSE is Better)", fontsize=14, fontweight="bold")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.xlabel("Model Architecture")

    for bar in bars.patches:
        plt.annotate(
            f"{bar.get_height():,.0f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig("notebooks/plots/model_vs_baseline_comparison.png", dpi=150)
    plt.close()
    print("Đã xuất thành công 3 biểu đồ vào thư mục notebooks/plots/!")


if __name__ == "__main__":
    create_eda_and_comparison_plots()
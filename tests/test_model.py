import os
import tempfile
import unittest
from src.model import model_train, model_load, model_predict


class TestModel(unittest.TestCase):
    def test_train_and_predict_isolated(self):
        with tempfile.TemporaryDirectory() as tmp_model_dir, tempfile.TemporaryDirectory() as tmp_log_dir:
            # 1. Kiểm tra huấn luyện trong môi trường cô lập
            metrics = model_train(
                data_dir="data",
                model_dir=tmp_model_dir,
                log_dir=tmp_log_dir,
                test=True,
            )
            self.assertIn("all", metrics)
            self.assertIn("Baseline_Ridge", metrics["all"])

            # 2. Kiểm tra tải mô hình
            all_data, all_models = model_load(
                prefix="test",
                data_dir="data",
                model_dir=tmp_model_dir,
            )
            self.assertIn("all", all_models)

            # 3. Kiểm tra dự đoán với ngày hợp lệ (lấy ngày gần cuối trong tập dữ liệu)
            sample_date = all_data["all"]["date"].iloc[-35]
            res = model_predict(
                country="all",
                year=sample_date.year,
                month=sample_date.month,
                day=sample_date.day,
                all_models=all_models,
                all_data=all_data,
                model_dir=tmp_model_dir,
                log_dir=tmp_log_dir,
                test=True,
            )
            self.assertIn("y_pred", res)
            self.assertIsInstance(res["y_pred"], float)


if __name__ == "__main__":
    unittest.main()
import os
import csv
import tempfile
import unittest
from src.logger import update_train_log, update_predict_log


class TestLogger(unittest.TestCase):
    def test_train_log_isolated(self):
        with tempfile.TemporaryDirectory() as tmp_log_dir:
            update_train_log(
                tag="all",
                period="2018-01-01 to 2019-06-01",
                rmse=12345.67,
                runtime="000:02",
                model_version=0.1,
                model_version_note="unit test",
                log_dir=tmp_log_dir,
                test=True,
            )
            log_path = os.path.join(tmp_log_dir, "train-test.log")
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 2)  # Header + 1 dòng log
            self.assertEqual(rows[1][2], "all")

    def test_predict_log_isolated(self):
        with tempfile.TemporaryDirectory() as tmp_log_dir:
            update_predict_log(
                country="united_kingdom",
                y_pred=54321.0,
                y_proba=None,
                target_date="2019-05-01",
                runtime="000:01",
                model_version=0.1,
                log_dir=tmp_log_dir,
                test=True,
            )
            log_path = os.path.join(tmp_log_dir, "predict-test.log")
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][2], "united_kingdom")


if __name__ == "__main__":
    unittest.main()
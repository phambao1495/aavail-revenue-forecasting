import tempfile
import unittest
from app import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_health_check(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")

    def test_predict_missing_fields(self):
        response = self.client.post("/predict", json={"country": "all"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_train_and_predict_endpoints_isolated(self):
        with tempfile.TemporaryDirectory() as tmp_model_dir, tempfile.TemporaryDirectory() as tmp_log_dir:
            # Gọi POST /train ở chế độ test cô lập
            train_resp = self.client.post("/train", json={
                "test": True,
                "data_dir": "data",
                "model_dir": tmp_model_dir,
                "log_dir": tmp_log_dir,
            })
            self.assertEqual(train_resp.status_code, 200)

            # Gọi POST /predict cho cả 'all' và một quốc gia cụ thể ('United Kingdom')
            for target_country in ["all", "United Kingdom"]:
                pred_resp = self.client.post("/predict", json={
                    "country": target_country,
                    "year": "2019",
                    "month": "05",
                    "day": "01",
                    "test": True,
                    "model_dir": tmp_model_dir,
                    "log_dir": tmp_log_dir,
                })
                self.assertEqual(pred_resp.status_code, 200)
                body = pred_resp.get_json()
                self.assertIn("y_pred", body)


if __name__ == "__main__":
    unittest.main()
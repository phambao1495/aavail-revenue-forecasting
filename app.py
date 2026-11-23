import os
from flask import Flask, request, jsonify
from src.model import model_train, model_predict, model_load

app = Flask(__name__)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "AAVAIL Revenue Forecasting API",
        "endpoints": ["/predict (POST)", "/train (POST)"]
    }), 200


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    test_mode = bool(data.get("test", False))
    data_dir = data.get("data_dir", "data")
    model_dir = data.get("model_dir", "models")
    log_dir = data.get("log_dir", "logs")

    try:
        results = model_train(
            data_dir=data_dir,
            model_dir=model_dir,
            log_dir=log_dir,
            test=test_mode
        )
        return jsonify({"status": "success", "metrics": results}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    # Kiểm tra các trường đầu vào bắt buộc (target date và country)
    required_fields = ["country", "year", "month", "day"]
    missing = [f for f in required_fields if f not in data or str(data[f]).strip() == ""]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    country = str(data["country"]).strip()
    year = str(data["year"]).strip()
    month = str(data["month"]).strip()
    day = str(data["day"]).strip()

    test_mode = bool(data.get("test", False))
    model_dir = data.get("model_dir", "models")
    log_dir = data.get("log_dir", "logs")

    try:
        result = model_predict(
            country=country,
            year=year,
            month=month,
            day=day,
            model_dir=model_dir,
            log_dir=log_dir,
            test=test_mode
        )
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
try:
    with open("housing_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    model = None
    print(f"Model not found: {e}")
    
# Load scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    scaler = None
    print(f"Scaler not found: {e}")


@app.route("/")
def home():
    return "Housing Price Prediction API is Running"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded correctly"}), 500

    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data["features"]

        # Validate structure
        if not isinstance(features, list) or not all(isinstance(f, list) for f in features):
            return jsonify({"error": "'features' must be a list of lists"}), 400

        expected_length = 12
        for idx, row in enumerate(features):
            if len(row) != expected_length:
                return jsonify({
                    "error": f"Input at index {idx} does not have exactly {expected_length} values"
                }), 400
            if not all(isinstance(x, (float, int)) for x in row):
                return jsonify({
                    "error": f"All values at index {idx} must be numeric"
                }), 400

        # Convert and scale
        input_array = np.array(features)
        scaled_input = scaler.transform(input_array)

        # Predict
        predictions = model.predict(scaled_input)

        return jsonify({
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)

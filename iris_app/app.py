from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Model file not found!")


@app.route("/")
def home():
    return "ML Model is Running"


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded correctly"}), 500

    try:
        data = request.get_json()

        # Check if "features" key exists
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data["features"]

        # Check if features is a list of lists
        if not isinstance(features, list) or not all(isinstance(f, list) for f in features):
            return jsonify({"error": "'features' must be a list of lists"}), 400

        # Check each input has exactly 4 float values
        for idx, row in enumerate(features):
            if len(row) != 4:
                return jsonify({"error": f"Input at index {idx} does not have exactly 4 values"}), 400
            if not all(isinstance(x, (float, int)) for x in row):
                return jsonify({"error": f"All values at index {idx} must be numbers"}), 400

        input_features = np.array(features)

        predictions = model.predict(input_features)
        confidences = np.max(model.predict_proba(input_features), axis=1)

        return jsonify({
            "predictions": predictions.tolist(),
            "confidences": confidences.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)

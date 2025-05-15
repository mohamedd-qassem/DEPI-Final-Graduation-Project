from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/RF_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        # Extract form inputs
        age = int(data.get("age", 0))
        tumor_size = float(data.get("tumor_size", 0))
        inv_nodes = float(data.get("inv_nodes", 0))
        breast_quadrant = data.get("breast_quadrant")

        # Handle breast quadrant safely
        breast_quadrant_mapping = {
            "Lower inner": [1, 0, 0, 0],
            "Lower outer": [0, 1, 0, 0],
            "Upper inner": [0, 0, 1, 0],
            "Upper outer": [0, 0, 0, 1],
        }
        quadrant_values = breast_quadrant_mapping.get(breast_quadrant, [1, 0, 0, 0])

        # Convert categorical fields safely
        history_no = 1 if data.get("history") == "No" else 0
        history_yes = 1 if data.get("history") == "Yes" else 0
        metastasis_no = 1 if data.get("metastasis") == "No" else 0
        metastasis_yes = 1 if data.get("metastasis") == "Yes" else 0
        breast_left = 1 if data.get("breast") == "Left" else 0
        breast_right = 1 if data.get("breast") == "Right" else 0
        menopause_no = 1 if data.get("menopause") == "No" else 0
        menopause_yes = 1 if data.get("menopause") == "Yes" else 0

        # Model feature array
        features = [
            age, tumor_size, inv_nodes,
            *quadrant_values,
            history_no, history_yes,
            metastasis_no, metastasis_yes,
            breast_left, breast_right,
            menopause_no, menopause_yes
        ]

        # Convert to NumPy array for prediction
        input_data = np.array([features])

        # Make a prediction
        prediction = model.predict(input_data)
        result = "High Risk" if prediction[0] == 1 else "Low Risk"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin.html")
def admin():
    return render_template("admin.html")

if __name__ == "__main__":
    app.run(debug=True)

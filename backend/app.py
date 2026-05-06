from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Fake Email Detection API is running ✔"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        email_text = data["text"]

        # preprocess (same as training)
        email_text = email_text.replace("Subject:", "")

        # transform
        vect_text = vectorizer.transform([email_text])

        # predict
        prediction = model.predict(vect_text)[0]

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
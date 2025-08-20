from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)

# Load model & tools
model = tf.keras.models.load_model("lstm_multilabel_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0]
    labels = mlb.inverse_transform((pred > 0.5).astype(int))

    result = ", ".join(labels[0]) if labels[0] else "Not Cyberbullying"
    confidence = f"{np.max(pred) * 100:.2f}%"

    return jsonify({"prediction": result, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)

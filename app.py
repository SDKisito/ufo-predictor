import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("ufo-model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input
    seconds = request.form['seconds']
    latitude = request.form['latitude']
    longitude = request.form['longitude']

    # Prepare the input features
    int_features = [int(seconds), float(latitude), float(longitude)]
    final_features = [np.array(int_features)]

    # Make prediction
    prediction = model.predict(final_features)

    output = prediction[0]
    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template("index.html", prediction_text=f"Likely country: {countries[output]}")

if __name__ == "__main__":
    app.run(debug=True)

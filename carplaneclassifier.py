from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)

# Load the model once
new_model = load_model('models/imageclassifier.h5')


# Function to run the model on the image
def model(image):
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return "Unable to process image!"

    # Resize image to match model input
    resizedImg = tf.image.resize(img, (256, 256))

    # Predict using the model
    results = new_model.predict(np.expand_dims(resizedImg / 255, 0))

    # Assuming the output is a probability between 0 and 1
    if results[0] > 0.5:  # If the probability for "plane" is greater than 0.5
        return "plane"
    else:
        return "car"


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        imageFile = request.files.get('imagefile')
        if imageFile and imageFile.filename:
            imageData = imageFile.read()
            prediction = model(imageData)

            return redirect(url_for("result", results=prediction))

        return redirect(url_for("result", results="not found!"))
    else:
        return render_template("index.html")


@app.route("/results")
def result():
    results = request.args.get("results", "No result provided")
    return render_template("results.html", content=results)


if __name__ == "__main__":
    app.run(debug=True)  # Use debug mode for better error tracking

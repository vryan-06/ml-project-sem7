from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
curr_loc = os.path.dirname(os.path.realpath(__file__))
model = load_model("SignLanguage_recognition_inceptionv3.h5")
print("Model has been loaded")

class_index_to_value = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

predicted_value="-"
@app.route("/")
def index():
    return "<h1>Sign Language Recognition</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    if model:
        try:
            image_file = request.files.get("image")

            if image_file is None:
                return jsonify({"error": "No image provided"})

            # Loading and Pre0Processing
            img = Image.open(image_file)
            # print(img.mode)
            img = img.convert("RGB")
            # print(img.mode)
            img = img.resize((224, 224))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Rescale the image (similar to the training data)

            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            predicted_value = class_index_to_value[class_index]
            print("Predicted Sign is : ", predicted_value)

            return jsonify({"predicted": predicted_value})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Model error"})

if __name__ == "__main__":
    app.run(debug=True)

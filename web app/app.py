from flask import Flask, render_template, request
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# load the model
model = load_model("cnn_model")


@app.route('/', methods=['GET'])
def start():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    # get the image
    img = image.load_img(image_path, target_size=(200, 200))

    # preprocessing
    img = image.img_to_array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)  # as model expect a bunch of images
    img = img / 255

    # prediction
    pred = model.predict(img)

    # output
    possibility = round((pred[0][0]) * 100, 2)
    classes = ["Normal", "Pneumonia"]
    result = [1 if element > 0.5 else 0 for element in pred][0]
    result = classes[result]

    return render_template('result.html', prediction=possibility, result=result, img_path=image_path)


if __name__ == '__main__':
    app.run(port=3000, debug=True)

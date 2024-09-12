import os
import requests
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = os.path.join(os.getcwd(), 'static')

logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        logging.debug(f"Current working directory: {os.getcwd()}")
        
        image = request.files["file"]
        if image.filename == '':
            logging.debug("Filename is invalid")
            return redirect(request.url)

        filename = secure_filename(image.filename)
        logging.debug(f"Filename received: {filename}")

        basedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
        image.save(img_path)
        logging.debug(f"Image saved to: {img_path}")

        message = ""
        prediction = None  # define prediction variable here

        try:
            res = requests.post("http://torchserve-mar:8080/predictions/mnist", files={'data': open(img_path, 'rb')})
            logging.debug(f"Status code: {res.status_code}")
            if res.status_code == 200:
                response_json = res.json()
                if isinstance(response_json, int):  # check if response_json is an integer
                    prediction = response_json  # assign the integer value to prediction
                else:
                    prediction = response_json['prediction']  # assign the value from the 'prediction' key to prediction
                logging.debug(f"Response content: {prediction}")
                
                # Map the prediction to the corresponding MNIST digit
                digit_map = {
                    0: "Zero",
                    1: "One",
                    2: "Two",
                    3: "Three",
                    4: "Four",
                    5: "Five",
                    6: "Six",
                    7: "Seven",
                    8: "Eight",
                    9: "Nine"
                }
                message = digit_map.get(prediction, "Unknown digit")
            else:
                logging.error("Failed to get response from TorchServe model")
                message = "Error: Failed to get response"
        except Exception as e:
            logging.error(f"Exception occurred: {str(e)}")
            message = f"Error: {str(e)}"
            
        return render_template('index.html', prediction_text=message) 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
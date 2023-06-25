import base64
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

from util import classify_image, load_saved_artifacts

from response import CustomResponse



app = Flask(__name__)

CORS(app)


app.debug = True


load_saved_artifacts()



@app.route("/", methods=["GET"])
def home():
    return {"message": "Welcome to the classifier"}



@app.route("/classify_image", methods=["GET", "POST"])
def classify_image_route():

    if request.method == "POST":

        image_data = request.files.get("img")

        encoded_image = base64.b64encode(image_data.read()).decode('utf-8')

        if image_data is None:
            return CustomResponse("Image not Founds", status=400)
        
        result = classify_image(encoded_image)

        if result == []:
            return CustomResponse("Could not properly detect Face and eyes, please use different picture", success=False)
            
        return CustomResponse("Image Classified Sucessfully", data=result[0])


    return jsonify({"message" : "Welcome to the classifier"})





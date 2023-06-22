from flask import Flask, request, Response





app = Flask(__name__)


@app.route("/classify_image", methods=["GET", "POST"])
def home():
    print(request)
    return {"wisdom": "yeh"}



if __name__  == "__main__":
    app.run(port=8000, debug=True)
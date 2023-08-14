from pathlib import Path
from flask import Flask, jsonify, request
from infrastructure.FaceEncoder import FaceEncoder as FaceEncoderClass
from infrastructure.FaceDetector import FaceDetector as FaceDetectorClass

FaceDetector = FaceDetectorClass()
FaceEncoder = FaceEncoderClass()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

@app.route('/encode', methods=['GET', 'POST'])
def encode():
    """
    Encodes the faces contained in ./training/{label}/img_{num}.jpeg
    and stores them in a pickle file in ./output.
    Only needs to be run when the initial training set is created or modified.
    """
    model = "hog"
    try:
        FaceEncoder.encode_known_faces()
        return jsonify({"message": "Encoding complete"})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/recognise', methods=['POST'])
def recognise():
    """
    TODO: create request schema validation
    Load unknown files and classify them using the encoding created from
    encode_known_faces.
    """
    img = request.files['image']
    try:
        names, image = FaceDetector.recognise_faces(image_file=img)
        return jsonify({"names": names, "authenticated": True})
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)